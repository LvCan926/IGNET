import pandas as pd
import torch
import numpy as np
import os

import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import random
import time

from utils import load_config_test, set_seeds
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics
from traj_process_mul import run_inference, evaluate_metrics
from models.TP.dual_modal_flow import DualModalFlowPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练轨迹流量GAN模型"
    )
    # 与test.py保持一致的基本参数
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "tune"], default="train"
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--save_model", action="store_true", help="save model")
    parser.add_argument(
        "--load_model", type=str, default=None, help="path of pre-trained model"
    )
    parser.add_argument("--logging_path", type=str, default=None)
    parser.add_argument(
        "--config_root",
        type=str,
        default="config/",
        help="root path to config file",
    )
    parser.add_argument("--scene", type=str, default="BJTaxi", help="scene name")
    parser.add_argument(
        "--aug_scene", action="store_true", help="trajectron++ augmentation"
    )
    parser.add_argument(
        "--w_mse", type=float, default=0, help="loss weight of mse_loss"
    )
    parser.add_argument("--clusterGMM", action="store_true")
    parser.add_argument(
        "--cluster_method", type=str, default="kmeans", help="clustering method"
    )
    parser.add_argument("--cluster_n", type=int, help="n cluster centers")
    parser.add_argument(
        "--cluster_name", type=str, default="", help="clustering model name"
    )
    parser.add_argument(
        "--manual_weights", nargs='+', default=None, type=int)
    parser.add_argument("--var_init", type=float, default=0.7, help="init var")
    parser.add_argument("--learnVAR", action="store_true")
    
    # GAN训练特有参数
    parser.add_argument("--gan_epochs", type=int, default=100, help="GAN训练的轮数")
    parser.add_argument("--lambda_adv", type=float, default=0.1, help="对抗损失权重")
    parser.add_argument("--lambda_gp", type=float, default=10, help="梯度惩罚系数")
    parser.add_argument("--n_critic", type=int, default=3, help="每次生成器更新前判别器更新的次数")
    parser.add_argument("--batch_size", type=int, default=512, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--eval_interval", type=int, default=5, help="评估间隔（每多少个epoch评估一次）")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载工作线程数量")
    
    # 添加聚类相关的命令行参数
    parser.add_argument("--use_clustering", action="store_true", help="使用轨迹聚类减少数据量")
    parser.add_argument("--cluster_count", type=int, default=10, help="每个场景每个时间步的聚类数量")
    
    parser.add_argument("--seed", type=int, default=0, help="设定随机种子")
    parser.add_argument("--batch_size", type=int, default=None, help="设置批量大小")
    parser.add_argument("--num_workers", type=int, default=4, help="设置数据加载的工作线程数")



def train_flow_gan():
    args = parse_args()
    scene = args.scene
    
    # 设置配置文件和模型路径（如果未指定）
    if not args.config_file:
        args.config_file = f"./config/{scene}.yml"
    
    if not args.load_model:
        args.load_model = f"./checkpoint/{scene}.ckpt"
    
    # 加载配置
    cfg = load_config_test(args)

    # 设置模型参数（与test.py保持一致）
    args.clusterGMM = cfg.MGF.ENABLE
    args.cluster_n = cfg.MGF.CLUSTER_N
    args.var_init = cfg.MGF.VAR_INIT
    args.learn_var = cfg.MGF.VAR_LEARNABLE

    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 构建模型并加载预训练权重
    model = Build_Model(cfg, args)
    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint["state"], strict=False)
    print("预训练模型已加载")
    
    # 构建评估指标
    metrics = Build_Metrics(cfg)
    
    metrics = TrajectoryMetrics(metrics)
    model.to(device)

    # 设置GAN训练参数
    model.lambda_adv = args.lambda_adv
    model.lambda_gp = args.lambda_gp
    
    # 更新优化器学习率
    model.g_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    model.d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # 加载训练数据
    train_data_loader = unified_loader(
        cfg, 
        rand=True, 
        split="train", 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        use_clustering=args.use_clustering,
        cluster_count=args.cluster_count
    )
    val_data_loader = unified_loader(
        cfg, 
        rand=False, 
        split="val", 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        use_clustering=args.use_clustering,
        cluster_count=args.cluster_count
    )
    
    # 设置随机种子
    set_seeds(random.randint(0, 1000))
    
    print("开始GAN训练...")
    for epoch in range(args.gan_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_traj_loss = 0.0
        epoch_total_g_loss = 0.0
        batch_count = 0
        
        # 将数据加载器转换为列表的列表，每个内部列表包含一个批次
        batched_data = [[batch] for batch in train_data_loader]
        
        # 替换为单一的epoch级别进度条
        progress_bar = tqdm(total=len(batched_data), desc=f"Epoch {epoch+1}/{args.gan_epochs}", position=0, leave=True)
        
        for batch_list in batched_data:
            # 使用run_inference获取预测轨迹，但禁用其内部进度条
            obs_list, gt_list, pred_list, best_preds, obs_ts_array, gt_ts_array = run_inference(
                cfg, model, metrics, batch_list, disable_tqdm=True
            )
            
            # 切换到训练模式
            model.train()
            
            # 训练GAN
            gan_losses = model.train_gan(
                best_preds, gt_list, gt_ts_array, n_critic=args.n_critic
            )
            
            # 累计损失
            epoch_d_loss += gan_losses["d_loss"]
            epoch_g_loss += gan_losses["g_loss"]
            epoch_traj_loss += gan_losses["traj_loss"]
            epoch_total_g_loss += gan_losses["total_g_loss"]
            batch_count += 1
            
            # 更新进度条
            progress_bar.update(1)
        
        # 关闭进度条
        progress_bar.close()
        
        # 计算平均损失
        avg_d_loss = epoch_d_loss / batch_count if batch_count > 0 else 0
        avg_g_loss = epoch_g_loss / batch_count if batch_count > 0 else 0
        avg_traj_loss = epoch_traj_loss / batch_count if batch_count > 0 else 0
        avg_total_g_loss = epoch_total_g_loss / batch_count if batch_count > 0 else 0
        
        # 打印每个epoch的损失
        print(f"Completed Epoch {epoch+1}/{args.gan_epochs}")
        print(f"D Loss: {avg_d_loss:.4f}")
        print(f"G Loss: {avg_g_loss:.4f}")
        print(f"Traj Loss: {avg_traj_loss:.4f}")
        print(f"Total G Loss: {avg_total_g_loss:.4f}")
        
        # 定期评估模型
        if epoch % args.eval_interval == 0 or epoch == args.gan_epochs - 1:
            print(f"正在评估 Epoch {epoch+1} 的模型...")
            # 加载测试数据
            test_data_loader = unified_loader(cfg, rand=False, split="test", batch_size=args.batch_size*2, num_workers=args.num_workers)
            
            # 运行测试
            obs_list, gt_list, pred_list, best_preds_list, obs_ts_array, gt_ts_array = run_inference(
                cfg, model, metrics, test_data_loader, disable_tqdm=True
            )
            
            # 评估轨迹预测指标
            minade, minfde = evaluate_metrics(args, gt_list, pred_list)
            print(f"轨迹评估 - ADE: {minade:.4f}, FDE: {minfde:.4f}")
            
            # 评估流量指标
            flow_metrics = model.forward(best_preds_list, gt_list, gt_ts_array)
            print(f"流量评估:")
            print(f"Outflow RMSE: {flow_metrics['outflow_rmse']:.4f}")
            print(f"Outflow MAE: {flow_metrics['outflow_mae']:.4f}")
            print(f"Inflow RMSE: {flow_metrics['inflow_rmse']:.4f}") 
            print(f"Inflow MAE: {flow_metrics['inflow_mae']:.4f}")
    
    # 保存训练后的模型
    if args.save_model:
        save_path = f"./checkpoint/{scene}_flow_gan.ckpt"
        model.save(Path(save_path))
        print(f"GAN模型已保存到 {save_path}")


if __name__ == "__main__":
    st_time = time.time()
    train_flow_gan()
    end_time = time.time()
    print(f"训练总耗时: {end_time-st_time:.2f} 秒")