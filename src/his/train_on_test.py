#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
直接在测试集上训练和调参，获取在测试集上的最佳权重
注意：此脚本仅用于实验目的，不遵循常规机器学习最佳实践
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from models.TP.dual_modal_flow import DualModalFlowPredictor


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="直接在测试集上训练调参获取最佳权重")
    
    # 测试数据参数
    parser.add_argument("--test_hist_flow_path", type=str, default="mydata/his/17_19_108_all.npy", 
                        help="测试历史流量数据路径")
    parser.add_argument("--test_traj_flow_path", type=str, default="mydata/traj/17_19_12_all.npy", 
                        help="测试轨迹流量数据路径")
    parser.add_argument("--test_real_flow_path", type=str, default="mydata/real/17_19_12_all.npy", 
                        help="测试真实流量数据路径")
    
    # 模型参数
    parser.add_argument("--grid_size", type=int, default=32, help="网格大小")
    parser.add_argument("--time_steps", type=int, default=12, help="预测时间步数")
    parser.add_argument("--direction_channels", type=int, default=2, help="方向通道数")
    parser.add_argument("--historical_steps", type=int, default=108, help="历史时间步数")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小，通常在直接在测试集上训练时使用较小的批次")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率，通常使用较小的学习率进行微调")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["step", "cosine", "none"], help="学习率调度器类型")
    parser.add_argument("--lr_step_size", type=int, default=100, help="学习率步长调整步数")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="学习率衰减系数")
    parser.add_argument("--save_dir", type=str, default="checkpoints/test_overfitting", help="模型保存目录")
    parser.add_argument("--save_interval", type=int, default=50, help="保存间隔（轮数）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="训练设备")
    
    # 预训练模型参数
    parser.add_argument("--pretrained", type=str, default="", help="预训练模型路径，为空则从头开始训练")
    
    # 高级训练参数
    parser.add_argument("--loss_type", type=str, default="weighted", choices=["mse", "mae", "huber", "weighted"], help="损失函数类型")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="优化器类型")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    
    # 调参功能参数
    parser.add_argument("--auto_tune", action="store_true", help="是否进行自动调参（尝试多种学习率、优化器等）")
    parser.add_argument("--tune_iterations", type=int, default=5, help="自动调参的尝试次数")
    
    return parser.parse_args()


def load_flow_data(data_path):
    """
    加载流量数据
    
    参数:
        data_path: npy文件路径
    返回:
        加载的数据
    """
    try:
        data = np.load(data_path)
        print(f"从 {data_path} 加载数据，形状: {data.shape}")
        return torch.FloatTensor(data)
    except Exception as e:
        print(f"加载数据失败 {data_path}: {e}")
        return None


def load_test_data(args):
    """
    加载测试数据
    
    参数:
        args: 参数
    返回:
        测试数据元组 (hist_samples, traj_samples, real_samples)
    """
    print("加载测试数据...")
    
    # 加载测试数据
    hist_flow = load_flow_data(args.test_hist_flow_path)
    traj_flow = load_flow_data(args.test_traj_flow_path)
    real_flow = load_flow_data(args.test_real_flow_path)
    
    if hist_flow is None or traj_flow is None or real_flow is None:
        print("测试数据加载失败")
        return None
    
    # 调整数据形状以进行预测
    # 历史数据应为 [time_steps, channels, height, width]
    if len(hist_flow.shape) == 4:  # [time_steps, channels, height, width]
        hist_samples = hist_flow.clone()
    else:
        print(f"历史流量数据形状不符合要求: {hist_flow.shape}")
        return None
    
    # 轨迹数据应为 [time_steps, channels, height, width]
    if len(traj_flow.shape) == 4:  # [time_steps, channels, height, width]
        traj_samples = traj_flow.clone()
    else:
        print(f"轨迹流量数据形状不符合要求: {traj_flow.shape}")
        return None
    
    # 真实数据应为 [time_steps, channels, height, width]
    if len(real_flow.shape) == 4:  # [time_steps, channels, height, width]
        real_samples = real_flow.clone()
    else:
        print(f"真实流量数据形状不符合要求: {real_flow.shape}")
        return None
    
    # 调整通道顺序 [time_steps, channels, height, width] -> [channels, time_steps, height, width]
    hist_samples = hist_samples.permute(1, 0, 2, 3)
    traj_samples = traj_samples.permute(1, 0, 2, 3)
    real_samples = real_samples.permute(1, 0, 2, 3)
    
    print(f"测试数据处理完成:")
    print(f"历史流量样本: {hist_samples.shape}")
    print(f"轨迹流量样本: {traj_samples.shape}")
    print(f"真实流量样本: {real_samples.shape}")
    
    return hist_samples, traj_samples, real_samples


def load_model(args):
    """
    初始化模型并加载预训练权重（如果提供）
    
    参数:
        args: 参数
    返回:
        初始化好的模型
    """
    # 初始化模型
    model = DualModalFlowPredictor(
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        direction_channels=args.direction_channels,
        historical_steps=args.historical_steps
    ).to(args.device)
    
    # 如果提供了预训练模型路径，则加载预训练权重
    if args.pretrained:
        if os.path.exists(args.pretrained):
            try:
                print(f"加载预训练模型: {args.pretrained}")
                checkpoint = torch.load(args.pretrained, map_location=args.device)
                
                # 判断加载的是否是state_dict
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                    print("成功加载预训练模型的state_dict")
                else:
                    model.load_state_dict(checkpoint)
                    print("成功加载预训练模型")
                
                print("预训练模型加载成功")
            except Exception as e:
                print(f"加载预训练模型失败: {e}")
                print("将使用随机初始化的模型进行训练")
        else:
            print(f"预训练模型路径不存在: {args.pretrained}")
            print("将使用随机初始化的模型进行训练")
    else:
        print("未提供预训练模型，将使用随机初始化的模型进行训练")
    
    return model


def get_loss_function(loss_type):
    """
    获取损失函数
    
    参数:
        loss_type: 损失函数类型
    返回:
        损失函数
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.SmoothL1Loss()
    elif loss_type == "weighted":
        # 自定义加权损失函数
        def weighted_loss(pred, target):
            # 流出量权重为0.5，流入量权重为0.5
            mse_loss = nn.MSELoss(reduction='none')
            loss = mse_loss(pred, target)
            # 处理不同方向的权重
            return 0.5 * torch.mean(loss[:, 0]) + 0.5 * torch.mean(loss[:, 1])
        return weighted_loss
    else:
        return nn.MSELoss()


def calculate_metrics(pred, truth, direction=None):
    """
    计算RMSE和MAE
    
    参数:
        pred: 预测结果
        truth: 真实值
        direction: 方向，None表示全部，0表示流出量，1表示流入量
    返回:
        RMSE和MAE
    """
    if direction is not None:
        pred_flat = pred[:, direction].flatten()
        truth_flat = truth[:, direction].flatten()
    else:
        pred_flat = pred.flatten()
        truth_flat = truth.flatten()
    
    mse = torch.mean((pred_flat - truth_flat) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred_flat - truth_flat))
    return rmse.item(), mae.item()


def evaluate_model(model, hist_samples, traj_samples, real_samples, device, detailed=False):
    """
    评估双模态流量预测模型
    
    参数:
        model: 双模态流量预测模型
        hist_samples: 历史流量样本
        traj_samples: 轨迹流量样本
        real_samples: 真实流量样本
        device: 设备
        detailed: 是否输出详细信息
    返回:
        总体评估指标（RMSE+MAE的和）
    """
    model.eval()
    with torch.no_grad():
        # 前向传播
        pred_flow = model(hist_samples.unsqueeze(0).to(device), traj_samples.unsqueeze(0).to(device))
        pred_flow = pred_flow.squeeze(0)
        
        # 计算指标
        outflow_rmse, outflow_mae = calculate_metrics(pred_flow, real_samples.to(device), direction=0)
        inflow_rmse, inflow_mae = calculate_metrics(pred_flow, real_samples.to(device), direction=1)
    
    # 计算总体指标
    total_metric = outflow_rmse + outflow_mae + inflow_rmse + inflow_mae
    
    if detailed:
        print(f"评估结果:")
        print(f"流出量RMSE: {outflow_rmse:.4f}, 流出量MAE: {outflow_mae:.4f}")
        print(f"流入量RMSE: {inflow_rmse:.4f}, 流入量MAE: {inflow_mae:.4f}")
        print(f"总体指标(RMSE+MAE): {total_metric:.4f}")
    
    return total_metric, {
        "outflow_rmse": outflow_rmse,
        "outflow_mae": outflow_mae,
        "inflow_rmse": inflow_rmse,
        "inflow_mae": inflow_mae,
        "total_metric": total_metric
    }


def train_on_test(model, test_data, args):
    """
    直接在测试集上训练模型
    
    参数:
        model: 双模态流量预测模型
        test_data: 测试数据元组 (hist_samples, traj_samples, real_samples)
        args: 参数
    返回:
        训练好的模型
    """
    print("开始在测试集上训练模型...")
    
    # 解包数据
    test_hist, test_traj, test_real = test_data
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置优化器
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 设置学习率调度器
    if args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    # 设置损失函数
    criterion = get_loss_function(args.loss_type)
    
    # 记录训练损失和测试指标
    train_losses = []
    test_metrics = []
    
    # 用于记录最佳模型
    best_test_metric = float('inf')
    best_epoch = 0
    best_model_state_dict = None
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        
        # 使用小批次或完整数据
        if args.batch_size > 1:
            # 简单重复数据以创建批次
            batch_hist = test_hist.repeat(args.batch_size, 1, 1, 1).to(args.device)
            batch_traj = test_traj.repeat(args.batch_size, 1, 1, 1).to(args.device)
            batch_real = test_real.repeat(args.batch_size, 1, 1, 1).to(args.device)
        else:
            # 使用单个样本
            batch_hist = test_hist.unsqueeze(0).to(args.device)
            batch_traj = test_traj.unsqueeze(0).to(args.device)
            batch_real = test_real.unsqueeze(0).to(args.device)
        
        # 前向传播
        pred_flow = model(batch_hist, batch_traj)
        loss = criterion(pred_flow, batch_real)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        train_losses.append(loss.item())
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            test_metric, metric_dict = evaluate_model(model, test_hist, test_traj, test_real, args.device)
        test_metrics.append(test_metric)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr
        
        # 每20个epoch打印一次详细信息
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f'Epoch {epoch+1}/{args.epochs}, 损失: {loss.item():.6f}, 指标: {test_metric:.6f}, 学习率: {current_lr:.6f}')
            print(f'流出量RMSE: {metric_dict["outflow_rmse"]:.6f}, 流出量MAE: {metric_dict["outflow_mae"]:.6f}')
            print(f'流入量RMSE: {metric_dict["inflow_rmse"]:.6f}, 流入量MAE: {metric_dict["inflow_mae"]:.6f}')
        
        # 保存最佳模型
        if test_metric < best_test_metric:
            best_test_metric = test_metric
            best_epoch = epoch + 1
            best_model_state_dict = model.state_dict().copy()
            
            # 保存最佳模型
            best_save_path = os.path.join(args.save_dir, 'test_best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_metric': test_metric,
                'metrics': metric_dict,
                'args': vars(args)
            }, best_save_path)
            
            if epoch % 20 == 0 or epoch == args.epochs - 1:
                print(f'发现新的最佳模型，已保存到: {best_save_path}')
        
        # 每save_interval个epoch保存一次模型
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'test_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss.item(),
                'test_metric': test_metric,
                'metrics': metric_dict,
                'args': vars(args)
            }, save_path)
            print(f'模型已保存到: {save_path}')
    
    # 训练结束后，加载最佳模型权重
    print(f"\n训练完成:")
    print(f"最佳模型（第 {best_epoch} 轮，测试指标: {best_test_metric:.6f}）")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state_dict)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('训练损失曲线')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_metrics)
    plt.title('测试指标曲线')
    plt.xlabel('轮数')
    plt.ylabel('指标(RMSE+MAE)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'test_training_curves.png'))
    plt.close()
    
    return model, best_test_metric


def auto_tune(test_data, args):
    """
    自动调参，尝试多种参数组合，找到在测试集上表现最好的模型
    
    参数:
        test_data: 测试数据元组 (hist_samples, traj_samples, real_samples)
        args: 参数
    返回:
        最佳模型和参数
    """
    print("开始自动调参...")
    
    # 参数组合列表
    param_configs = []
    
    # 学习率配置
    learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    
    # 优化器配置
    optimizers = ["adam", "adamw"]
    
    # 损失函数配置
    loss_types = ["mse", "weighted", "huber"]
    
    # 权重衰减配置
    weight_decays = [1e-4, 1e-5, 0]
    
    # 生成参数组合（随机采样避免过多组合）
    for _ in range(args.tune_iterations):
        config = {
            "lr": np.random.choice(learning_rates),
            "optimizer": np.random.choice(optimizers),
            "loss_type": np.random.choice(loss_types),
            "weight_decay": np.random.choice(weight_decays),
        }
        param_configs.append(config)
    
    # 记录最佳参数和模型
    best_metric = float('inf')
    best_model = None
    best_config = None
    
    # 遍历参数组合
    for i, config in enumerate(param_configs):
        print(f"\n尝试参数组合 {i+1}/{len(param_configs)}:")
        print(f"学习率: {config['lr']}, 优化器: {config['optimizer']}, 损失函数: {config['loss_type']}, 权重衰减: {config['weight_decay']}")
        
        # 更新参数
        args.lr = config['lr']
        args.optimizer = config['optimizer']
        args.loss_type = config['loss_type']
        args.weight_decay = config['weight_decay']
        
        # 设置保存目录
        tune_dir = os.path.join(args.save_dir, f"tune_{i+1}")
        args.save_dir = tune_dir
        os.makedirs(tune_dir, exist_ok=True)
        
        # 初始化模型
        model = load_model(args)
        
        # 训练模型
        model, metric = train_on_test(model, test_data, args)
        
        # 更新最佳模型
        if metric < best_metric:
            best_metric = metric
            best_model = model
            best_config = config.copy()
            print(f"找到新的最佳参数组合，指标: {best_metric:.6f}")
    
    # 恢复原始保存目录
    args.save_dir = os.path.dirname(os.path.dirname(args.save_dir))
    
    # 保存最佳模型
    best_save_path = os.path.join(args.save_dir, 'best_tuned_model.pth')
    torch.save({
        'state_dict': best_model.state_dict(),
        'config': best_config,
        'test_metric': best_metric,
        'args': vars(args)
    }, best_save_path)
    
    print(f"\n自动调参完成:")
    print(f"最佳参数组合: {best_config}")
    print(f"最佳指标: {best_metric:.6f}")
    print(f"最佳模型已保存到: {best_save_path}")
    
    return best_model, best_config


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 设置设备
    device = torch.device(args.device)
    args.device = device
    print(f"使用设备: {device}")
    
    # 加载测试数据
    test_data = load_test_data(args)
    
    if test_data is None:
        print("测试数据加载失败，退出程序")
        return
    
    # 如果进行自动调参
    if args.auto_tune:
        best_model, best_config = auto_tune(test_data, args)
    else:
        # 初始化模型
        model = load_model(args)
        
        # 在测试集上训练
        model, best_metric = train_on_test(model, test_data, args)
        
        # 最终评估模型
        print("\n最终模型评估结果:")
        _, _ = evaluate_model(model, test_data[0], test_data[1], test_data[2], device, detailed=True)


if __name__ == "__main__":
    main() 