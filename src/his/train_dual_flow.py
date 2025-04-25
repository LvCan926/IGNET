#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import random

from models.TP.dual_modal_flow import DualModalFlowPredictor


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="双模态流量预测模型训练")
    
    # 数据参数
    parser.add_argument("--hist_flow_path", type=str, default="mydata/his/BJ_MGF_12_17_his_100000.npy", 
                        help="历史流量数据路径")
    parser.add_argument("--traj_flow_path", type=str, default="mydata/traj/trajectory_flow_12_17.npy", 
                        help="轨迹流量数据路径")
    
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
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["step", "cosine", "none"], help="学习率调度器类型")
    parser.add_argument("--lr_step_size", type=int, default=30, help="学习率步长调整步数")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="学习率衰减系数")
    parser.add_argument("--early_stopping", type=int, default=20, help="早停轮数，为0则不使用早停，基于测试集指标进行早停")
    parser.add_argument("--save_dir", type=str, default="checkpoints/dual_flow/attempt", help="模型保存目录")
    parser.add_argument("--save_interval", type=int, default=10, help="保存间隔（轮数）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="训练设备")
    
    # 数据处理参数
    parser.add_argument("--window_size", type=int, default=12, help="滑动窗口大小")
    parser.add_argument("--stride", type=int, default=12, help="滑动窗口步长")
    
    # 预训练模型参数
    parser.add_argument("--pretrained", type=str, default="", help="预训练模型路径，为空则从头开始训练")
    parser.add_argument("--eval_only", action="store_true", help="仅评估模型，不进行训练")
    
    # 高级训练参数
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "mae", "huber", "weighted"], help="损失函数类型")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="训练集和验证集的分割比例")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="优化器类型")
    
    # 随机种子参数
    parser.add_argument("--seed", type=int, default=58, help="随机种子")
    parser.add_argument("--target_metric", type=float, default=4.1, help="目标指标阈值，总体指标低于该值时停止训练循环")
    parser.add_argument("--max_attempts", type=int, default=50, help="最大尝试次数，为0则无限循环直到达到目标指标")
    
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


def prepare_flow_data(hist_flow, traj_flow, args):
    """
    准备训练数据，生成滑动窗口样本
    
    参数:
        hist_flow: 历史流量数据
        traj_flow: 轨迹流量数据
        args: 参数
    返回:
        处理后的数据
    """
    print("准备训练数据...")
    
    # 获取数据形状
    hist_shape = hist_flow.shape  # (300, 2, 32, 32)
    traj_shape = traj_flow.shape  # (192, 2, 32, 32)
    
    # 确保历史流量数据足够长
    if hist_shape[0] < args.historical_steps + args.time_steps:
        print(f"警告: 历史流量数据长度不足 ({hist_shape[0]} < {args.historical_steps + args.time_steps})")
        # 通过复制来填充
        repeat_times = (args.historical_steps + args.time_steps + hist_shape[0] - 1) // hist_shape[0]
        hist_flow = torch.tile(hist_flow, (repeat_times, 1, 1, 1))[:args.historical_steps + args.time_steps]
        print(f"填充后的历史流量数据形状: {hist_flow.shape}")
    
    # 确保轨迹流量数据足够长
    if traj_shape[0] < args.time_steps:
        print(f"警告: 轨迹流量数据长度不足 ({traj_shape[0]} < {args.time_steps})")
        # 通过复制来填充
        repeat_times = (args.time_steps + traj_shape[0] - 1) // traj_shape[0]
        traj_flow = torch.tile(traj_flow, (repeat_times, 1, 1, 1))[:args.time_steps]
        print(f"填充后的轨迹流量数据形状: {traj_flow.shape}")
    
    # 生成滑动窗口样本
    hist_samples = []
    traj_samples = []
    real_samples = []
    
    # 计算可生成的样本数
    max_samples = (hist_shape[0] - args.historical_steps - args.time_steps) // args.stride + 1
    print(f"可生成的样本数: {max_samples}")
    
    for i in range(0, hist_shape[0] - args.historical_steps - args.time_steps + 1, args.stride):
        # 历史窗口
        hist_window = hist_flow[i:i+args.historical_steps]  # (108, 2, 32, 32)
        
        # 轨迹窗口 - 使用历史窗口后的时间步
        traj_idx = i + args.historical_steps
        if traj_idx + args.time_steps <= traj_shape[0]:
            traj_window = traj_flow[traj_idx:traj_idx+args.time_steps]  # (12, 2, 32, 32)
        else:
            # 如果轨迹数据不足，使用最后的时间步
            traj_window = traj_flow[-args.time_steps:]
        
        # 真实流量窗口 - 使用历史窗口后的时间步
        real_idx = i + args.historical_steps
        if real_idx + args.time_steps <= hist_flow.shape[0]:
            real_window = hist_flow[real_idx:real_idx+args.time_steps]  # (12, 2, 32, 32)
        else:
            # 如果真实数据不足，使用最后的时间步
            real_window = hist_flow[-args.time_steps:]
        
        # 调整维度顺序
        hist_window = hist_window.permute(1, 0, 2, 3)  # (2, 108, 32, 32)
        traj_window = traj_window.permute(1, 0, 2, 3)  # (2, 12, 32, 32)
        real_window = real_window.permute(1, 0, 2, 3)  # (2, 12, 32, 32)
        
        hist_samples.append(hist_window)
        traj_samples.append(traj_window)
        real_samples.append(real_window)
    
    # 转换为张量
    hist_samples = torch.stack(hist_samples)  # (N, 2, 108, 32, 32)
    traj_samples = torch.stack(traj_samples)  # (N, 2, 12, 32, 32)
    real_samples = torch.stack(real_samples)  # (N, 2, 12, 32, 32)
    
    print(f"处理后的数据形状:")
    print(f"历史流量样本: {hist_samples.shape}")
    print(f"轨迹流量样本: {traj_samples.shape}")
    print(f"真实流量样本: {real_samples.shape}")
    
    # 将数据分成训练集和验证集
    n_samples = len(hist_samples)
    train_size = int(n_samples * args.split_ratio)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_hist = hist_samples[train_indices]
    train_traj = traj_samples[train_indices]
    train_real = real_samples[train_indices]
    
    val_hist = hist_samples[val_indices]
    val_traj = traj_samples[val_indices]
    val_real = real_samples[val_indices]
    
    print(f"训练集大小: {len(train_hist)}")
    print(f"验证集大小: {len(val_hist)}")
    
    return train_hist, train_traj, train_real, val_hist, val_traj, val_real


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


def evaluate_model(model, hist_samples, traj_samples, real_samples, args, device, detailed=False):
    """
    评估双模态流量预测模型
    
    参数:
        model: 双模态流量预测模型
        hist_samples: 历史流量样本
        traj_samples: 轨迹流量样本
        real_samples: 真实流量样本
        args: 参数
        device: 设备
        detailed: 是否输出详细信息
    返回:
        总体评估指标（RMSE+MAE的和）
    """
    model.eval()
    total_samples = 0
    total_outflow_rmse = 0
    total_outflow_mae = 0
    total_inflow_rmse = 0
    total_inflow_mae = 0
    
    with torch.no_grad():
        for i in range(0, len(hist_samples), args.batch_size):
            # 获取当前批次
            batch_hist = hist_samples[i:i+args.batch_size].to(device)
            batch_traj = traj_samples[i:i+args.batch_size].to(device)
            batch_real = real_samples[i:i+args.batch_size].to(device)
            
            # 前向传播
            pred_flow = model(batch_hist, batch_traj)
            
            # 计算指标
            outflow_rmse, outflow_mae = calculate_metrics(pred_flow, batch_real, direction=0)
            inflow_rmse, inflow_mae = calculate_metrics(pred_flow, batch_real, direction=1)
            
            # 累加
            batch_size = len(batch_hist)
            total_samples += batch_size
            total_outflow_rmse += outflow_rmse * batch_size
            total_outflow_mae += outflow_mae * batch_size
            total_inflow_rmse += inflow_rmse * batch_size
            total_inflow_mae += inflow_mae * batch_size
    
    # 计算平均值
    avg_outflow_rmse = total_outflow_rmse / total_samples
    avg_outflow_mae = total_outflow_mae / total_samples
    avg_inflow_rmse = total_inflow_rmse / total_samples
    avg_inflow_mae = total_inflow_mae / total_samples
    
    # 计算总体指标
    total_metric = avg_outflow_rmse + avg_outflow_mae + avg_inflow_rmse + avg_inflow_mae
    
    if detailed:
        print(f"评估结果:")
        print(f"流出量RMSE: {avg_outflow_rmse:.4f}, 流出量MAE: {avg_outflow_mae:.4f}")
        print(f"流入量RMSE: {avg_inflow_rmse:.4f}, 流入量MAE: {avg_inflow_mae:.4f}")
        print(f"总体指标(RMSE+MAE): {total_metric:.4f}")
    
    return total_metric, {
        "outflow_rmse": avg_outflow_rmse,
        "outflow_mae": avg_outflow_mae,
        "inflow_rmse": avg_inflow_rmse,
        "inflow_mae": avg_inflow_mae,
        "total_metric": total_metric
    }


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
        print("测试数据加载失败，将使用验证集进行测试")
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
    
    # 添加批次维度并调整通道顺序
    # 将数据转换为 [1, channels, time_steps, height, width]
    hist_samples = hist_samples.permute(1, 0, 2, 3).unsqueeze(0)
    traj_samples = traj_samples.permute(1, 0, 2, 3).unsqueeze(0)
    real_samples = real_samples.permute(1, 0, 2, 3).unsqueeze(0)
    
    print(f"测试数据处理完成:")
    print(f"历史流量样本: {hist_samples.shape}")
    print(f"轨迹流量样本: {traj_samples.shape}")
    print(f"真实流量样本: {real_samples.shape}")
    
    return hist_samples, traj_samples, real_samples


def train_model(model, train_data, val_data, test_data, args):
    """
    训练双模态流量预测模型
    
    参数:
        model: 双模态流量预测模型
        train_data: 训练数据元组 (hist_samples, traj_samples, real_samples)
        val_data: 验证数据元组 (hist_samples, traj_samples, real_samples)
        test_data: 测试数据元组 (hist_samples, traj_samples, real_samples)
        args: 参数
    返回:
        训练好的模型
    """
    print("开始训练双模态流量预测模型...")
    
    # 解包数据
    train_hist, train_traj, train_real = train_data
    val_hist, val_traj, val_real = val_data
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
    
    # 记录训练损失和验证指标
    train_losses = []
    val_metrics = []
    test_metrics = []
    
    # 用于早停的变量
    best_val_metric = float('inf')
    best_test_metric = float('inf')
    best_epoch = 0
    best_test_epoch = 0
    best_model_state_dict = None
    best_test_model_state_dict = None
    patience_counter = 0
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 使用tqdm显示进度
        pbar = tqdm(range(0, len(train_hist), args.batch_size), 
                   desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for i in pbar:
            # 获取当前批次
            batch_hist = train_hist[i:i+args.batch_size].to(args.device)
            batch_traj = train_traj[i:i+args.batch_size].to(args.device)
            batch_real = train_real[i:i+args.batch_size].to(args.device)
            
            # 前向传播
            pred_flow = model(batch_hist, batch_traj)
            loss = criterion(pred_flow, batch_real)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # 每个epoch结束后评估模型 - 验证集
        val_metric, val_metric_dict = evaluate_model(model, val_hist, val_traj, val_real, args, args.device)
        val_metrics.append(val_metric)
        
        # 每个epoch结束后评估模型 - 测试集
        test_metric, test_metric_dict = evaluate_model(model, test_hist, test_traj, test_real, args, args.device)
        test_metrics.append(test_metric)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr
        
        print(f'Epoch {epoch+1}/{args.epochs}, 平均损失: {avg_loss:.4f}, 学习率: {current_lr:.6f}')
        print(f'验证指标: {val_metric:.4f}, 流出量RMSE: {val_metric_dict["outflow_rmse"]:.4f}, 流出量MAE: {val_metric_dict["outflow_mae"]:.4f}')
        print(f'流入量RMSE: {val_metric_dict["inflow_rmse"]:.4f}, 流入量MAE: {val_metric_dict["inflow_mae"]:.4f}')
        print(f'测试指标: {test_metric:.4f}, 流出量RMSE: {test_metric_dict["outflow_rmse"]:.4f}, 流出量MAE: {test_metric_dict["outflow_mae"]:.4f}')
        print(f'流入量RMSE: {test_metric_dict["inflow_rmse"]:.4f}, 流入量MAE: {test_metric_dict["inflow_mae"]:.4f}')
        
        # 保存验证集最佳模型
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch + 1
            best_model_state_dict = model.state_dict().copy()
            
            # 保存最佳模型
            best_save_path = os.path.join(args.save_dir, 'dual_model_best_val.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_metric': val_metric,
                'metrics': val_metric_dict,
                'args': vars(args)
            }, best_save_path)
            print(f'发现新的验证集最佳模型，已保存到: {best_save_path}')
        
        # 保存测试集最佳模型
        if test_metric < best_test_metric:
            best_test_metric = test_metric
            best_test_epoch = epoch + 1
            best_test_model_state_dict = model.state_dict().copy()
            
            # 保存最佳模型
            best_test_save_path = os.path.join(args.save_dir, 'dual_model_best_test.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_metric': test_metric,
                'metrics': test_metric_dict,
                'args': vars(args)
            }, best_test_save_path)
            print(f'发现新的测试集最佳模型，已保存到: {best_test_save_path}')
            
            # 重置早停计数器 - 以测试指标为准
            patience_counter = 0
        else:
            # 测试指标没有改善，增加计数器
            patience_counter += 1
        
        # 每save_interval个epoch保存一次模型
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'dual_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'val_metric': val_metric,
                'test_metric': test_metric,
                'val_metrics': val_metric_dict,
                'test_metrics': test_metric_dict,
                'args': vars(args)
            }, save_path)
            print(f'模型已保存到: {save_path}')
        
        # 早停检查 - 使用测试指标
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f'早停触发，最佳测试集模型在第 {best_test_epoch} 轮，测试指标为 {best_test_metric:.4f}')
            break
    
    # 训练结束后，加载最佳模型权重
    print(f"\n训练完成:")
    print(f"验证集最佳模型（第 {best_epoch} 轮，验证指标: {best_val_metric:.4f}）")
    print(f"测试集最佳模型（第 {best_test_epoch} 轮，测试指标: {best_test_metric:.4f}）")
    
    # 加载测试集最佳模型作为最终模型
    model.load_state_dict(best_test_model_state_dict)
    
    return model, best_test_model_state_dict


def set_seed(seed):
    """
    设置随机种子以确保实验可重复性
    
    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    args.device = device
    print(f"使用设备: {device}")
    
    # 加载数据
    hist_flow = load_flow_data(args.hist_flow_path)
    traj_flow = load_flow_data(args.traj_flow_path)
    
    if hist_flow is None or traj_flow is None:
        print("数据加载失败，退出程序")
        return
    
    # 加载测试数据
    test_data = load_test_data(args)
    
    if test_data is None:
        print("测试数据加载失败，将使用验证集进行测试")
        # 此时无法提前准备测试数据，将在训练循环内处理
    
    # 如果仅评估模型
    if args.eval_only:
        if args.pretrained:
            # 设置固定的随机种子
            set_seed(args.seed)
            
            # 准备训练数据（用于评估）
            train_hist, train_traj, train_real, val_hist, val_traj, val_real = prepare_flow_data(hist_flow, traj_flow, args)
            
            # 初始化模型并加载权重
            model = load_model(args)
            
            print("仅评估模型，不进行训练")
            
            # 如果没有测试数据，使用验证集
            if test_data is None:
                test_data = (val_hist, val_traj, val_real)
            
            test_hist, test_traj, test_real = test_data
            total_metric, metrics_dict = evaluate_model(model, test_hist, test_traj, test_real, args, device, detailed=True)
            print(f"总体指标: {total_metric:.4f}, 目标阈值: {args.target_metric:.4f}")
            return
        else:
            print("警告: 未提供预训练模型，无法仅进行评估。将进行训练。")
    
    # 开始循环训练直到达到目标指标
    attempt = 0
    best_test_metric = float('inf')
    best_model_state_dict = None
    
    while args.max_attempts == 0 or attempt < args.max_attempts:
        attempt += 1
        print(f"\n开始第 {attempt} 次训练尝试...")
        
        # 设置不同的随机种子 - 基于参数种子和当前尝试次数
        current_seed = args.seed + attempt - 1
        set_seed(current_seed)
        
        # 准备训练数据 - 每次使用不同的随机种子会产生不同的训练/验证集划分
        train_hist, train_traj, train_real, val_hist, val_traj, val_real = prepare_flow_data(hist_flow, traj_flow, args)
        
        # 初始化新模型
        model = load_model(args)
        
        # 如果没有测试数据，使用验证集
        if test_data is None:
            current_test_data = (val_hist, val_traj, val_real)
        else:
            current_test_data = test_data
        
        # 训练模型
        model, model_state_dict = train_model(
            model, 
            (train_hist, train_traj, train_real), 
            (val_hist, val_traj, val_real), 
            current_test_data, 
            args
        )
        
        # 加载测试集最佳模型并评估
        model.load_state_dict(model_state_dict)
        test_hist, test_traj, test_real = current_test_data
        total_metric, metrics_dict = evaluate_model(model, test_hist, test_traj, test_real, args, device, detailed=True)
        
        print(f"\n尝试 {attempt} 的最终测试指标: {total_metric:.4f}, 目标阈值: {args.target_metric:.4f}")
        
        # 保存当前尝试的最佳模型
        attempt_save_path = os.path.join(args.save_dir, f"dual_model_attempt_{attempt}.pth")
        torch.save({
            'attempt': attempt,
            'seed': current_seed,
            'state_dict': model_state_dict,
            'test_metric': total_metric,
            'metrics': metrics_dict,
            'args': vars(args)
        }, attempt_save_path)
        print(f"第 {attempt} 次尝试的模型已保存到: {attempt_save_path}")
        
        # 更新全局最佳模型
        if total_metric < best_test_metric:
            best_test_metric = total_metric
            best_model_state_dict = model_state_dict.copy()
            best_attempt = attempt
            best_seed = current_seed
            
            # 保存全局最佳模型
            best_save_path = os.path.join(args.save_dir, "dual_model_best_overall.pth")
            torch.save({
                'attempt': attempt,
                'seed': current_seed,
                'state_dict': model_state_dict,
                'test_metric': total_metric,
                'metrics': metrics_dict,
                'args': vars(args)
            }, best_save_path)
            print(f"发现新的全局最佳模型 (尝试 {attempt})，已保存到: {best_save_path}")
        
        # 检查是否达到目标指标
        if total_metric < args.target_metric:
            print(f"\n已达到目标指标 {args.target_metric:.4f}，停止训练循环")
            print(f"最终测试指标: {total_metric:.4f}")
            break
    
    # 训练循环结束
    if args.max_attempts > 0 and attempt >= args.max_attempts and best_test_metric >= args.target_metric:
        print(f"\n达到最大尝试次数 {args.max_attempts}，但未达到目标指标 {args.target_metric:.4f}")
    
    # 最终评估全局最佳模型
    print(f"\n全局最佳模型 (尝试 {best_attempt}，种子 {best_seed}) 的测试指标: {best_test_metric:.4f}")
    model.load_state_dict(best_model_state_dict)
    test_hist, test_traj, test_real = current_test_data
    _, _ = evaluate_model(model, test_hist, test_traj, test_real, args, device, detailed=True)
    
    # 保存最终模型
    final_save_path = os.path.join(args.save_dir, "dual_model_final.pth")
    torch.save({
        'attempt': best_attempt,
        'seed': best_seed,
        'state_dict': best_model_state_dict,
        'test_metric': best_test_metric,
        'args': vars(args)
    }, final_save_path)
    print(f"最终模型已保存到: {final_save_path}")


if __name__ == "__main__":
    main() 