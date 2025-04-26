import numpy as np
import torch
from models.TP.dual_modal_flow import DualModalFlowPredictor
import h5py
import os
import argparse

def calculate_metrics(pred, truth, direction):
    """计算RMSE和MAE，direction=0表示流出量，direction=1表示流入量"""
    pred_flat = pred[:, direction].flatten()
    truth_flat = truth[:, direction].flatten()
    
    mse = np.mean((pred_flat - truth_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - truth_flat))
    return rmse, mae


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试双模态流量预测模型")
    
    # 数据参数
    parser.add_argument("--hist_flow_path", type=str, default="mydata/his/17_19_108_all.npy", 
                        help="历史流量数据路径")
    parser.add_argument("--traj_flow_path", type=str, default="mydata/traj/17_19_12_all.npy", 
                        help="轨迹流量数据路径")
    parser.add_argument("--real_flow_path", type=str, default="mydata/real/17_19_12_all.npy", 
                        help="真实流量数据路径")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="checkpoints/real/dual_flow/dual_model_epoch_100.pth", 
                        help="模型路径，默认使用最佳模型")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="测试设备")
    
    # 添加聚类相关的命令行参数
    parser.add_argument("--use_clustering", action="store_true", help="使用轨迹聚类减少数据量")
    parser.add_argument("--cluster_count", type=int, default=10, help="每个场景每个时间步的聚类数量")
    
    return parser.parse_args()


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
    """主函数"""
    # 解析参数
    args = parse_args()
    device = torch.device(args.device)
    
    # 加载测试数据
    his_flow = np.load(args.hist_flow_path)
    traj_pred = np.load(args.traj_flow_path)
    real_flow = np.load(args.real_flow_path)
    
    new_flow = np.load("mydata/new/new_flow_forced.npy")
    
    #traj_pred = np.zeros_like(traj_pred)
    
    print(f"历史流量数据形状: {his_flow.shape}")
    print(f"轨迹流量数据形状: {traj_pred.shape}")
    print(f"真实流量数据形状: {real_flow.shape}")
    print(f"轨迹流量最大值: {traj_pred.max()}")
    
    # 将数据转换为张量
    his_flow = torch.from_numpy(his_flow).float()
    traj_pred = torch.from_numpy(traj_pred).float()
    
    # 调整维度顺序
    his_flow = his_flow.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 2, time_steps, 32, 32]
    traj_pred = traj_pred.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 2, time_steps, 32, 32]
    
    # 初始化模型
    model = DualModalFlowPredictor()
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 判断加载的是否是state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"成功加载模型 state_dict: {args.model_path}")
        if "metrics" in checkpoint:
            print(f"训练时的验证指标: {checkpoint['metrics']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"成功加载模型: {args.model_path}")
    
    model = model.to(device)
    model.eval()
    
    # 将数据移动到设备上
    his_flow = his_flow.to(device)
    traj_pred = traj_pred.to(device)
    
    # 推理
    with torch.no_grad():
        total_pred = model(his_flow, traj_pred)
    
    # 转换回原始格式 [time_steps, direction_channels, grid_size, grid_size]
    total_pred = total_pred.squeeze(0).permute(1, 0, 2, 3).cpu().detach().numpy()
    traj_pred = traj_pred.squeeze(0).permute(1, 0, 2, 3).cpu().detach().numpy()
    
    # 计算真实流出量和预测流出量的均值
    real_outflow_mean = np.mean(real_flow[:, 0])
    pred_outflow_mean = np.mean(total_pred[:, 0])
    
    print(f"\n流量评估指标:")
    print(f"真实流出量均值: {real_outflow_mean:.4f}, 预测流出量均值: {pred_outflow_mean:.4f}")
    
    # 计算真实流入量和预测流入量的均值
    real_inflow_mean = np.mean(real_flow[:, 1])
    pred_inflow_mean = np.mean(total_pred[:, 1])
    
    print(f"真实流入量均值: {real_inflow_mean:.4f}, 预测流入量均值: {pred_inflow_mean:.4f}")
    
    # 计算误差
    inflow_rmse, inflow_mae = calculate_metrics(total_pred, real_flow, direction=1)
    outflow_rmse, outflow_mae = calculate_metrics(total_pred, real_flow, direction=0)
    
    traj_inflow_rmse, traj_inflow_mae = calculate_metrics(traj_pred, real_flow, direction=1)
    traj_outflow_rmse, traj_outflow_mae = calculate_metrics(traj_pred, real_flow, direction=0)
    
    # 计算综合指标
    total_metric = inflow_rmse + inflow_mae + outflow_rmse + outflow_mae
    traj_metric = traj_inflow_rmse + traj_inflow_mae + traj_outflow_rmse + traj_outflow_mae
    
    # 表格形式
    print("\n详细评估结果:")
    print("| 模型      | 流入量RMSE | 流入量MAE | 流出量RMSE | 流出量MAE | 综合指标 |")
    print("|-----------|-----------|-----------|-----------|-----------|----------|")
    print(f"| 双模态模型 | {inflow_rmse:.4f} | {inflow_mae:.4f} | {outflow_rmse:.4f} | {outflow_mae:.4f} | {total_metric:.4f} |")
    print(f"| 轨迹流量  | {traj_inflow_rmse:.4f} | {traj_inflow_mae:.4f} | {traj_outflow_rmse:.4f} | {traj_outflow_mae:.4f} | {traj_metric:.4f} |")
    
    # 计算改进百分比
    improvement = (traj_metric - total_metric) / traj_metric * 100
    print(f"\n相对于仅使用轨迹流量的改进: {improvement:.2f}%")


if __name__ == "__main__":
    main()
