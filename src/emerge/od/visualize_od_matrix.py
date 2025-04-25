import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from matplotlib.colors import LogNorm
import h5py

def load_od_matrix(file_path):
    """加载OD矩阵数据"""
    print(f"加载OD矩阵: {file_path}")
    
    # 检查文件扩展名判断格式
    if file_path.endswith('.h5'):
        # 使用h5py加载h5格式
        with h5py.File(file_path, 'r') as f:
            # 加载OD矩阵
            od_matrix = f['od_matrix'][:]
            
            # 加载邻域映射
            adjacency_map = {}
            adj_group = f['adjacency_map']
            for key in adj_group.keys():
                grid_index = int(key)
                adjacency_map[grid_index] = list(adj_group[key][:])
            
            print(f"加载了h5格式OD矩阵，形状: {od_matrix.shape}")
            return od_matrix, adjacency_map
    else:
        # 使用numpy加载npy格式
        data = np.load(file_path, allow_pickle=True)
        
        # 检查是否是新格式的OD矩阵（包含邻域映射）
        if isinstance(data, np.ndarray) and data.dtype == np.dtype('O') and isinstance(data.item(), dict):
            data_dict = data.item()
            od_matrix = data_dict['od_matrix']
            
            # 适配新旧命名格式
            if 'adjacency_map' in data_dict:
                adjacency_map = data_dict['adjacency_map']
            elif 'adjacent_grids_map' in data_dict:
                adjacency_map = data_dict['adjacent_grids_map']
            else:
                adjacency_map = None
            
            print(f"加载了npy格式OD矩阵，形状: {od_matrix.shape}")
            return od_matrix, adjacency_map
        else:
            raise ValueError("不支持的OD矩阵格式。请使用包含邻域映射的新格式OD矩阵。")

def visualize_temporal_pattern(od_matrix, output_dir):
    """可视化不同时间步的OD流量模式"""
    time_steps, grid_count, _, _ = od_matrix.shape
    
    # 计算每个时间步的总流量
    time_flow = np.zeros(time_steps)
    for t in range(time_steps):
        # 统计所有有效的占比值
        time_flow[t] = np.sum(od_matrix[t, :, :, 0] > 0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(time_steps), time_flow, 'o-', linewidth=2)
    plt.xlabel('Time Step (Time - 1020)')
    plt.ylabel('Number of Active OD Pairs')
    plt.title('OD Flow Pattern at Different Time Steps')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'temporal_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"时间模式可视化已保存到 {os.path.join(output_dir, 'temporal_pattern.png')}")

def visualize_avg_travel_time(od_matrix, output_dir):
    """可视化平均旅行时间"""
    time_steps, grid_count, _, _ = od_matrix.shape
    
    # 计算每个时间步的平均旅行时间
    avg_times = np.zeros(time_steps)
    counts = np.zeros(time_steps)
    
    for t in range(time_steps):
        # 只考虑有效的OD对(占比>0)
        mask = od_matrix[t, :, :, 0] > 0
        if np.any(mask):
            avg_times[t] = np.sum(od_matrix[t, :, :, 1][mask]) / np.sum(mask)
            counts[t] = np.sum(mask)
    
    # 绘制平均旅行时间
    plt.figure(figsize=(10, 6))
    plt.plot(range(time_steps), avg_times, 'o-', linewidth=2)
    plt.xlabel('Time Step (Time - 1020)')
    plt.ylabel('Average Travel Time')
    plt.title('Average Travel Time at Different Time Steps')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'avg_travel_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"平均旅行时间可视化已保存到 {os.path.join(output_dir, 'avg_travel_time.png')}")

def visualize_od_heatmap(od_matrix, time_step, output_dir, grid_size=32):
    """可视化特定时间步的OD矩阵热力图"""
    ratio_matrix = od_matrix[time_step, :, :, 0]  # 占比矩阵
    time_matrix = od_matrix[time_step, :, :, 1]   # 时间矩阵
    
    # 创建可视化目录
    os.makedirs(os.path.join(output_dir, f'time_step_{time_step}'), exist_ok=True)
    
    # 绘制占比热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(ratio_matrix.reshape(grid_size, grid_size, grid_size, grid_size).sum(axis=(0, 1)), 
                cmap='viridis', norm=LogNorm(), annot=False)
    plt.title(f'Destination Heatmap at Time Step {time_step} (Sum of Ratio)')
    plt.xlabel('Destination X')
    plt.ylabel('Destination Y')
    plt.savefig(os.path.join(output_dir, f'time_step_{time_step}/destination_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制出发点热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(ratio_matrix.reshape(grid_size, grid_size, grid_size, grid_size).sum(axis=(2, 3)), 
                cmap='viridis', norm=LogNorm(), annot=False)
    plt.title(f'Origin Heatmap at Time Step {time_step} (Sum of Ratio)')
    plt.xlabel('Origin X')
    plt.ylabel('Origin Y')
    plt.savefig(os.path.join(output_dir, f'time_step_{time_step}/origin_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制平均旅行时间热力图
    valid_mask = ratio_matrix > 0
    time_vis_matrix = np.zeros_like(time_matrix)
    time_vis_matrix[valid_mask] = time_matrix[valid_mask]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(time_vis_matrix.reshape(grid_size, grid_size, grid_size, grid_size).mean(axis=(0, 1)), 
                cmap='coolwarm', annot=False)
    plt.title(f'Average Travel Time Heatmap at Time Step {time_step} (Destination)')
    plt.xlabel('Destination X')
    plt.ylabel('Destination Y')
    plt.savefig(os.path.join(output_dir, f'time_step_{time_step}/travel_time_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"时间步 {time_step} 的热力图已保存到 {os.path.join(output_dir, f'time_step_{time_step}')}")

def main():
    parser = argparse.ArgumentParser(description='可视化增强版OD矩阵')
    parser.add_argument('--input', type=str, default='Data/enhanced_od_matrix.h5', help='输入OD矩阵文件路径')
    parser.add_argument('--output-dir', type=str, default='Visualizations', help='输出可视化结果目录')
    parser.add_argument('--time-step', type=int, default=0, help='要可视化的时间步 (默认为0)')
    parser.add_argument('--grid-size', type=int, default=32, help='网格大小 (默认32x32)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载OD矩阵
    od_matrix, adjacency_map = load_od_matrix(args.input)
    
    # 可视化时间模式
    visualize_temporal_pattern(od_matrix, args.output_dir)
    
    # 可视化平均旅行时间
    visualize_avg_travel_time(od_matrix, args.output_dir)
    
    # 可视化特定时间步的OD矩阵热力图
    time_steps = od_matrix.shape[0]
    if args.time_step >= 0 and args.time_step < time_steps:
        visualize_od_heatmap(od_matrix, args.time_step, args.output_dir, args.grid_size)
    else:
        print(f"无效的时间步: {args.time_step}，应在 [0, {time_steps-1}] 范围内")
        # 可视化所有时间步
        for t in range(min(3, time_steps)):  # 为了演示，只可视化前3个时间步
            visualize_od_heatmap(od_matrix, t, args.output_dir, args.grid_size)
    
    print("可视化完成！")

if __name__ == "__main__":
    main() 