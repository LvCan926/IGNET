import numpy as np
import os
import sys
import argparse
from matplotlib import pyplot as plt
import networkx as nx
from tqdm import tqdm
import h5py

# 添加NewBornFlow目录到系统路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'NewBornFlow'))

# 导入generate_flow模块
from generate_flow import load_od_matrix, generate_flow_from_vehicles

def visualize_grid_data(data, title, save_path=None):
    """可视化网格数据"""
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='流量')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_adjacency_map(adjacency_map, grid_size, output_dir):
    """
    可视化邻域连接结构
    """
    print("生成邻域连接结构可视化...")
    
    # 创建一个无向图
    G = nx.Graph()
    
    # 添加节点和边
    for grid, neighbors in adjacency_map.items():
        # 添加网格节点
        x = grid % grid_size
        y = grid // grid_size
        G.add_node(grid, pos=(x, y))
        
        # 添加连接边
        for neighbor in neighbors:
            G.add_edge(grid, neighbor)
    
    # 获取节点位置
    pos = nx.get_node_attributes(G, 'pos')
    
    # 创建可视化
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_size=10, node_color='blue', width=0.5, with_labels=False)
    plt.title(f"邻域连接结构 (网格大小: {grid_size}x{grid_size})")
    
    # 保存图像
    output_file = os.path.join(output_dir, 'adjacency_map.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"邻域连接结构可视化已保存到 {output_file}")

def generate_random_flow(od_matrix, adjacency_map, output_dir):
    """
    使用OD矩阵生成随机流量数据以进行测试
    """
    print("生成测试流量数据...")
    
    time_steps, grid_count, max_neighbors, _ = od_matrix.shape
    grid_size = int(np.sqrt(grid_count))
    
    # 创建随机新生车辆数据
    random_vehicles = np.random.randint(0, 10, size=(time_steps, grid_size, grid_size))
    
    # 初始化流量数据（流入和流出）
    flow = np.zeros((time_steps, 2, grid_size, grid_size))
    
    # 对每个时间步处理
    for t in tqdm(range(time_steps), desc="计算流量"):
        # 对每个网格处理
        for i in range(grid_size):
            for j in range(grid_size):
                # 当前网格索引
                grid_idx = i * grid_size + j
                
                # 只处理有车辆的网格
                if random_vehicles[t, i, j] > 0:
                    vehicles = random_vehicles[t, i, j]
                    
                    # 确保网格在邻域映射中
                    if grid_idx in adjacency_map:
                        neighbors = adjacency_map[grid_idx]
                        
                        # 处理每个邻域位置（索引从1开始，跳过自身）
                        for adj_idx in range(1, min(max_neighbors, len(neighbors) + 1)):
                            ratio = od_matrix[t, grid_idx, adj_idx, 0]
                            avg_time = od_matrix[t, grid_idx, adj_idx, 1]
                            
                            # 跳过比例为0的情况
                            if ratio <= 0:
                                continue
                                
                            try:
                                if adj_idx - 1 >= len(neighbors):
                                    continue
                                    
                                # 获取目标网格
                                dest_idx = neighbors[adj_idx - 1]
                                
                                # 计算目标网格的坐标
                                dest_i = dest_idx // grid_size
                                dest_j = dest_idx % grid_size
                                
                                # 计算流量
                                flow_count = vehicles * ratio
                                
                                # 根据平均时间确定到达时间
                                if avg_time > 0:
                                    arrival_time = int(t + avg_time)
                                else:
                                    arrival_time = t
                                
                                # 记录流出量
                                flow[t, 0, i, j] += flow_count
                                
                                # 记录流入量（如果在时间范围内）
                                if arrival_time < time_steps:
                                    flow[arrival_time, 1, dest_i, dest_j] += flow_count
                            except (IndexError, ValueError):
                                continue
    
    # 保存可视化
    plt.figure(figsize=(16, 8))
    
    # 流出量
    plt.subplot(121)
    outflow = np.sum(flow[:, 0], axis=0)
    plt.imshow(outflow, cmap='hot')
    plt.colorbar()
    plt.title('总流出量')
    
    # 流入量
    plt.subplot(122)
    inflow = np.sum(flow[:, 1], axis=0)
    plt.imshow(inflow, cmap='hot')
    plt.colorbar()
    plt.title('总流入量')
    
    output_file = os.path.join(output_dir, 'flow_test.png')
    plt.savefig(output_file)
    plt.close()
    
    print(f"流量测试可视化已保存到 {output_file}")
    
    return flow

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

def main():
    parser = argparse.ArgumentParser(description='测试OD矩阵和流量生成')
    parser.add_argument('--od-matrix', type=str, default='Data/enhanced_od_matrix.h5', help='OD矩阵文件路径')
    parser.add_argument('--output-dir', type=str, default='Data/test_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载OD矩阵
    try:
        od_matrix_data = load_od_matrix(args.od_matrix)
        od_matrix, adjacency_map = od_matrix_data
        
        # 检查邻域映射格式
        if adjacency_map is not None:
            if isinstance(adjacency_map, dict):
                print(f"邻域映射为字典格式，包含 {len(adjacency_map)} 个网格")
                
                # 获取一个示例网格的邻域
                sample_grid = list(adjacency_map.keys())[0]
                print(f"网格 {sample_grid} 的邻域: {adjacency_map[sample_grid]}")
                
                # 计算每个网格邻域数
                neighbor_counts = {grid: len(neighbors) for grid, neighbors in adjacency_map.items()}
                print(f"平均邻域数: {np.mean(list(neighbor_counts.values())):.2f}")
                
                # 显示邻域大小分布
                lengths = {}
                for grid, neighbors in adjacency_map.items():
                    length = len(neighbors)
                    if length not in lengths:
                        lengths[length] = 0
                    lengths[length] += 1
                
                print("邻域大小分布:")
                grid_count = len(adjacency_map)
                for length, count in sorted(lengths.items()):
                    print(f"  长度 {length}: {count} 个网格 ({100*count/grid_count:.1f}%)")
                
                # 可视化邻域结构
                grid_size = int(np.sqrt(od_matrix.shape[1]))
                visualize_adjacency_map(adjacency_map, grid_size, args.output_dir)
            else:
                print("邻域映射不是字典格式，无法进行测试")
                return
        else:
            print("未找到有效的邻域映射，无法进行测试")
            return
            
        # 计算比例分布统计
        ratios = od_matrix[:, :, :, 0]
        times = od_matrix[:, :, :, 1]
        
        # 统计每个网格的流出比例总和
        row_sums = np.sum(ratios, axis=2)
        
        # 计算有意义的比例和（非零比例和）
        nonzero_mask = row_sums > 0
        meaningful_sums = row_sums[nonzero_mask]
        
        print(f"流出比例和的平均值: {np.mean(meaningful_sums):.4f}")
        print(f"流出比例和的最小值: {np.min(meaningful_sums):.4f}")
        print(f"流出比例和的最大值: {np.max(meaningful_sums):.4f}")
        
        # 统计平均时间
        nonzero_mask = times > 0
        if np.any(nonzero_mask):
            meaningful_times = times[nonzero_mask]
            print(f"平均出行时间: {np.mean(meaningful_times):.2f} 分钟")
            print(f"最短出行时间: {np.min(meaningful_times):.2f} 分钟")
            print(f"最长出行时间: {np.max(meaningful_times):.2f} 分钟")
        
        # 生成随机流量数据进行测试
        flow = generate_random_flow(od_matrix, adjacency_map, args.output_dir)
        
        print("测试完成！")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 