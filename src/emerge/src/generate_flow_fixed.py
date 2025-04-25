import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
import h5py

def load_od_matrix(file_path):
    """
    加载OD矩阵
    
    参数:
    - file_path: OD矩阵文件路径
    
    返回:
    - od_matrix: OD矩阵数据
    - adjacency_map: 邻域映射
    """
    print(f"加载OD矩阵: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到OD矩阵文件: {file_path}")
    
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

def generate_flow_from_vehicles(new_vehicles, od_matrix_data, min_steps=5, max_steps=10, time_offset=1128):
    """
    根据新生车辆数和OD矩阵生成新生流量，确保每辆车至少生成min_steps的流量
    
    参数:
    - new_vehicles: 新生车辆数，形状为(time_steps, grid_size, grid_size)
    - od_matrix_data: OD矩阵数据，元组(od_matrix, adjacency_map)
    - min_steps: 每辆车生成的最小流量步数
    - max_steps: 每辆车生成的最大流量步数
    - time_offset: 时间步偏移量，用于映射新生车辆时间步到OD矩阵时间步
    
    返回:
    - flow: 新生流量，形状为(time_steps, 2, grid_size, grid_size)
    """
    print("根据新生车辆数和OD矩阵生成新生流量...")
    
    # 解析OD矩阵数据
    od_matrix, adjacency_map = od_matrix_data
    
    if adjacency_map is None:
        raise ValueError("缺少邻域映射。请确保使用了正确的OD矩阵格式。")
    
    time_steps, height, width = new_vehicles.shape
    grid_size = height  # 假设height和width相等
    
    # 检查OD矩阵格式
    od_time_steps, od_grid_size, n_neighbors, _ = od_matrix.shape
    assert od_grid_size == grid_size * grid_size, "OD矩阵网格大小与新生车辆网格大小不匹配"
    print(f"OD矩阵有 {n_neighbors} 个邻域索引(包括自身)")
    print("使用标准邻域流动模式")
    
    # 验证邻域映射表格式
    if isinstance(adjacency_map, dict):
        print("使用全局邻域映射字典")
        # 验证映射是否完整
        if len(adjacency_map) != grid_size * grid_size:
            print(f"警告: 邻域映射字典中网格数量 ({len(adjacency_map)}) 与实际网格数量 ({grid_size * grid_size}) 不匹配")
    else:
        raise ValueError("邻域映射不是字典格式。请使用正确的OD矩阵格式。")
    
    # 初始化新生流量：(时间步数, 2, 网格高度, 网格宽度)
    # 第二个维度表示流入和流出
    flow = np.zeros((time_steps, 2, grid_size, grid_size))
    
    # 跟踪每个时间步、每个网格的车辆
    # 我们将模拟每辆车的完整行程，确保每辆车生成足够的流量
    vehicle_tracker = {}  # (t, i, j) -> vehicles_list
    next_vehicle_id = 0
    
    # 第一步：初始化所有新生车辆
    for t in range(time_steps):
        for i in range(grid_size):
            for j in range(grid_size):
                vehicles_count = new_vehicles[t, i, j]
                if vehicles_count > 0:
                    # 创建车辆对象: (id, 起始网格, 当前位置, 行驶步数, 待行驶步数)
                    grid_idx = i * grid_size + j
                    vehicles = []
                    for _ in range(int(vehicles_count)):
                        # 随机决定这辆车需要行驶的总步数
                        steps_to_travel = np.random.randint(min_steps, max_steps+1)
                        vehicles.append((next_vehicle_id, grid_idx, grid_idx, 0, steps_to_travel))
                        next_vehicle_id += 1
                    
                    # 如果有小数部分，添加一辆概率性车辆
                    frac_part = vehicles_count - int(vehicles_count)
                    if frac_part > 0 and np.random.random() < frac_part:
                        steps_to_travel = np.random.randint(min_steps, max_steps+1)
                        vehicles.append((next_vehicle_id, grid_idx, grid_idx, 0, steps_to_travel))
                        next_vehicle_id += 1
                    
                    if vehicles:
                        vehicle_tracker[(t, i, j)] = vehicles
    
    print(f"初始化了 {next_vehicle_id} 辆车")
    
    # 第二步：模拟每辆车的行程
    for t in tqdm(range(time_steps), desc="计算流量"):
        # 获取当前时间步的OD矩阵索引
        od_t = t % od_time_steps
        
        # 处理当前时间步的所有车辆
        for i in range(grid_size):
            for j in range(grid_size):
                if (t, i, j) in vehicle_tracker:
                    vehicles = vehicle_tracker[(t, i, j)]
                    grid_idx = i * grid_size + j
                    
                    # 这些车辆将移动到下一个位置
                    for vehicle_id, start_grid, current_grid, steps_traveled, total_steps in vehicles:
                        # 如果已经行驶了足够的步数，车辆停止
                        if steps_traveled >= total_steps:
                            continue
                        
                        # 车辆还需要继续行驶
                        if current_grid not in adjacency_map:
                            continue  # 如果当前位置不在邻域映射中，车辆停止
                        
                        # 选择下一个网格
                        valid_next_grids = []
                        for adj_idx in range(1, n_neighbors):  # 跳过自身
                            try:
                                if adj_idx - 1 < len(adjacency_map[current_grid]):
                                    dest_idx = adjacency_map[current_grid][adj_idx - 1]
                                    ratio = od_matrix[od_t, current_grid, adj_idx, 0]
                                    
                                    if ratio > 0:
                                        valid_next_grids.append((dest_idx, ratio))
                            except (IndexError, KeyError):
                                continue
                        
                        # 如果没有有效的下一个网格，车辆停止
                        if not valid_next_grids:
                            continue
                        
                        # 根据比例选择下一个网格
                        next_grid_choices, weights = zip(*valid_next_grids)
                        next_grid = np.random.choice(next_grid_choices, p=np.array(weights)/sum(weights))
                        
                        # 计算下一个时间步
                        next_t = t + 1
                        if next_t < time_steps:
                            # 计算新位置的坐标
                            next_i = next_grid // grid_size
                            next_j = next_grid % grid_size
                            
                            # 记录流量
                            flow[t, 0, i, j] += 1  # 流出
                            flow[t, 1, next_i, next_j] += 1  # 流入
                            
                            # 更新车辆位置
                            updated_vehicle = (vehicle_id, start_grid, next_grid, steps_traveled + 1, total_steps)
                            
                            # 将车辆添加到下一个时间步的位置
                            if (next_t, next_i, next_j) not in vehicle_tracker:
                                vehicle_tracker[(next_t, next_i, next_j)] = []
                            vehicle_tracker[(next_t, next_i, next_j)].append(updated_vehicle)
    
    # 后处理：检查和调整全局流入流出总量
    total_out = np.sum(flow[:, 0])
    total_in = np.sum(flow[:, 1])
    
    print(f"流量检查 - 总流出量: {total_out:.2f}, 总流入量: {total_in:.2f}")
    
    # 如果流入量明显与流出量不同，发出警告
    if abs(total_in - total_out) / (total_out + 1e-10) > 0.01:  # 允许1%的误差
        print("警告: 流入量和流出量不完全相等，可能是由于模拟中的边界效应")
    
    # 计算平均每辆车生成的流量
    if next_vehicle_id > 0:
        avg_flow_per_vehicle = total_out / next_vehicle_id
        print(f"平均每辆车生成的流量: {avg_flow_per_vehicle:.2f}")
    
    # 计算每个网格单元的平均流量
    avg_flow_per_cell = total_out / (time_steps * grid_size * grid_size)
    print(f"每个网格单元的平均流量: {avg_flow_per_cell:.4f}")
    
    print("新生流量生成完成！")
    return flow

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成新生流量')
    parser.add_argument('--predicted-vehicles', type=str, default='Data/predicted_vehicles.npy', help='预测车辆数文件路径')
    parser.add_argument('--od-matrix', type=str, default='Data/enhanced_od_matrix.h5', help='OD矩阵文件路径')
    parser.add_argument('--output', type=str, default='Data/new_flow_fixed.npy', help='输出新生流量文件路径')
    parser.add_argument('--min-steps', type=int, default=5, help='每辆车生成的最小流量步数')
    parser.add_argument('--max-steps', type=int, default=7, help='每辆车生成的最大流量步数')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载预测车辆数
    print(f"加载预测车辆数: {args.predicted_vehicles}")
    predicted_vehicles = np.load(args.predicted_vehicles)
    print(f"预测车辆数形状: {predicted_vehicles.shape}")
    print(f"总的新生车辆数: {np.sum(predicted_vehicles)}")
    
    # 加载OD矩阵
    od_matrix_data = load_od_matrix(args.od_matrix)
    
    # 生成新生流量
    flow = generate_flow_from_vehicles(
        predicted_vehicles, 
        od_matrix_data,
        min_steps=args.min_steps,
        max_steps=args.max_steps
    )
    
    # 保存新生流量
    print(f"保存新生流量到: {args.output}")
    np.save(args.output, flow)
    
    return flow

if __name__ == "__main__":
    main() 