import numpy as np
import h5py
import os
from tqdm import tqdm
import argparse

def load_od_matrix(file_path):
    """加载OD矩阵，只获取邻域映射信息"""
    print(f"加载OD矩阵: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # 加载邻域映射
        adjacency_map = {}
        adj_group = f['adjacency_map']
        for key in adj_group.keys():
            grid_index = int(key)
            adjacency_map[grid_index] = list(adj_group[key][:])
        
    return adjacency_map

def generate_forced_flow(new_vehicles, adjacency_map, steps_per_vehicle=6):
    """
    生成强制流量，确保每辆车生成指定的流量步数
    
    参数:
    - new_vehicles: 新生车辆数，形状为(time_steps, grid_size, grid_size)
    - adjacency_map: 邻域映射
    - steps_per_vehicle: 每辆车强制生成的流量步数
    
    返回:
    - flow: 新生流量，形状为(time_steps, 2, grid_size, grid_size)
    """
    print("生成强制流量...")
    
    time_steps, height, width = new_vehicles.shape
    grid_size = height  # 假设height和width相等
    
    # 初始化新生流量：(时间步数, 2, 网格高度, 网格宽度)
    # 第二个维度表示流入和流出
    flow = np.zeros((time_steps, 2, grid_size, grid_size))
    
    # 跟踪每个时间步、每个网格的车辆
    vehicle_tracker = {}  # (t, i, j) -> vehicles_list
    next_vehicle_id = 0
    
    # 第一步：初始化所有新生车辆
    for t in range(time_steps):
        for i in range(grid_size):
            for j in range(grid_size):
                vehicles_count = new_vehicles[t, i, j]
                if vehicles_count > 0:
                    # 创建车辆对象: (id, 起始网格, 当前位置, 行驶步数)
                    grid_idx = i * grid_size + j
                    vehicles = []
                    for _ in range(int(vehicles_count)):
                        vehicles.append((next_vehicle_id, grid_idx, grid_idx, 0))
                        next_vehicle_id += 1
                    
                    # 如果有小数部分，添加一辆概率性车辆
                    frac_part = vehicles_count - int(vehicles_count)
                    if frac_part > 0 and np.random.random() < frac_part:
                        vehicles.append((next_vehicle_id, grid_idx, grid_idx, 0))
                        next_vehicle_id += 1
                    
                    if vehicles:
                        vehicle_tracker[(t, i, j)] = vehicles
    
    print(f"初始化了 {next_vehicle_id} 辆车")
    
    # 第二步：模拟每辆车的行程
    for t in tqdm(range(time_steps), desc="计算流量"):
        for i in range(grid_size):
            for j in range(grid_size):
                if (t, i, j) in vehicle_tracker:
                    vehicles = vehicle_tracker[(t, i, j)]
                    grid_idx = i * grid_size + j
                    
                    # 这些车辆将移动到下一个位置
                    for vehicle_id, start_grid, current_grid, steps_traveled in vehicles:
                        # 如果已经行驶了足够的步数，车辆停止
                        if steps_traveled >= steps_per_vehicle:
                            continue
                        
                        # 车辆还需要继续行驶
                        next_grid = None
                        
                        # 如果当前位置有邻域映射，从中选择；否则随机选择
                        if current_grid in adjacency_map and adjacency_map[current_grid]:
                            next_grid = np.random.choice(adjacency_map[current_grid])
                        else:
                            # 随机选择一个相邻网格，确保不会超出边界
                            di, dj = np.random.choice([-1, 0, 1], size=2)
                            next_i, next_j = i + di, j + dj
                            
                            # 确保在网格范围内
                            if 0 <= next_i < grid_size and 0 <= next_j < grid_size:
                                next_grid = next_i * grid_size + next_j
                            else:
                                # 如果超出范围，保持原地
                                next_grid = current_grid
                        
                        # 计算下一个时间步
                        next_t = t + 1
                        if next_t < time_steps and next_grid is not None:
                            # 计算新位置的坐标
                            next_i = next_grid // grid_size
                            next_j = next_grid % grid_size
                            
                            # 记录流量
                            flow[t, 0, i, j] += 1  # 流出
                            flow[t, 1, next_i, next_j] += 1  # 流入
                            
                            # 更新车辆位置
                            updated_vehicle = (vehicle_id, start_grid, next_grid, steps_traveled + 1)
                            
                            # 将车辆添加到下一个时间步的位置
                            if (next_t, next_i, next_j) not in vehicle_tracker:
                                vehicle_tracker[(next_t, next_i, next_j)] = []
                            vehicle_tracker[(next_t, next_i, next_j)].append(updated_vehicle)
                        # 如果已经到了最后一个时间步，但仍未行走完应走的步数，也记录部分流量
                        elif next_grid is not None:
                            # 计算新位置的坐标
                            next_i = next_grid // grid_size
                            next_j = next_grid % grid_size
                            
                            # 只记录流出量，不记录流入量(因为已经超出预测范围)
                            flow[t, 0, i, j] += 1  # 流出
    
    # 后处理：检查和调整全局流入流出总量
    total_out = np.sum(flow[:, 0])
    total_in = np.sum(flow[:, 1])
    
    print(f"流量检查 - 总流出量: {total_out:.2f}, 总流入量: {total_in:.2f}")
    
    # 计算平均每辆车生成的流量
    if next_vehicle_id > 0:
        avg_flow_per_vehicle = total_out / next_vehicle_id
        print(f"平均每辆车生成的流量: {avg_flow_per_vehicle:.2f}")
    
    # 计算每个网格单元的平均流量
    avg_flow_per_cell = total_out / (time_steps * grid_size * grid_size)
    print(f"每个网格单元的平均流量: {avg_flow_per_cell:.4f}")
    
    # 计算理论流量
    total_vehicles = np.sum(new_vehicles)
    theoretical_flow = total_vehicles * steps_per_vehicle
    theoretical_avg_flow = theoretical_flow / (time_steps * grid_size * grid_size)
    print(f"总新生车辆数: {total_vehicles}")
    print(f"理论总流量: {theoretical_flow}")
    print(f"理论平均流量: {theoretical_avg_flow:.4f}")
    print(f"实际/理论比例: {total_out/theoretical_flow:.4f}")
    
    return flow

def generate_random_vehicles():
    """
    生成随机车辆数据用于测试
    
    返回:
    - vehicles: 随机生成的车辆数据，形状为(12, 32, 32)
    """
    print("生成随机测试车辆数据...")
    grid_size = 32
    time_steps = 12
    
    # 创建极度稀疏的矩阵 (约5%的值非零)
    vehicles = np.zeros((time_steps, grid_size, grid_size))
    
    # 确定大约5%的随机位置
    total_cells = time_steps * grid_size * grid_size
    nonzero_count = int(total_cells * 0.05)
    
    # 随机生成非零值位置
    for _ in range(nonzero_count):
        t = np.random.randint(0, time_steps)
        i = np.random.randint(0, grid_size)
        j = np.random.randint(0, grid_size)
        
        # 生成1-3范围内的随机值
        vehicles[t, i, j] = np.random.uniform(1, 3)
    
    # 创建几个热点区域 (持续多个时间步的高值区域)
    hotspots = 5
    for _ in range(hotspots):
        i = np.random.randint(5, grid_size-5)
        j = np.random.randint(5, grid_size-5)
        
        # 在热点周围区域生成较高值
        for t in range(time_steps):
            if np.random.random() < 0.7:  # 70%的时间步都有事件
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        if np.random.random() < 0.3:  # 只有30%的邻居有事件
                            ni, nj = i+di, j+dj
                            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                vehicles[t, ni, nj] = np.random.uniform(2, 5)
    
    print(f"随机车辆数据形状: {vehicles.shape}")
    print(f"非零值比例: {np.count_nonzero(vehicles) / vehicles.size:.4f}")
    print(f"总车辆数: {np.sum(vehicles):.2f}")
    
    return vehicles

def generate_random_od_matrix(file_path):
    """
    生成随机OD矩阵数据并保存到h5文件
    
    参数:
    - file_path: 保存路径
    """
    print(f"生成随机OD矩阵数据并保存到: {file_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    grid_size = 32
    total_grids = grid_size * grid_size
    time_steps = 12
    max_neighbors = 9  # 包括自身
    
    # 生成OD矩阵
    od_matrix = np.zeros((time_steps, total_grids, max_neighbors, 2))
    
    # 创建邻域映射
    adjacency_map = {}
    
    # 对于每个网格生成邻域
    for i in range(grid_size):
        for j in range(grid_size):
            grid_idx = i * grid_size + j
            neighbors = []
            
            # 添加自身
            neighbors.append(grid_idx)
            
            # 添加相邻网格（上下左右和对角线）
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # 跳过自身
                    
                    ni, nj = i + di, j + dj
                    # 检查边界
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbor_idx = ni * grid_size + nj
                        neighbors.append(neighbor_idx)
            
            # 存储邻域列表
            adjacency_map[grid_idx] = neighbors
            
            # 生成随机OD概率和时间
            for t in range(time_steps):
                # 自身留存比例 (网格中留下的比例)
                od_matrix[t, grid_idx, 0, 0] = np.random.uniform(0.1, 0.3)
                # 自身到自身的时间为0
                od_matrix[t, grid_idx, 0, 1] = 0
                
                # 生成到邻居的概率和时间
                remaining_prob = 1.0 - od_matrix[t, grid_idx, 0, 0]
                
                # 邻居数量 (不包括自身)
                n_neighbors = len(neighbors) - 1
                
                if n_neighbors > 0:
                    # 为每个邻居随机生成一个权重
                    weights = np.random.random(n_neighbors)
                    # 归一化权重，使总和等于剩余概率
                    weights = weights / weights.sum() * remaining_prob
                    
                    # 为每个邻居分配概率和时间
                    for idx, neighbor in enumerate(neighbors[1:], 1):
                        if idx < max_neighbors:
                            od_matrix[t, grid_idx, idx, 0] = weights[idx-1]  # 概率
                            od_matrix[t, grid_idx, idx, 1] = np.random.uniform(1, 10)  # 时间
    
    # 保存到h5文件
    with h5py.File(file_path, 'w') as f:
        # 创建并保存OD矩阵
        f.create_dataset('od_matrix', data=od_matrix)
        
        # 创建并保存邻域映射
        adj_group = f.create_group('adjacency_map')
        for key, value in adjacency_map.items():
            adj_group.create_dataset(str(key), data=value)
    
    print(f"随机OD矩阵已保存: 形状={od_matrix.shape}")
    print(f"非零元素比例: {np.count_nonzero(od_matrix) / od_matrix.size:.4f}")
    
    return od_matrix, adjacency_map

def generate_flow(vehicles, od_matrix_data):
    """
    根据车辆数和OD矩阵生成流量
    
    参数:
    - vehicles: 车辆数，形状为(time_steps, grid_size, grid_size)
    - od_matrix_data: OD矩阵数据，元组(od_matrix, adjacency_map)
    
    返回:
    - flow: 生成的流量，形状为(time_steps, 2, grid_size, grid_size)
    """
    print("生成强制流量...")
    
    od_matrix, adjacency_map = od_matrix_data
    time_steps, height, width = vehicles.shape
    grid_size = height
    
    # 检查OD矩阵格式
    od_time_steps, od_grid_size, n_neighbors, _ = od_matrix.shape
    
    # 初始化流量：(时间步数, 2, 网格高度, 网格宽度)
    # 第二个维度表示流入和流出
    flow = np.zeros((time_steps, 2, grid_size, grid_size))
    
    # 初始化总车辆数 (用于验证)
    total_vehicles = 0
    
    # 对每个时间步计算流量分配
    for t in tqdm(range(time_steps), desc="计算流量"):
        # 映射到OD矩阵的时间步
        od_t = t % od_time_steps  
        
        # 对每个网格处理车辆
        for i in range(grid_size):
            for j in range(grid_size):
                # 将2D网格索引转换为1D索引
                grid_idx = i * grid_size + j
                
                # 当前网格的车辆数
                current_vehicles = vehicles[t, i, j]
                total_vehicles += current_vehicles
                
                if current_vehicles > 0:
                    # 处理每个邻域位置
                    for adj_idx in range(n_neighbors):
                        # 获取比例和平均时间
                        ratio = od_matrix[od_t, grid_idx, adj_idx, 0]
                        avg_time = od_matrix[od_t, grid_idx, adj_idx, 1]
                        
                        if ratio > 0:
                            try:
                                # 获取目标网格索引
                                if grid_idx not in adjacency_map:
                                    continue
                                    
                                if adj_idx >= len(adjacency_map[grid_idx]):
                                    continue
                                    
                                dest_idx = adjacency_map[grid_idx][adj_idx]
                            except (IndexError, KeyError):
                                continue
                            
                            # 计算流向目标网格的车辆数
                            flow_count = current_vehicles * ratio
                            
                            # 将目标1D索引转换回2D索引
                            dest_i = dest_idx // grid_size
                            dest_j = dest_idx % grid_size
                            
                            # 计算出发和到达的时间步
                            departure_time = t
                            
                            # 根据平均时间确定到达时间
                            if avg_time > 0:
                                arrival_time = int(t + avg_time)
                            else:
                                arrival_time = t + 1
                            
                            # 添加到流出量（在出发时间记录）
                            if departure_time < time_steps:
                                flow[departure_time, 0, i, j] += flow_count
                            
                            # 添加到流入量（在到达时间记录）
                            if arrival_time < time_steps:
                                flow[arrival_time, 1, dest_i, dest_j] += flow_count
    
    # 检查流量平衡
    total_out = np.sum(flow[:, 0])
    total_in = np.sum(flow[:, 1])
    
    print(f"流量检查 - 总流出量: {total_out:.2f}, 总流入量: {total_in:.2f}")
    
    # 计算理论值
    avg_flow = np.sum(flow) / (flow.shape[0] * flow.shape[2] * flow.shape[3] * 2)
    print(f"每个网格单元的平均流量: {avg_flow:.4f}")
    
    print(f"总新生车辆数: {total_vehicles:.1f}")
    
    # 如果所有车辆都会流出，理论上总流出量应该等于总车辆数
    theoretical_flow = total_vehicles
    print(f"理论总流量: {theoretical_flow:.1f}")
    print(f"理论平均流量: {theoretical_flow/(time_steps*grid_size*grid_size*2):.4f}")
    print(f"实际/理论比例: {total_out/theoretical_flow:.4f}")
    
    return flow

def analyze_flow(flow):
    """
    分析流量数据
    
    参数:
    - flow: 流量数据，形状为(time_steps, 2, grid_size, grid_size)
    """
    time_steps, _, height, width = flow.shape
    
    # 基本统计
    print("\n流量分析:")
    print(f"形状: {flow.shape}")
    print(f"最小值: {np.min(flow):.4f}")
    print(f"最大值: {np.max(flow):.4f}")
    print(f"平均值: {np.mean(flow):.4f}")
    print(f"非零值占比: {np.count_nonzero(flow) / flow.size:.4f}")
    
    # 流入流出平衡
    total_out = np.sum(flow[:, 0])
    total_in = np.sum(flow[:, 1])
    print(f"总流出量: {total_out:.2f}")
    print(f"总流入量: {total_in:.2f}")
    print(f"流入/流出比例: {total_in/total_out if total_out > 0 else 0:.4f}")
    
    # 分时间步统计
    print("\n各时间步统计:")
    for t in range(time_steps):
        step_out = np.sum(flow[t, 0])
        step_in = np.sum(flow[t, 1])
        print(f"  时间步 {t}: 流出={step_out:.2f}, 流入={step_in:.2f}, 比例={(step_in/step_out if step_out > 0 else 0):.4f}")
    
    # 空间分布
    print("\n空间分布:")
    grid_out_sum = np.sum(flow[:, 0], axis=0)
    grid_in_sum = np.sum(flow[:, 1], axis=0)
    
    # 前5个最大流出网格
    max_out_indices = np.argsort(grid_out_sum.flatten())[-5:]
    print("前5个最大流出网格:")
    for idx in max_out_indices:
        i, j = idx // width, idx % width
        print(f"  网格({i},{j}): {grid_out_sum[i,j]:.2f}")
    
    # 前5个最大流入网格
    max_in_indices = np.argsort(grid_in_sum.flatten())[-5:]
    print("前5个最大流入网格:")
    for idx in max_in_indices:
        i, j = idx // width, idx % width
        print(f"  网格({i},{j}): {grid_in_sum[i,j]:.2f}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成新生流量')
    parser.add_argument('--predicted-vehicles', type=str, default='NewBornFlow/Data/predicted_vehicles.npy', help='预测车辆数文件路径')
    parser.add_argument('--od-matrix', type=str, default='NewBornFlow/Data/enhanced_od_matrix.h5', help='OD矩阵文件路径')
    parser.add_argument('--output', type=str, default='NewBornFlow/Data/new_flow_forced.npy', help='输出新生流量文件路径')
    parser.add_argument('--force-random', action='store_true', help='强制使用随机生成的测试数据')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 生成或加载预测车辆数
    if args.force_random or not os.path.exists(args.predicted_vehicles):
        print("使用随机生成的车辆数据...")
        vehicles = generate_random_vehicles()
        
        # 保存随机车辆数据
        np.save(args.predicted_vehicles, vehicles)
        print(f"随机车辆数据已保存到: {args.predicted_vehicles}")
    else:
        print(f"加载预测车辆数: {args.predicted_vehicles}")
        vehicles = np.load(args.predicted_vehicles)
        print(f"预测车辆数形状: {vehicles.shape}")
        print(f"总的新生车辆数: {np.sum(vehicles)}")
    
    # 生成或加载OD矩阵
    if args.force_random or not os.path.exists(args.od_matrix):
        print(f"OD矩阵文件不存在: {args.od_matrix}")
        od_matrix_data = generate_random_od_matrix(args.od_matrix)
    else:
        print(f"加载OD矩阵: {args.od_matrix}")
        from generate_flow import load_od_matrix
        try:
            od_matrix_data = load_od_matrix(args.od_matrix)
        except FileNotFoundError:
            print(f"无法加载OD矩阵，生成随机数据")
            od_matrix_data = generate_random_od_matrix(args.od_matrix)
    
    # 生成新生流量
    flow = generate_flow(vehicles, od_matrix_data)
    
    # 分析流量
    analyze_flow(flow)
    
    # 保存新生流量
    print(f"保存新生流量到: {args.output}")
    np.save(args.output, flow)
    
    print("新生流量生成完成！")
    
    return flow

if __name__ == "__main__":
    main()