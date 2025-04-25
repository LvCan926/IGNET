import numpy as np
import pandas as pd
import argparse
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import h5py

# 预定义8个邻域的相对位置（顺序固定）
ADJACENCY_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

def create_global_adjacency_map(grid_size):
    """
    创建全局邻域映射表，保证所有网格使用相同的邻域顺序
    返回字典：{grid_index: [neighbor1, neighbor2, ...]}
    """
    adjacency_map = {}
    for grid_index in range(grid_size * grid_size):
        y, x = grid_index // grid_size, grid_index % grid_size
        neighbors = []
        
        # 按预定义的偏移顺序遍历8个方向
        for dy, dx in ADJACENCY_OFFSETS:
            ny, nx = y + dy, x + dx
            
            # 检查边界，只添加有效的邻域
            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                neighbor_index = ny * grid_size + nx
                neighbors.append(neighbor_index)
        
        adjacency_map[grid_index] = neighbors
    
    return adjacency_map

def is_adjacent(grid1, grid2, grid_size):
    """
    判断两个网格是否相邻（包括对角线相邻）
    grid1和grid2是一维索引，需要先转换为二维坐标
    """
    x1, y1 = grid1 % grid_size, grid1 // grid_size
    x2, y2 = grid2 % grid_size, grid2 // grid_size
    
    # 计算曼哈顿距离
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    
    # 相邻网格（包括对角线相邻）
    return dx <= 1 and dy <= 1 and (dx + dy > 0)

def get_adjacency_index(grid_index, adj_grid, adjacency_map):
    """
    查找邻域网格在邻域列表中的索引位置
    返回索引值，若不在列表中则返回-1
    """
    try:
        return adjacency_map[grid_index].index(adj_grid) + 1  # +1是因为索引0保留给自身
    except ValueError:
        return -1

def process_trajectory_data(file_path, time_steps=12, grid_size=32, time_offset=1128, 
                            lon_range=(116.0, 117.0), lat_range=(39.6, 40.6), sample_ratio=1.0):
    """处理轨迹数据，计算增强版OD矩阵（只考虑相邻网格间流动）"""
    print(f"正在读取数据: {file_path}")
    # 读取数据
    df = pd.read_csv(file_path, sep='\t')
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 数据采样
    if sample_ratio < 1.0:
        # 获取唯一的entity_id
        unique_entities = df['entity_id'].unique()
        
        # 随机选择一部分entity_id
        sample_size = int(len(unique_entities) * sample_ratio)
        sampled_entities = random.sample(list(unique_entities), sample_size)
        
        # 过滤数据
        df = df[df['entity_id'].isin(sampled_entities)]
        print(f"采样后数据量: {len(df)} 条记录, {len(sampled_entities)} 个实体")
    
    # 创建全局邻域映射表
    global_adj_map = create_global_adjacency_map(grid_size)
    
    # 创建一个字典来存储所有相邻网格间的流动数据
    # 格式: {(time_step, current_grid, adj_idx): [count, total_time]}
    flow_data = {}
    
    # 按entity_id分组处理
    print("按entity_id分组处理...")
    entity_groups = df.groupby('entity_id')
    total_entities = len(entity_groups)
    print(f"共 {total_entities} 个实体")
    
    processed_entities = 0
    valid_transfers = 0  # 表示有效的转移记录
    
    # 打印数据的前几行，检查字段格式
    print("数据样例:")
    print(df.head())
    print("可用列:", df.columns.tolist())
    
    # 检查时间范围是否在有效区间内
    min_time = df['time'].min()
    max_time = df['time'].max()
    print(f"数据时间范围: {min_time} - {max_time}")
    print(f"有效时间范围: {time_offset} - {time_offset + time_steps - 1}")
    
    if min_time > time_offset + time_steps - 1 or max_time < time_offset:
        print("警告: 轨迹数据时间范围与指定的时间范围没有重叠")
    
    non_adjacent_count = 0
    
    for entity_id, entity_data in entity_groups:
        processed_entities += 1
        if processed_entities % 1000 == 0:
            print(f"已处理 {processed_entities}/{total_entities} 个实体")
            
        # 按时间排序
        entity_data = entity_data.sort_values(by='time')
        
        if len(entity_data) <= 1:
            continue
        
        # 处理每个连续的点对
        for i in range(len(entity_data) - 1):
            current_point = entity_data.iloc[i]
            next_point = entity_data.iloc[i + 1]
            
            current_time = current_point['time']
            next_time = next_point['time']
            
            # 计算当前时间步
            time_step = int(current_time - time_offset)
            
            # 检查时间步是否在有效范围内
            if time_step < 0 or time_step >= time_steps:
                continue
            
            # 计算当前网格和下一个网格的索引
            current_grid = map_to_grid_index(current_point['lon'], current_point['lat'], grid_size, lon_range, lat_range)
            next_grid = map_to_grid_index(next_point['lon'], next_point['lat'], grid_size, lon_range, lat_range)
            
            # 如果网格相同，则跳过（没有流动）
            if current_grid == next_grid:
                continue
            
            # 只考虑相邻网格之间的流动（包括对角线相邻）
            if is_adjacent(current_grid, next_grid, grid_size):
                # 查找邻域索引位置
                adj_idx = get_adjacency_index(current_grid, next_grid, global_adj_map)
                
                # 确保是找到了有效的邻域索引
                if adj_idx == -1:
                    continue
                
                # 计算时间差
                time_diff = next_time - current_time
                
                # 更新流动数据
                key = (time_step, current_grid, adj_idx)
                if key not in flow_data:
                    flow_data[key] = [0, 0]
                    
                flow_data[key][0] += 1  # 增加计数
                flow_data[key][1] += time_diff  # 累加时间差
                
                valid_transfers += 1
            else:
                non_adjacent_count += 1
    
    print(f"处理完成！有效流动记录数: {valid_transfers}")
    print(f"非相邻流动记录数: {non_adjacent_count}")
    
    # 找出每个网格的最大邻域数量
    max_neighbors = max(len(neighbors) for neighbors in global_adj_map.values())
    print(f"每个网格最多有 {max_neighbors} 个邻域")
    
    # 初始化最终OD矩阵
    # 形状: [时间步, 网格数, max_neighbors+1, 2] 
    # 其中max_neighbors+1表示当前网格(索引0)和它的所有相邻网格
    # 最后一维的2表示 [流动比例, 平均时间]
    od_matrix = np.zeros((time_steps, grid_size*grid_size, max_neighbors+1, 2))
    
    # 计算每个时间步内，每个网格的总流出量
    grid_outflow_totals = {}
    for (t, current, adj_idx), (count, _) in flow_data.items():
        key = (t, current)
        if key not in grid_outflow_totals:
            grid_outflow_totals[key] = 0
        grid_outflow_totals[key] += count
    
    # 填充OD矩阵
    print("正在计算最终相邻网格OD矩阵...")
    
    # 遍历所有网格和时间步
    for t in range(time_steps):
        for grid_index in range(grid_size * grid_size):
            # 检查该网格在该时间步是否有流出
            if (t, grid_index) not in grid_outflow_totals or grid_outflow_totals[(t, grid_index)] == 0:
                continue
                
            # 总流出量
            total_outflow = grid_outflow_totals[(t, grid_index)]
            
            # 初始化第0个索引为自身网格，比例为0（不流动）
            od_matrix[t, grid_index, 0, 0] = 0
            od_matrix[t, grid_index, 0, 1] = 0
            
            # 处理每个邻域位置
            for adj_idx in range(1, max_neighbors+1):  # 邻域索引从1开始
                # 检查是否有流向该邻域位置的记录
                key = (t, grid_index, adj_idx)
                if key in flow_data:
                    count, total_time = flow_data[key]
                    
                    # 计算比例和平均时间
                    ratio = count / total_outflow
                    avg_time = total_time / count if count > 0 else 0
                    
                    # 存储到OD矩阵
                    od_matrix[t, grid_index, adj_idx, 0] = ratio
                    od_matrix[t, grid_index, adj_idx, 1] = avg_time
    
    # 验证每个网格的流出比例总和是否为1
    for t in range(time_steps):
        for grid in range(grid_size * grid_size):
            total_ratio = od_matrix[t, grid, :, 0].sum()
            # 如果有流出，则比例和应该接近1
            if total_ratio > 0 and not np.isclose(total_ratio, 1.0, atol=1e-6):
                print(f"警告: 时间步{t}，网格{grid}的流出比例总和为{total_ratio}，应为1.0")
    
    return od_matrix, global_adj_map

def map_to_grid_index(lon, lat, grid_size, lon_range, lat_range):
    """
    将经纬度映射到网格索引
    """
    MIN_LON, MAX_LON = lon_range
    MIN_LAT, MAX_LAT = lat_range
    
    # 计算网格索引
    x = int((lon - MIN_LON) / (MAX_LON - MIN_LON) * grid_size)
    y = int((lat - MIN_LAT) / (MAX_LAT - MIN_LAT) * grid_size)
    
    # 确保在有效范围内
    x = max(0, min(x, grid_size-1))
    y = max(0, min(y, grid_size-1))
    
    # 转换为一维索引
    return y * grid_size + x

def save_od_matrix(od_matrix, adjacency_map, output_file):
    """保存OD矩阵和邻域映射到文件"""
    # 将输出文件扩展名改为.h5
    if not output_file.endswith('.h5'):
        output_file = os.path.splitext(output_file)[0] + '.h5'
    
    # 使用h5py保存数据
    with h5py.File(output_file, 'w') as f:
        # 保存OD矩阵
        f.create_dataset('od_matrix', data=od_matrix, compression='gzip')
        
        # 保存邻域映射（需要转换为可以存储在HDF5的格式）
        # 创建一个组来存储邻域映射
        adj_group = f.create_group('adjacency_map')
        
        # 遍历邻域映射并保存
        for grid_index, neighbors in adjacency_map.items():
            adj_group.create_dataset(str(grid_index), data=np.array(neighbors))
    
    print(f"增强版OD矩阵已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='计算增强版OD矩阵（只考虑相邻网格流动）')
    parser.add_argument('--input', type=str, default='Data/MGF/txt/BJ_MGF_processed_10080.txt', help='输入轨迹数据文件路径')
    parser.add_argument('--output', type=str, default='NewBornFlow/Data/enhanced_od_matrix.h5', help='输出OD矩阵文件路径')
    parser.add_argument('--time-steps', type=int, default=12, help='时间步数量(默认12)')
    parser.add_argument('--time-offset', type=int, default=7200, help='时间步偏移量(默认7200)')
    parser.add_argument('--grid-size', type=int, default=32, help='网格大小(默认32x32)')
    parser.add_argument('--min-lon', type=float, default=116.2503565, help='最小经度')
    parser.add_argument('--max-lon', type=float, default=116.50032165, help='最大经度')
    parser.add_argument('--min-lat', type=float, default=39.7997164, help='最小纬度')
    parser.add_argument('--max-lat', type=float, default=39.9999216, help='最大纬度')
    parser.add_argument('--sample-ratio', type=float, default=1.0, help='数据采样比例(0-1之间，默认1.0表示使用全部数据)')
    
    args = parser.parse_args()
    
    print("开始处理轨迹数据...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    lon_range = (args.min_lon, args.max_lon)
    lat_range = (args.min_lat, args.max_lat)
    
    od_matrix, adjacency_map = process_trajectory_data(
        args.input, 
        args.time_steps, 
        args.grid_size, 
        args.time_offset,
        lon_range,
        lat_range,
        args.sample_ratio
    )
    
    # 保存结果
    save_od_matrix(od_matrix, adjacency_map, args.output)
    
    # 打印一些统计信息
    non_zero_entries = np.count_nonzero(od_matrix[:, :, :, 0])
    total_entries = args.time_steps * args.grid_size * args.grid_size * od_matrix.shape[2]
    sparsity = 100 * (1 - non_zero_entries / total_entries)
    
    print(f"OD矩阵形状: {od_matrix.shape}")
    print(f"非零元素数量: {non_zero_entries}")
    print(f"稀疏度: {sparsity:.2f}%")

if __name__ == "__main__":
    main() 