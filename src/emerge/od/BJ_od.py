import pandas as pd
from collections import defaultdict
import json

def map_to_grid(lon, lat, grid_size=0.005):
    """将经纬度映射到网格"""

    max_lon = 116.50032165
    min_lon = 116.2503565
    max_lat = 39.9999216
    min_lat = 39.7997164
    col = int((lon - min_lon) // grid_size)
    row = int((lat - min_lat) // grid_size)
    return (row, col)

def process_od_matrix(file_path, grid_size=0.005):
    # 读取数据
    df = pd.read_csv(file_path, sep='\t')
    
    # 初始化OD矩阵结构
    od_matrix = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {'count': 0, 'time_sum': 0}
            )
        )
    )
    
    # 按轨迹分组处理
    for track_id, group in df.groupby('track_id'):
        if len(group) < 2:
            continue  # 单点轨迹无效
        
        # 按时间排序
        sorted_group = group.sort_values('time')
        start = sorted_group.iloc[0]
        
        # 计算出发时间步（时间-1020映射到0~19）
        start_time = start['time']
        time_step = start_time - 1020  # 直接减1020得到时间步
        if not (0 <= time_step < 20):
            continue
        
        # 出发网格
        start_lon, start_lat = start['lon'], start['lat']
        start_grid = map_to_grid(start_lon, start_lat, grid_size)
        
        # 处理轨迹中的每个后续点
        for idx, row in sorted_group.iloc[1:].iterrows():
            current_time = row['time']
            delta_step = current_time - start_time  # 时间步差
            
            # 目标网格
            dest_grid = map_to_grid(row['lon'], row['lat'], grid_size)
            
            # 更新OD矩阵
            od_matrix[time_step][start_grid][dest_grid]['count'] += 1
            od_matrix[time_step][start_grid][dest_grid]['time_sum'] += delta_step
    
    # 计算概率和平均时间
    enhanced_od = defaultdict(dict)
    for time_step in od_matrix:
        enhanced_od[time_step] = {}
        for src in od_matrix[time_step]:
            total = sum(info['count'] for info in od_matrix[time_step][src].values())
            enhanced_od[time_step][src] = {
                dest: {
                    'prob': info['count'] / total,
                    'avg_steps': info['time_sum'] / info['count']
                }
                for dest, info in od_matrix[time_step][src].items()
            }
    
    return enhanced_od

# 使用示例
enhanced_od = process_od_matrix('Data/code/BJ_MGF_processed_17_19.txt')

# 保存结果
with open('enhanced_od.json', 'w') as f:
    json.dump(enhanced_od, f, indent=2)
