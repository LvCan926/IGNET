import pandas as pd
import json
from datetime import datetime
import numpy as np
import time
import os  # 导入 os 模块用于创建目录

# 计算时间编码（每分钟一编码）
def encode_time(time_str):
    base_time = datetime(2015, 11, 1, 0, 0, 0)
    current_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
    time_diff = current_time - base_time
    return int(time_diff.total_seconds() / 60)

# 定义网格映射函数，将经纬度映射到32x32网格
def map_to_grid(lon, lat, min_lon=116.2503565, max_lon=116.50032165,
                min_lat=39.7997164, max_lat=39.9999216, grid_size=32):
    latitude_step = (max_lat - min_lat) / grid_size
    longitude_step = (max_lon - min_lon) / grid_size
    row = int((lat - min_lat) / latitude_step)
    col = int((lon - min_lon) / longitude_step)

    # 确保 row 和 col 在 0 到 grid_size-1 的范围内
    row = max(0, min(row, grid_size - 1))
    col = max(0, min(col, grid_size - 1))

    return row, col

# 加载 JSON 文件中的路段经纬度信息
try:
    with open('data/rid_gps.json', 'r') as f:
        rid_gps = json.load(f)
except FileNotFoundError:
    print("错误: rid_gps.json 文件未找到。请确保该文件位于 data 目录下。")
    exit() # 如果文件不存在，则退出脚本

# 根据路段ID获取对应的经纬度
def get_lon_lat_for_rid(rid):
    rid_str = str(rid) # 将rid转换为字符串进行查找
    if rid_str in rid_gps:
        lon, lat = rid_gps[rid_str]
        return round(lon, 8), round(lat, 8)  # 返回经纬度坐标，保留八位小数
    else:
        # print(f"Warning: No GPS data for rid {rid}") # 减少不必要的打印
        return np.nan, np.nan

# --- 定义参数 ---
input_csv_path = 'data/BJ_Taxi_201511_trajectory.csv'
output_dir = 'data/raw/BJTaxi'
chunk_size = 10000  # 每次处理的行数，根据内存调整

# 定义时间范围
train_end_minute = 5759
val_end_minute = 7199
test_end_minute = 10080

# 初始化列表用于存储分割后的数据
train_data = []
val_data = []
test_data = []

new_entity_id_counter = 0  # 初始化新的 entity_id 计数器 (在所有块之间共享)
processed_rows = 0
start_time = time.time()

print(f"开始处理文件: {input_csv_path}，分块大小: {chunk_size}")
print(f"时间分割点 (分钟): Train <= {train_end_minute}, Val <= {val_end_minute}, Test <= {test_end_minute}")

# 检查输入文件是否存在
if not os.path.exists(input_csv_path):
    print(f"错误: 输入文件 {input_csv_path} 未找到。")
    exit()

# 分块读取和处理CSV
try:
    for chunk_index, chunk in enumerate(pd.read_csv(input_csv_path, chunksize=chunk_size)):
        # 只处理前10个块
        if(chunk_index == 10):
            break
        
        print(f"处理块 {chunk_index + 1}...")
        # 遍历块中的每一行（每一行代表一个原始轨迹）
        for index, row in chunk.iterrows():
            # 提取路段ID和时间信息
            rid_list_str = row.get('rid_list', '') # 使用get防止KeyError
            time_list_str = row.get('time_list', '') # 使用get防止KeyError

            # 确保 rid_list 和 time_list 是字符串类型再分割
            if not isinstance(rid_list_str, str) or not isinstance(time_list_str, str):
                continue # 跳过格式不正确的数据行

            rid_list = rid_list_str.split(',')
            time_list = time_list_str.split(',')

            # 检查rid_list和time_list长度是否一致
            if len(rid_list) != len(time_list):
                continue # 跳过数据不一致的行

            trajectory_has_points = False # 标记当前轨迹是否有有效点加入

            # 对每个路段，获取对应的经纬度并处理时间
            for rid_str, timestamp in zip(rid_list, time_list):
                try:
                    rid = int(rid_str)
                except ValueError:
                    continue # 跳过无效的rid

                lon, lat = get_lon_lat_for_rid(rid)
                if np.isnan(lon) or np.isnan(lat):
                    continue # 如果rid没有对应的经纬度，跳过这个点

                try:
                    time_encoded = encode_time(timestamp) # 使用分钟编码
                except ValueError:
                    continue # 跳过无效的时间戳

                # 根据时间编码分流数据点
                point_data = [time_encoded, new_entity_id_counter, lon, lat]
                if time_encoded <= train_end_minute:
                    train_data.append(point_data)
                    trajectory_has_points = True
                elif time_encoded <= val_end_minute:
                    val_data.append(point_data)
                    trajectory_has_points = True
                elif time_encoded <= test_end_minute:
                    test_data.append(point_data)
                    trajectory_has_points = True
                # else: # 超出时间范围的点可以忽略
                #     pass

            # 只有当轨迹至少有一个有效点被添加时，才增加 entity_id 计数器
            if trajectory_has_points:
                new_entity_id_counter += 1
            processed_rows += 1

        print(f"块 {chunk_index + 1} 处理完毕。已处理 {processed_rows} 行。当前 Entity ID: {new_entity_id_counter}")

except FileNotFoundError:
    print(f"错误: {input_csv_path} 文件在处理过程中丢失或无法访问。")
    exit()
except Exception as e:
    print(f"处理过程中发生错误: {e}")
    exit()

print("所有块处理完毕。开始整理和保存数据...")

# --- 整理、排序和保存结果 ---
os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在
output_columns = ['time', 'entity_id', 'lon', 'lat']

def save_data(data_list, filename):
    if data_list:
        print(f"处理 {filename}... ({len(data_list)} 条记录)")
        df = pd.DataFrame(data_list, columns=['time', 'entity_id', 'lon', 'lat'])
        print(f"  排序 {filename}...")
        df = df.sort_values(by=['time', 'entity_id']).reset_index(drop=True)
        output_path = os.path.join(output_dir, filename)
        print(f"  保存 {filename} 到 {output_path}...")
        df.to_csv(output_path, sep='	', index=False, header=False)
        print(f"  {filename} 保存完毕。")
    else:
        print(f"{filename} 没有数据，跳过保存。")

# 保存训练集、验证集和测试集
save_data(train_data, 'train.txt')
save_data(val_data, 'val.txt')
save_data(test_data, 'test.txt')

end_time = time.time()
print(f"数据处理总耗时：{end_time - start_time:.2f} 秒")
print(f"总共处理了 {processed_rows} 行原始数据，生成了 {new_entity_id_counter} 个轨迹(Entity ID)。")
print(f"数据已分别保存到 {output_dir} 目录下的 train.txt, val.txt, test.txt 文件中。")
