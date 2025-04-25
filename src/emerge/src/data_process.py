import numpy as np
import pandas as pd
from tqdm import tqdm

def load_and_process_data(file_path, time_start=0, time_end=10079):
    """
    加载并处理数据文件
    
    参数:
    - file_path: 数据文件路径
    - time_start: 开始时间步
    - time_end: 结束时间步
    
    返回:
    - 处理后的数据，形状为(时间步数量, 网格大小, 网格大小)的numpy数组
    """
    print(f"加载数据文件: {file_path}")
    print(f"处理时间步 {time_start} 到 {time_end}")
    
    # 读取数据
    # 修改列名定义（包含所有5列）
    column_names = ['entity_id', 'track_id', 'time', 'lon', 'lat']

    # 读取数据时指定正确的列名
    chunks = pd.read_csv(
        file_path, 
        sep='\t', 
        header=None, 
        names=column_names,  # 使用正确的5列名
        skiprows=1,  # 跳过原始标题行
        usecols=['entity_id', 'time', 'lon', 'lat'],  # 明确指定需要使用的列（可选）
        chunksize=100000
    )

    # 过滤所需的时间步
    filtered_data = []
    
    for chunk in tqdm(chunks, desc="读取数据"):
        # 确保time_step列是整数类型
        chunk['time'] = chunk['time'].astype(int)
        # 只保留指定时间范围内的数据
        time_filtered = chunk[(chunk['time'] >= time_start) & (chunk['time'] <= time_end)]
        if not time_filtered.empty:
            filtered_data.append(time_filtered)
    
    if not filtered_data:
        raise ValueError(f"在时间范围 {time_start} 到 {time_end} 内没有找到数据")
    
    # 合并所有过滤后的数据块
    df = pd.concat(filtered_data)
    print(f"过滤后的数据量: {len(df)} 行")
    
    # 将经纬度转换为网格索引
    # 这里假设数据已经在一个固定的地理区域内，我们需要将其映射到32x32的网格
    # 确定经纬度的边界
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    
    # 计算网格大小
    grid_size = 32
    lon_step = (lon_max - lon_min) / grid_size
    lat_step = (lat_max - lat_min) / grid_size
    
    # 分配网格索引
    df['grid_x'] = ((df['lon'] - lon_min) / lon_step).astype(int).clip(0, grid_size-1)
    df['grid_y'] = ((df['lat'] - lat_min) / lat_step).astype(int).clip(0, grid_size-1)
    
    # 创建一个时间步 x 网格 x 网格的张量来存储新生车辆数
    time_steps = sorted(df['time'].unique())
    time_steps_count = len(time_steps)
    
    new_vehicles = np.zeros((time_steps_count, grid_size, grid_size))
    
    # 创建时间步映射
    time_to_idx = {t: i for i, t in enumerate(time_steps)}
    
    print("计算每个时间步和网格的新生车辆数...")
    
    # 首先检查唯一实体ID数量
    unique_entities = df['entity_id'].nunique()
    print(f"唯一实体ID数量: {unique_entities}")

    # 检查时间步数量
    unique_times = df['time'].nunique()
    print(f"唯一时间步数量: {unique_times}")

    # 修改分组处理代码
    cnt = 0

    # 直接检查首次出现
    first_appearances = df.sort_values('time').drop_duplicates(subset='entity_id', keep='first')
    print(f"首次出现记录数: {len(first_appearances)}")

    for _, row in tqdm(first_appearances.iterrows(), desc="处理车辆"):
        time = row['time']
        if time in time_to_idx:  # 确保时间在映射中
            t_idx = time_to_idx[time]
            grid_x = int(row['grid_x'])
            grid_y = int(row['grid_y'])
            new_vehicles[t_idx, grid_x, grid_y] += 1
            cnt += 1
    
    print(f"数据处理完成。输出形状: {new_vehicles.shape}")
    print(f"cnt = {cnt}")
    return new_vehicles, time_steps, time_to_idx

def split_train_test(data, time_steps, train_end=7199):
    """
    将数据分为训练集和测试集
    
    参数:
    - data: 形状为(时间步数量, 网格大小, 网格大小)的numpy数组
    - time_steps: 时间步列表
    - train_end: 训练集的最后一个时间步
    
    返回:
    - train_data: 训练数据
    - test_data: 测试数据
    """
    # 找出训练集结束的索引
    train_end_idx = time_steps.index(train_end)
    
    # 分割数据
    train_data = data[:train_end_idx+1]
    test_data = data[train_end_idx+1:]
    
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    
    return train_data, test_data

if __name__ == "__main__":
    # 加载并处理数据
    data, time_steps, time_to_idx = load_and_process_data(
        "Data/MGF/txt/BJ_MGF_processed_10080.txt",
        time_start=7200,
        time_end=10079
    )
    
    # 分割训练集和测试集
    train_data, test_data = split_train_test(data, time_steps)
    
    # 保存处理后的数据
    np.save("NewBornFlow/train_data.npy", train_data)
    np.save("NewBornFlow/test_data.npy", test_data)
    # np.save("time_steps.npy", np.array(time_steps))
    
    print("数据已保存到 train_data.npy 和 test_data.npy") 