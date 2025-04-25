import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

class MinMaxNormalizer:
    """最大最小值归一化器"""
    def __init__(self, max_val=None, min_val=None):
        self.max = max_val
        self.min = min_val
        
    def fit(self, data):
        self.max = np.max(data)
        self.min = np.min(data)
        return self
        
    def transform(self, data):
        return (2.0 * data - (self.max + self.min)) / (self.max - self.min)
    
    def inverse_transform(self, data):
        return (data * (self.max - self.min) + (self.max + self.min)) / 2.0
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def create_time_features(data, historical_steps, closeness_steps=3, period_steps=3, trend_steps=3, 
                        T_closeness=1, T_period=24, T_trend=24*7):
    """
    创建多尺度时间特征
    
    参数:
        data: 流量数据 [time_steps, channels, height, width]
        historical_steps: 总的历史时间步数
        closeness_steps: 短期时间步数
        period_steps: 周期（日）时间步数
        trend_steps: 趋势（周）时间步数
        T_closeness: 短期时间间隔
        T_period: 周期时间间隔
        T_trend: 趋势时间间隔
    
    返回:
        X_closeness: 短期特征
        X_period: 周期特征
        X_trend: 趋势特征
    """
    len_total = len(data)
    
    # 确定要跳过的时间步数（用于确保所有特征都可用）
    if trend_steps > 0:
        number_of_skip_hours = T_trend * trend_steps
    elif period_steps > 0:
        number_of_skip_hours = T_period * period_steps
    elif closeness_steps > 0:
        number_of_skip_hours = T_closeness * closeness_steps
    else:
        raise ValueError("至少需要一种时间特征")
    
    # 确保不超过historical_steps
    number_of_skip_hours = min(number_of_skip_hours, historical_steps)
    
    # 创建短期特征
    if closeness_steps > 0:
        X_closeness = data[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(closeness_steps - 1):
            X_closeness = np.concatenate(
                (X_closeness, data[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    else:
        X_closeness = None
    
    # 创建周期（日）特征
    if period_steps > 0:
        X_period = data[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(period_steps - 1):
            X_period = np.concatenate(
                (X_period, data[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]),
                axis=1)
    else:
        X_period = None
    
    # 创建趋势（周）特征
    if trend_steps > 0:
        X_trend = data[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(trend_steps - 1):
            X_trend = np.concatenate(
                (X_trend, data[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]),
                axis=1)
    else:
        X_trend = None
    
    return X_closeness, X_period, X_trend

def load_flow_data(dataset_path, historical_steps, prediction_steps, 
                 closeness_steps=3, period_steps=3, trend_steps=3,
                 T_closeness=1, T_period=24, T_trend=24*7, 
                 test_ratio=0.2, val_ratio=0.1, T=48):
    """
    加载流量数据并处理为模型输入格式
    
    参数:
        dataset_path: 数据集路径
        historical_steps: 历史观测的总时间步数
        prediction_steps: 预测的总时间步数
        closeness_steps: 短期时间步数
        period_steps: 周期（日）时间步数
        trend_steps: 趋势（周）时间步数
        T_closeness, T_period, T_trend: 各时间特征的间隔
        test_ratio, val_ratio: 测试集和验证集比例
        T: 每天的时间步数
    
    返回:
        训练、验证和测试数据
    """
    # 加载数据
    f = h5py.File(dataset_path, 'r')
    all_data = f['data'][:]
    timestamps = f['date'][:]
    f.close()
    
    len_total, feature, map_height, map_width = all_data.shape
    print('all_data shape: ', all_data.shape)
    
    # 数据归一化
    normalizer = MinMaxNormalizer()
    all_data = normalizer.fit_transform(all_data)
    print('归一化后数据: mean=', np.mean(all_data), ' variance=', np.std(all_data))
    
    # 创建多尺度时间特征
    X_closeness, X_period, X_trend = create_time_features(
        all_data, historical_steps, closeness_steps, period_steps, trend_steps, 
        T_closeness, T_period, T_trend)
    
    # 计算最大跳过步数
    if trend_steps > 0:
        number_of_skip_hours = T_trend * trend_steps
    elif period_steps > 0:
        number_of_skip_hours = T_period * period_steps
    elif closeness_steps > 0:
        number_of_skip_hours = T_closeness * closeness_steps
    number_of_skip_hours = min(number_of_skip_hours, historical_steps)
    
    # 目标值（实际流量）
    Y = all_data[number_of_skip_hours:len_total]
    
    # 分割数据集
    len_test = int(len(Y) * test_ratio)
    len_val = int(len(Y) * val_ratio)
    len_train = len(Y) - len_test - len_val
    
    # 训练集
    train_end = len_train
    X_closeness_train = X_closeness[:train_end] if X_closeness is not None else None
    X_period_train = X_period[:train_end] if X_period is not None else None
    X_trend_train = X_trend[:train_end] if X_trend is not None else None
    Y_train = Y[:train_end]
    
    # 验证集
    val_start, val_end = train_end, train_end + len_val
    X_closeness_val = X_closeness[val_start:val_end] if X_closeness is not None else None
    X_period_val = X_period[val_start:val_end] if X_period is not None else None
    X_trend_val = X_trend[val_start:val_end] if X_trend is not None else None
    Y_val = Y[val_start:val_end]
    
    # 测试集
    test_start = val_end
    X_closeness_test = X_closeness[test_start:] if X_closeness is not None else None
    X_period_test = X_period[test_start:] if X_period is not None else None
    X_trend_test = X_trend[test_start:] if X_trend is not None else None
    Y_test = Y[test_start:]
    
    print('训练集大小:', len(Y_train))
    print('验证集大小:', len(Y_val))
    print('测试集大小:', len(Y_test))
    
    X_train = [x for x in [X_closeness_train, X_period_train, X_trend_train] if x is not None]
    X_val = [x for x in [X_closeness_val, X_period_val, X_trend_val] if x is not None]
    X_test = [x for x in [X_closeness_test, X_period_test, X_trend_test] if x is not None]
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, normalizer

def load_historical_flow(h5_path, historical_steps, direction_channels, grid_size):
    """
    从h5文件加载历史流量数据
    
    参数:
        h5_path: h5文件路径
        historical_steps: 历史时间步数
        direction_channels: 方向通道数
        grid_size: 网格大小
        
    返回:
        历史流量数据 [historical_steps, direction_channels, grid_size, grid_size]
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"h5文件中的键: {list(f.keys())}")
            
            # 假设h5文件中的流量数据存储在'data'键下
            if 'data' in f:
                historical_flow = f['data'][:]
            else:
                # 尝试找到第一个数据集
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        historical_flow = f[key][:]
                        print(f"从键 '{key}' 加载数据")
                        break
                else:
                    raise ValueError("h5文件中没有找到数据集")
            
            # 打印数据形状以进行调试
            print(f"加载的历史流量数据形状: {historical_flow.shape}")
            
            # 检查数据维度，确保格式正确
            if len(historical_flow.shape) != 4:
                # 如果数据不是4维的，尝试重塑
                print(f"警告: 历史流量数据的维度不正确: {len(historical_flow.shape)}")
                if len(historical_flow.shape) == 3:
                    if historical_flow.shape[0] > 100:  # 可能是 [time, grid, grid]
                        print("尝试将3维数据转换为4维 [time, 2, grid, grid]")
                        # 假设前两个维度是时间和网格，缺少方向通道
                        dup_data = np.repeat(historical_flow[:, np.newaxis, :, :], 2, axis=1)
                        historical_flow = dup_data
                elif len(historical_flow.shape) == 2:
                    if historical_flow.shape[0] > 100:  # 可能是 [time, grid*grid]
                        print("尝试将2维数据转换为4维 [time, 2, grid, grid]")
                        reshaped = historical_flow.reshape(historical_flow.shape[0], 1, grid_size, grid_size)
                        historical_flow = np.repeat(reshaped, 2, axis=1)
            
            # 重新打印处理后的数据形状
            print(f"处理后的历史流量数据形状: {historical_flow.shape}")
            
        # 确保数据维度正确
        expected_shape = (historical_steps, direction_channels, grid_size, grid_size)
        if historical_flow.shape != expected_shape:
            print(f"警告: 历史流量数据的形状不匹配: 期望 {expected_shape}, 实际 {historical_flow.shape}")
            
            # 如果时间步不匹配
            if historical_flow.shape[0] != historical_steps:
                if historical_flow.shape[0] > historical_steps:
                    # 如果数据太多，取最后的historical_steps个时间步
                    historical_flow = historical_flow[-historical_steps:]
                else:
                    # 如果数据太少，通过复制来填充
                    repeat_times = (historical_steps + historical_flow.shape[0] - 1) // historical_flow.shape[0]
                    historical_flow = np.tile(historical_flow, (repeat_times, 1, 1, 1))[:historical_steps]
            
            # 如果方向通道不匹配
            if historical_flow.shape[1] != direction_channels:
                temp = np.zeros((historical_flow.shape[0], direction_channels, grid_size, grid_size))
                for i in range(min(historical_flow.shape[1], direction_channels)):
                    temp[:, i] = historical_flow[:, i]
                historical_flow = temp
                
            # 如果网格大小不匹配
            if historical_flow.shape[2] != grid_size or historical_flow.shape[3] != grid_size:
                zoom_h = grid_size / historical_flow.shape[2]
                zoom_w = grid_size / historical_flow.shape[3]
                historical_flow = zoom(historical_flow, (1, 1, zoom_h, zoom_w))
            
            print(f"调整后的历史流量数据形状: {historical_flow.shape}")
            
        return historical_flow
        
    except Exception as e:
        print(f"加载历史流量数据失败: {e}")
        # 返回一个全零的历史流量数据
        return np.zeros((historical_steps, direction_channels, grid_size, grid_size))

class FlowDataset(Dataset):
    """流量数据集"""
    def __init__(self, historical_data, trajectory_data, targets):
        """
        初始化数据集
        
        参数:
            historical_data: 历史流量数据列表 [closeness, period, trend]
            trajectory_data: 轨迹预测流量数据
            targets: 目标流量数据
        """
        self.historical_data = historical_data
        self.trajectory_data = trajectory_data
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # 获取各时间尺度的历史数据
        hist_data = [x[idx] for x in self.historical_data]
        
        # 获取轨迹数据和目标
        traj_data = self.trajectory_data[idx]
        target = self.targets[idx]
        
        return hist_data, traj_data, target

def create_data_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, 
                       trajectory_train, trajectory_val, trajectory_test,
                       batch_size=32):
    """
    创建数据加载器
    
    参数:
        X_train, Y_train: 训练集
        X_val, Y_val: 验证集
        X_test, Y_test: 测试集
        trajectory_train, trajectory_val, trajectory_test: 轨迹数据
        batch_size: 批大小
    
    返回:
        训练、验证和测试数据加载器
    """
    # 创建数据集
    train_dataset = FlowDataset(X_train, trajectory_train, Y_train)
    val_dataset = FlowDataset(X_val, trajectory_val, Y_val)
    test_dataset = FlowDataset(X_test, trajectory_test, Y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader 