import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HawkesProcess:
    """
    Hawkes过程模型，用于预测每个时间步和网格的新生车辆数量
    
    参数:
    - alpha: 历史事件影响的幅度
    - beta: 历史事件影响的衰减率
    - mu: 基线强度
    """
    def __init__(self, alpha=0.8, beta=1.0, mu=0.2):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.models = {}  # 为每个entity_id存储单独的参数
        
    def intensity(self, t, history):
        """计算t时刻的强度函数值"""
        if len(history) == 0:
            return self.mu
        
        # 基线强度 + 所有历史事件的影响总和
        influence = np.sum(self.alpha * np.exp(-self.beta * (t - np.array(history))))
        return self.mu + influence
    
    def log_likelihood(self, params, times, T):
        """计算对数似然函数"""
        alpha, beta, mu = params
        if alpha < 0 or beta <= 0 or mu <= 0:
            return np.inf  # 参数必须为正
        
        # 计算第一部分: 每个事件点的强度函数的对数和
        first_term = 0
        for i, t in enumerate(times):
            history = times[:i]
            intensity_t = mu
            if len(history) > 0:
                intensity_t += np.sum(alpha * np.exp(-beta * (t - np.array(history))))
            first_term += np.log(intensity_t)
        
        # 计算第二部分: 强度函数在[0,T]上的积分
        second_term = mu * T
        for t in times:
            second_term += (alpha / beta) * (1 - np.exp(-beta * (T - t)))
            
        return -first_term + second_term
    
    def fit(self, data):
        """
        训练Hawkes过程模型
        
        参数:
        - data: 包含'entity_id', 'time_step', 'count'的DataFrame
        """
        print("开始训练Hawkes模型...")
        unique_entities = data['entity_id'].unique()
        
        for entity in tqdm(unique_entities, desc="训练实体"):
            entity_data = data[data['entity_id'] == entity].sort_values('time_step')
            
            # 提取时间点和计数
            times = entity_data['time_step'].values
            counts = entity_data['count'].values
            
            # 展开时间序列，为每个事件创建一个时间点
            event_times = []
            for t, count in zip(times, counts):
                event_times.extend([t] * count)
            
            if len(event_times) <= 1:
                # 如果数据点太少，使用默认参数
                self.models[entity] = (self.alpha, self.beta, self.mu)
                continue
                
            # 优化参数
            T = max(times) + 1  # 观察期
            initial_params = [self.alpha, self.beta, self.mu]
            
            try:
                result = minimize(
                    lambda params: self.log_likelihood(params, event_times, T),
                    initial_params,
                    bounds=[(0.001, 10), (0.001, 10), (0.001, 10)],
                    method='L-BFGS-B'
                )
                self.models[entity] = tuple(result.x)
            except Exception as e:
                print(f"实体 {entity} 优化失败: {e}")
                self.models[entity] = (self.alpha, self.beta, self.mu)
        
        print("Hawkes模型训练完成!")
    
    def predict(self, entity_id, time_step, history_times):
        """
        预测特定实体和时间步的新生车辆数量
        
        参数:
        - entity_id: 实体ID
        - time_step: 要预测的时间步
        - history_times: 历史事件时间点
        
        返回:
        - 预测的新生车辆数量
        """
        if entity_id not in self.models:
            alpha, beta, mu = self.alpha, self.beta, self.mu
        else:
            alpha, beta, mu = self.models[entity_id]
        
        # 计算当前时间步的强度函数值
        intensity = mu
        if len(history_times) > 0:
            history_times = np.array(history_times)
            intensity += np.sum(alpha * np.exp(-beta * (time_step - history_times)))
        
        # 将强度转换为期望的事件数
        # 在小时间间隔内，强度可近似为事件数的期望值
        expected_count = max(0, intensity - 1)  # 减去1得到新生车辆数
        
        return expected_count
    
    def evaluate(self, test_data):
        """
        评估模型在测试数据上的表现
        
        参数:
        - test_data: 测试数据集
        
        返回:
        - MSE: 均方误差
        - MAE: 平均绝对误差
        """
        predictions = []
        actual = []
        
        # 按实体分组
        grouped = test_data.groupby('entity_id')
        
        for entity_id, group in grouped:
            group = group.sort_values('time_step')
            times = group['time_step'].values
            counts = group['count'].values
            
            for i in range(1, len(times)):
                history_times = times[:i]
                pred = self.predict(entity_id, times[i], history_times)
                predictions.append(pred)
                actual.append(counts[i] - 1)  # 减去1得到新生车辆数
        
        predictions = np.array(predictions)
        actual = np.array(actual)
        
        mse = np.mean((predictions - actual) ** 2)
        mae = np.mean(np.abs(predictions - actual))
        
        return mse, mae

def load_and_process_data(file_path):
    """
    加载并处理数据文件
    
    参数:
    - file_path: 数据文件路径
    
    返回:
    - 处理后的DataFrame，包含entity_id, time_step, 和count列
    """
    # 假设数据格式，需要根据实际情况调整
    df = pd.read_csv(file_path, sep='\t', header=None)
    
    # 假设格式为: entity_id, time_step, grid_id, count等
    df.columns = ['entity_id', 'time_step', 'grid_id', 'count', 'other_features']
    
    # 对于Hawkes模型，我们只关心entity_id, time_step和count
    processed_df = df[['entity_id', 'time_step', 'count']]
    
    # 第一个数据点是对应时间步和网格的新生车辆数加一
    processed_df['count'] = processed_df['count'] + 1
    
    return processed_df

if __name__ == "__main__":
    # 加载数据
    train_data = load_and_process_data('train.txt')
    val_data = load_and_process_data('val.txt')
    test_data = load_and_process_data('test.txt')
    
    # 初始化并训练模型
    hawkes_model = HawkesProcess()
    hawkes_model.fit(train_data)
    
    # 在验证集上评估
    val_mse, val_mae = hawkes_model.evaluate(val_data)
    print(f"验证集结果 - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
    
    # 在测试集上评估
    test_mse, test_mae = hawkes_model.evaluate(test_data)
    print(f"测试集结果 - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}") 