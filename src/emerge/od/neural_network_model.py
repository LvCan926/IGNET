import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class VehicleCountDataset(Dataset):
    """
    用于车辆数量预测的数据集类
    """
    def __init__(self, data, seq_length=5):
        """
        初始化数据集
        
        参数:
        - data: 包含'entity_id', 'time_step', 'count'的DataFrame
        - seq_length: 序列长度，用于创建时间序列样本
        """
        self.data = data
        self.seq_length = seq_length
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        """准备训练样本，为每个实体创建时间序列"""
        samples = []
        
        # 按实体分组
        grouped = self.data.groupby('entity_id')
        
        for entity_id, group in grouped:
            # 按时间排序
            group = group.sort_values('time_step')
            
            # 提取时间步和计数
            times = group['time_step'].values
            counts = group['count'].values - 1  # 减去1得到新生车辆数
            
            # 创建序列样本
            for i in range(len(times) - self.seq_length):
                x_seq = counts[i:i+self.seq_length]
                y_next = counts[i+self.seq_length]
                
                # 添加样本
                samples.append({
                    'entity_id': entity_id,
                    'input_seq': torch.FloatTensor(x_seq),
                    'target': torch.FloatTensor([y_next])
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class LSTMModel(nn.Module):
    """
    基于LSTM的车辆数量预测模型
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # ReLU激活函数确保输出非负
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 输入形状: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        
        # 初始化隐藏状态和记忆单元
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        
        # 应用ReLU确保输出非负
        out = self.relu(out)
        
        return out

class NeuralNetworkPredictor:
    """
    神经网络模型预测器
    """
    def __init__(self, seq_length=5, hidden_size=64, num_layers=2, batch_size=64, epochs=50, lr=0.001):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        
        self.model = LSTMModel(
            input_size=1, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=1
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.history = []
    
    def fit(self, train_data, val_data=None):
        """
        训练模型
        
        参数:
        - train_data: 训练数据
        - val_data: 验证数据（可选）
        """
        print(f"开始训练神经网络模型，设备: {self.device}...")
        
        # 准备数据集
        train_dataset = VehicleCountDataset(train_data, seq_length=self.seq_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if val_data is not None:
            val_dataset = VehicleCountDataset(val_data, seq_length=self.seq_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 训练循环
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # 获取输入和目标
                x_seq = batch['input_seq'].unsqueeze(2).to(self.device)  # 添加特征维度
                y_target = batch['target'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                y_pred = self.model(x_seq)
                
                # 计算损失
                loss = self.criterion(y_pred, y_target)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            val_loss = 0
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        x_seq = batch['input_seq'].unsqueeze(2).to(self.device)
                        y_target = batch['target'].to(self.device)
                        
                        y_pred = self.model(x_seq)
                        loss = self.criterion(y_pred, y_target)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                print(f"Epoch {epoch+1}: 训练损失 = {train_loss:.4f}, 验证损失 = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: 训练损失 = {train_loss:.4f}")
            
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss if val_data is not None else None
            })
        
        print("神经网络模型训练完成!")
    
    def predict(self, entity_data):
        """
        预测特定实体的下一个时间步的车辆数量
        
        参数:
        - entity_data: 包含历史数据的DataFrame
        
        返回:
        - 预测的下一个时间步的车辆数量
        """
        self.model.eval()
        
        # 确保数据按时间排序
        entity_data = entity_data.sort_values('time_step')
        
        # 提取最近的seq_length个时间步的数据
        recent_counts = entity_data['count'].values[-self.seq_length:] - 1  # 减去1得到新生车辆数
        
        if len(recent_counts) < self.seq_length:
            # 如果数据不足，用0填充
            padding = np.zeros(self.seq_length - len(recent_counts))
            recent_counts = np.concatenate([padding, recent_counts])
        
        # 转换为张量
        x = torch.FloatTensor(recent_counts).unsqueeze(0).unsqueeze(2).to(self.device)
        
        # 预测
        with torch.no_grad():
            prediction = self.model(x)
        
        return prediction.item()
    
    def evaluate(self, test_data):
        """
        评估模型在测试数据上的表现
        
        参数:
        - test_data: 测试数据
        
        返回:
        - mse: 均方误差
        - mae: 平均绝对误差
        """
        self.model.eval()
        
        predictions = []
        actuals = []
        
        # 按实体分组
        grouped = test_data.groupby('entity_id')
        
        for entity_id, group in grouped:
            # 确保数据按时间排序
            group = group.sort_values('time_step')
            
            # 只有足够多的时间步才能预测
            if len(group) <= self.seq_length:
                continue
            
            # 对每个预测点进行评估
            for i in range(len(group) - self.seq_length):
                # 截取历史数据
                history = group.iloc[i:i+self.seq_length]
                
                # 预测下一个时间步
                pred = self.predict(history)
                
                # 实际值
                actual = group.iloc[i+self.seq_length]['count'] - 1  # 减去1得到新生车辆数
                
                predictions.append(pred)
                actuals.append(actual)
        
        # 计算指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        return mse, mae
    
    def plot_loss(self):
        """绘制训练和验证损失曲线"""
        epochs = [h['epoch'] for h in self.history]
        train_loss = [h['train_loss'] for h in self.history]
        val_loss = [h['val_loss'] for h in self.history if h['val_loss'] is not None]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b-', label='训练损失')
        
        if len(val_loss) > 0:
            plt.plot(epochs, val_loss, 'r-', label='验证损失')
        
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig('nn_loss.png')
        plt.close()

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
    
    # 对于神经网络模型，我们只关心entity_id, time_step和count
    processed_df = df[['entity_id', 'time_step', 'count']]
    
    # 第一个数据点是对应时间步和网格的新生车辆数加一
    processed_df['count'] = processed_df['count'] + 1
    
    return processed_df

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    train_data = load_and_process_data('train.txt')
    val_data = load_and_process_data('val.txt')
    test_data = load_and_process_data('test.txt')
    
    # 模型参数
    seq_length = 5
    hidden_size = 64
    num_layers = 2
    batch_size = 64
    epochs = 30
    lr = 0.001
    
    # 初始化模型
    model = NeuralNetworkPredictor(
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr
    )
    
    # 训练模型
    model.fit(train_data, val_data)
    
    # 绘制损失曲线
    model.plot_loss()
    
    # 在测试集上评估
    test_mse, test_mae = model.evaluate(test_data)
    print(f"测试集结果 - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}") 