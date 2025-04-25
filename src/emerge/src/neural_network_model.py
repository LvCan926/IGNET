import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class VehicleGridDataset(Dataset):
    """
    车辆网格数据集
    """
    def __init__(self, data, seq_length=5):
        """
        初始化数据集
        
        参数:
        - data: 形状为(时间步数, 网格大小, 网格大小)的数组
        - seq_length: 输入序列长度
        """
        self.data = data
        self.seq_length = seq_length
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        """准备训练样本"""
        n_steps, height, width = self.data.shape
        samples = []
        
        # 对于每个可能的输入序列位置
        for i in range(n_steps - self.seq_length):
            # 输入序列：时间步i到i+seq_length-1的数据
            x_seq = self.data[i:i+self.seq_length]
            
            # 目标：时间步i+seq_length的数据
            y_target = self.data[i+self.seq_length]
            
            samples.append({
                'input_seq': torch.FloatTensor(x_seq),
                'target': torch.FloatTensor(y_target)
            })
        
        return samples
    
    def __len__(self):
        """返回样本数量"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.samples[idx]

class ConvLSTMCell(nn.Module):
    """
    卷积LSTM单元
    将卷积和LSTM结合，以捕获时空特征
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        初始化卷积LSTM单元
        
        参数:
        - input_dim: 输入特征的通道数
        - hidden_dim: 隐藏状态的通道数
        - kernel_size: 卷积核大小
        - bias: 是否使用偏置
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # 卷积层：计算输入门、遗忘门、单元状态和输出门
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # i, f, c, o gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        """
        前向计算
        
        参数:
        - input_tensor: 输入张量，形状为(batch_size, input_dim, height, width)
        - cur_state: 当前状态(h, c)
        
        返回:
        - 下一个状态(h, c)
        """
        h_cur, c_cur = cur_state
        
        # 合并输入和隐藏状态
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # 计算门控值
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # 应用激活函数
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)     # 候选细胞状态
        
        # 更新细胞状态和隐藏状态
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width, device):
        """
        初始化隐藏状态
        
        参数:
        - batch_size: 批大小
        - height: 输入高度
        - width: 输入宽度
        - device: 设备
        
        返回:
        - 初始化的(h, c)
        """
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c

class ConvLSTM(nn.Module):
    """
    多层卷积LSTM网络
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        """
        初始化卷积LSTM网络
        
        参数:
        - input_dim: 输入特征的通道数
        - hidden_dim: 每层隐藏状态的通道数，可以是列表，表示每层的维度
        - kernel_size: 卷积核大小
        - num_layers: 层数
        - bias: 是否使用偏置
        """
        super(ConvLSTM, self).__init__()
        
        # 确保参数正确
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        
        # 创建多层卷积LSTM
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size,
                    bias=self.bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor, hidden_state=None):
        """
        前向计算
        
        参数:
        - input_tensor: 输入张量，形状为(batch_size, seq_len, channels, height, width)
        - hidden_state: 初始隐藏状态
        
        返回:
        - 输出张量，形状为(batch_size, seq_len, hidden_dim[-1], height, width)
        - 最终隐藏状态列表
        """
        batch_size, seq_len, _, height, width = input_tensor.size()
        device = input_tensor.device
        
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, device)
        
        # 输出的所有层和所有时间步的隐藏状态
        layer_output_list = []
        last_state_list = []
        
        # 处理每一层
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # 处理序列中的每一个时间步
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)
            
            # 为下一层准备输入
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        return layer_output_list[-1], last_state_list
    
    def _init_hidden(self, batch_size, height, width, device):
        """初始化所有层的隐藏状态"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, height, width, device)
            )
        return init_states

class VehicleGridPredictor(nn.Module):
    """
    基于卷积LSTM的车辆网格预测模型
    """
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, num_layers=2):
        """
        初始化模型
        
        参数:
        - input_dim: 输入通道数
        - hidden_dim: 隐藏状态通道数
        - kernel_size: 卷积核大小
        - num_layers: 卷积LSTM层数
        """
        super(VehicleGridPredictor, self).__init__()
        
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers
        )
        
        # 输出层：将隐藏状态映射到预测结果
        self.conv_output = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=True
        )
    
    def forward(self, x):
        """
        前向计算
        
        参数:
        - x: 输入张量，形状为(batch_size, seq_len, grid_height, grid_width)
        
        返回:
        - 预测结果，形状为(batch_size, grid_height, grid_width)
        """
        batch_size, seq_len, height, width = x.size()
        
        # 调整维度顺序为(batch_size, seq_len, channels, height, width)
        x = x.unsqueeze(2)  # 添加通道维度
        
        # 通过卷积LSTM
        lstm_output, _ = self.convlstm(x)
        
        # 只使用最后一个时间步的输出
        last_output = lstm_output[:, -1]
        
        # 通过输出层
        out = self.conv_output(last_output)
        
        # 移除通道维度
        out = out.squeeze(1)
        
        # 使用ReLU确保输出非负
        out = F.relu(out)
        
        return out

class NNGridModel:
    """
    神经网络网格模型，用于预测每个时间步所有网格的新生车辆数目
    """
    def __init__(self, seq_length=5, hidden_dim=64, num_layers=2, batch_size=16, epochs=30, lr=0.001, nonzero_weight=5.0):
        """
        初始化模型
        
        参数:
        - seq_length: 输入序列长度
        - hidden_dim: 隐藏状态维度
        - num_layers: 卷积LSTM层数
        - batch_size: 批大小
        - epochs: 训练轮数
        - lr: 学习率
        - nonzero_weight: 非零样本的权重倍数
        """
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.nonzero_weight = nonzero_weight
        
        # 创建模型
        self.model = VehicleGridPredictor(
            input_dim=1,
            hidden_dim=hidden_dim,
            kernel_size=3,
            num_layers=num_layers
        )
        
        # 使用GPU（如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = self.weighted_mse_loss  # 使用自定义的加权MSE损失
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 训练历史记录
        self.history = []
    
    def weighted_mse_loss(self, y_pred, y_true):
        """
        加权MSE损失函数，对非零样本赋予更高权重
        
        参数:
        - y_pred: 预测值
        - y_true: 真实值
        
        返回:
        - 加权MSE损失
        """
        # 创建权重矩阵，非零样本权重更高
        weight_matrix = torch.ones_like(y_true)
        weight_matrix[y_true > 0] = self.nonzero_weight
        
        # 计算加权MSE
        return torch.mean(weight_matrix * torch.pow(y_pred - y_true, 2))
    
    def fit(self, train_data, val_data=None):
        """
        训练模型
        
        参数:
        - train_data: 训练数据，形状为(训练时间步数, 网格大小, 网格大小)
        - val_data: 可选的验证数据
        """
        # 创建数据集和数据加载器
        train_dataset = VehicleGridDataset(train_data, self.seq_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if val_data is not None:
            val_dataset = VehicleGridDataset(val_data, self.seq_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 记录训练开始
        print(f"开始训练神经网络模型，使用加权MSE损失(非零权重={self.nonzero_weight})")
        print(f"数据集大小: {len(train_dataset)}个样本")
        print(f"设备: {self.device}")
        
        # 统计训练数据中非零样本的比例
        nonzero_count = np.count_nonzero(train_data)
        total_count = train_data.size
        nonzero_percent = 100 * nonzero_count / total_count
        print(f"训练数据中非零样本比例: {nonzero_percent:.2f}% ({nonzero_count}/{total_count})")
        
        # 开始训练
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            val_loss = 0.0
            
            # 训练循环
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                self.optimizer.zero_grad()
                
                x_seq = batch['input_seq'].to(self.device)
                y_target = batch['target'].to(self.device)
                
                # 前向传播
                y_pred = self.model(x_seq)
                
                # 计算加权损失
                loss = self.criterion(y_pred, y_target)
                
                # 反向传播与优化
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 在验证集上评估
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        x_seq = batch['input_seq'].to(self.device)
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
    
    def predict(self, history_data):
        """
        预测下一个时间步的网格车辆数
        
        参数:
        - history_data: 历史数据，形状为(seq_length, 网格大小, 网格大小)
        
        返回:
        - 预测结果，形状为(网格大小, 网格大小)
        """
        self.model.eval()
        
        # 确保历史数据长度正确
        if len(history_data) < self.seq_length:
            raise ValueError(f"历史数据长度必须至少为{self.seq_length}")
        
        # 取最近的seq_length个时间步
        recent_data = history_data[-self.seq_length:]
        
        # 转换为张量并添加批维度
        x = torch.FloatTensor(recent_data).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            prediction = self.model(x)
        
        # 转换回numpy数组
        return prediction.cpu().numpy()[0]
    
    def evaluate(self, test_data, train_data):
        """
        评估模型
        
        参数:
        - test_data: 测试数据，形状为(测试时间步数, 网格大小, 网格大小)
        - train_data: 训练数据，形状为(训练时间步数, 网格大小, 网格大小)
        
        返回:
        - mse: 均方误差
        - mae: 平均绝对误差
        """
        self.model.eval()
        
        n_test_steps = len(test_data)
        predictions = []
        
        # 初始历史数据为训练数据的最后seq_length个时间步
        history_data = train_data[-self.seq_length:]
        
        # 对每个测试时间步进行预测
        for i in range(n_test_steps):
            # 预测下一个时间步
            pred = self.predict(history_data)
            predictions.append(pred)
            
            # 更新历史数据（使用实际的测试数据）
            if i < n_test_steps - 1:  # 除了最后一步，每次都需要更新历史
                history_data = np.vstack((history_data[1:], test_data[i:i+1]))
        
        # 将预测结果转换为数组
        predictions = np.array(predictions)
        
        # 计算误差
        mse = np.mean((predictions - test_data) ** 2)
        mae = np.mean(np.abs(predictions - test_data))
        
        return mse, mae
    
    def plot_loss(self):
        """绘制训练过程的损失曲线"""
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
    
    def save_model(self, path):
        """保存模型到指定路径"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """从指定路径加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"模型已从 {path} 加载")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载训练和测试数据
    train_data = np.load("train_data.npy")
    test_data = np.load("test_data.npy")
    
    # 创建并训练神经网络模型
    nn_model = NNGridModel(
        seq_length=5,
        hidden_dim=64,
        num_layers=2,
        batch_size=16,
        epochs=30,
        lr=0.001
    )
    
    # 训练模型
    nn_model.fit(train_data)
    
    # 绘制损失曲线
    nn_model.plot_loss()
    
    # 评估模型
    mse, mae = nn_model.evaluate(test_data, train_data)
    print(f"神经网络模型评估结果 - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # 保存模型
    nn_model.save_model("nn_grid_model.pth") 