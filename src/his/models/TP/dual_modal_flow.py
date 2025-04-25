import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeFeatureEncoder(nn.Module):
    """时间特征编码器"""
    def __init__(self, input_channels, hidden_channels=64, output_channels=128):
        super(TimeFeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class TrajectoryEncoder(nn.Module):
    """轨迹数据编码器"""
    def __init__(self, input_channels, hidden_channels=64, output_channels=128):
        super(TrajectoryEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class GateModule(nn.Module):
    """门控融合模块"""
    def __init__(self, input_channels, output_channels):
        super(GateModule, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        # 拼接两个特征
        combined = torch.cat([x1, x2], dim=1)
        # 计算门控权重
        gate = self.gate_net(combined)
        # 加权融合
        fused = gate * x1 + (1 - gate) * x2
        return fused

class DualModalFlowPredictor(nn.Module):
    def __init__(self, grid_size=32, direction_channels=2, 
                 closeness_steps=3, period_steps=3, trend_steps=3,
                 hidden_channels=64, output_channels=128):
        """
        双模态流量预测模型，结合历史流量和轨迹预测流量
        
        参数:
            grid_size: 网格大小，默认为32x32
            direction_channels: 方向通道数，默认为2（流入/流出）
            closeness_steps: 短期时间步数（每个时间步有direction_channels个通道）
            period_steps: 周期（日）时间步数
            trend_steps: 趋势（周）时间步数
            hidden_channels: 隐藏层通道数
            output_channels: 输出层通道数
        """
        super(DualModalFlowPredictor, self).__init__()
        
        # 基础参数
        self.grid_size = grid_size
        self.direction_channels = direction_channels
        self.closeness_steps = closeness_steps
        self.period_steps = period_steps
        self.trend_steps = trend_steps
        
        # 短期时间特征编码器
        if closeness_steps > 0:
            self.closeness_encoder = TimeFeatureEncoder(
                closeness_steps * direction_channels, 
                hidden_channels, 
                output_channels
            )
        
        # 周期（日）时间特征编码器
        if period_steps > 0:
            self.period_encoder = TimeFeatureEncoder(
                period_steps * direction_channels, 
                hidden_channels, 
                output_channels
            )
        
        # 趋势（周）时间特征编码器
        if trend_steps > 0:
            self.trend_encoder = TimeFeatureEncoder(
                trend_steps * direction_channels, 
                hidden_channels, 
                output_channels
            )
        
        # 轨迹数据编码器
        self.trajectory_encoder = TrajectoryEncoder(
            direction_channels, 
            hidden_channels, 
            output_channels
        )
        
        # 历史数据特征融合
        historical_encoders_count = sum([closeness_steps > 0, period_steps > 0, trend_steps > 0])
        if historical_encoders_count > 1:
            if historical_encoders_count == 2:
                self.historical_fusion = GateModule(output_channels * 2, output_channels)
            else:  # 3个编码器
                self.historical_fusion1 = GateModule(output_channels * 2, output_channels)
                self.historical_fusion2 = GateModule(output_channels * 2, output_channels)
        
        # 双模态融合
        self.modal_fusion = GateModule(output_channels * 2, output_channels)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv2d(output_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, direction_channels, kernel_size=3, padding=1)
        )
        
        # 优化器
        self.optimizer = None
    
    def set_optimizer(self, lr=0.001, weight_decay=1e-5):
        """设置优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
    
    def forward(self, historical_inputs, trajectory_input):
        """
        前向传播
        
        参数:
            historical_inputs: 历史时间特征列表 [closeness, period, trend]
            trajectory_input: 轨迹预测数据 [batch_size, direction_channels, grid_size, grid_size]
            
        返回:
            预测的流量数据 [batch_size, direction_channels, grid_size, grid_size]
        """
        # 处理历史时间特征
        historical_features = []
        
        # 编码短期时间特征
        if self.closeness_steps > 0:
            closeness_feature = self.closeness_encoder(historical_inputs[0])
            historical_features.append(closeness_feature)
        
        # 编码周期（日）时间特征
        if self.period_steps > 0:
            period_idx = 1 if self.closeness_steps > 0 else 0
            if period_idx < len(historical_inputs):
                period_feature = self.period_encoder(historical_inputs[period_idx])
                historical_features.append(period_feature)
        
        # 编码趋势（周）时间特征
        if self.trend_steps > 0:
            trend_idx = (1 if self.closeness_steps > 0 else 0) + (1 if self.period_steps > 0 else 0)
            if trend_idx < len(historical_inputs):
                trend_feature = self.trend_encoder(historical_inputs[trend_idx])
                historical_features.append(trend_feature)
        
        # 融合历史时间特征
        if len(historical_features) == 1:
            historical_fused = historical_features[0]
        elif len(historical_features) == 2:
            historical_fused = self.historical_fusion(historical_features[0], historical_features[1])
        elif len(historical_features) == 3:
            temp_fused = self.historical_fusion1(historical_features[0], historical_features[1])
            historical_fused = self.historical_fusion2(temp_fused, historical_features[2])
        else:
            raise ValueError("至少需要一种历史时间特征")
        
        # 编码轨迹数据
        trajectory_feature = self.trajectory_encoder(trajectory_input)
        
        # 融合历史和轨迹特征
        fused_feature = self.modal_fusion(historical_fused, trajectory_feature)
        
        # 生成预测
        output = self.output_layer(fused_feature)
        
        return output
    
    def train_batch(self, historical_inputs, trajectory_input, target):
        """
        训练单批次数据
        
        参数:
            historical_inputs: 历史时间特征列表
            trajectory_input: 轨迹预测数据
            target: 目标流量数据
            
        返回:
            损失值
        """
        self.train()
        
        # 确保优化器已设置
        if self.optimizer is None:
            self.set_optimizer()
        
        # 前向传播
        pred = self(historical_inputs, trajectory_input)
        
        # 计算损失
        loss = F.mse_loss(pred, target)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data_loader):
        """
        评估模型
        
        参数:
            data_loader: 数据加载器
            
        返回:
            平均损失和评估指标
        """
        self.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for historical_inputs, trajectory_input, target in data_loader:
                # 确保数据在正确的设备上
                device = next(self.parameters()).device
                historical_inputs = [x.to(device) for x in historical_inputs]
                trajectory_input = trajectory_input.to(device)
                target = target.to(device)
                
                # 前向传播
                pred = self(historical_inputs, trajectory_input)
                
                # 计算损失
                loss = F.mse_loss(pred, target, reduction='sum')
                total_loss += loss.item()
                total_samples += target.size(0)
        
        return total_loss / total_samples
    
    def predict(self, historical_inputs, trajectory_input):
        """
        生成预测
        
        参数:
            historical_inputs: 历史时间特征列表
            trajectory_input: 轨迹预测数据
            
        返回:
            预测的流量数据
        """
        self.eval()
        with torch.no_grad():
            return self(historical_inputs, trajectory_input) 