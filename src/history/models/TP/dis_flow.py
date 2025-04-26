import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlowDiscriminator(nn.Module):
    def __init__(self, grid_size=32, time_steps=20, direction_channels=2):
        """
        流量判别器模型，用于区分真实流量和生成的流量
        
        参数:
            grid_size: 网格大小，默认为32x32
            time_steps: 时间步数，默认为2880
            direction_channels: 方向通道数，默认为2（流入/流出）
        """
        super(FlowDiscriminator, self).__init__()
        
        # 对长时间序列进行降采样
        self.temporal_sample_rate = 1  # 每10步取1步
        self.effective_time_steps = time_steps // self.temporal_sample_rate
        
        # 输入形状: [batch_size, direction_channels, time_steps, grid_size, grid_size]
        
        # 调整 3D 卷积参数以适应较短时间序列
        self.conv3d_1 = nn.Conv3d(
            direction_channels, 32, 
            kernel_size=(3, 3, 3),  # 减小时间维的 kernel
            stride=(2, 1, 1),       # 调整 stride 避免过度降维
            padding=(1, 1, 1)       # 适当 padding
        )
        self.bn3d_1 = nn.BatchNorm3d(32)
        
        self.conv3d_2 = nn.Conv3d(
            32, 64, 
            kernel_size=(3, 3, 3), 
            stride=(2, 2, 2),       # 调整 stride
            padding=(1, 1, 1)
        )
        self.bn3d_2 = nn.BatchNorm3d(64)
        
        # 池化层保持不变
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # 手动计算处理后时间维度（经测试后设为 2）
        self.reduced_time = 2
        
        # 调整 time_collapse 层
        self.time_collapse = nn.Conv3d(
            64, 128, 
            kernel_size=(self.reduced_time, 1, 1),  # 确保与处理后时间维度匹配
            stride=(1, 1, 1)
        )
        
        # # 3D卷积层用于时空特征提取，加大时间步长以快速减少时间维度
        # self.conv3d_1 = nn.Conv3d(direction_channels, 32, 
        #                           kernel_size=(5, 3, 3), 
        #                           stride=(3, 1, 1), 
        #                           padding=(2, 1, 1))
        # self.bn3d_1 = nn.BatchNorm3d(32)
        
        # self.conv3d_2 = nn.Conv3d(32, 64, 
        #                           kernel_size=(5, 3, 3), 
        #                           stride=(3, 2, 2), 
        #                           padding=(2, 1, 1))
        # self.bn3d_2 = nn.BatchNorm3d(64)
        
        # # 3D池化进一步减少时间维度
        # self.pool3d = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # # 计算3D卷积后的时间维度: T//10//3//3//2 ≈ T//180
        # self.reduced_time = max(1, self.effective_time_steps // 18)
        
        # # 使用3D到2D的转换卷积，合并时间维度到通道
        # self.time_collapse = nn.Conv3d(64, 128, 
        #                                kernel_size=(self.reduced_time, 1, 1),
        #                                stride=(1, 1, 1))
        
        # 空间卷积层 - 输入通道固定为128
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        
        # 计算全连接层的输入大小
        fc_size = 512 * (grid_size // (2*2)) * (grid_size // (2*2))  # 考虑conv3d_2和conv2的stride
        
        # 全连接层
        self.fc1 = nn.Linear(fc_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        
        # Dropout层
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入流量数据，形状为 [batch_size, time_steps, direction_channels, grid_size, grid_size]
        
        返回:
            未经sigmoid激活的critic值（对于WGAN-GP）
        """
        batch_size, time_steps, direction_channels, grid_h, grid_w = x.shape
        
        # 时间降采样
        indices = torch.linspace(0, time_steps-1, self.effective_time_steps).long()
        x = x[:, indices]
        
        # 重塑输入以进行3D卷积
        x = x.permute(0, 2, 1, 3, 4)  # [batch, direction, time, grid_h, grid_w]
        
        # 3D卷积提取时空特征
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = self.pool3d(x)
        
        # 使用3D卷积合并时间维度
        x = F.relu(self.time_collapse(x))  # 输出 [batch, 128, 1, H/2, W/2]
        
        # 去除时间维度（已经合并到通道）
        x = x.squeeze(2)  # [batch, 128, H/2, W/2]
        
        # 空间卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # 直接返回critic值，不使用sigmoid
        return x
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        计算梯度惩罚项，用于WGAN-GP训练
        
        参数:
            real_samples: 真实流量样本
            fake_samples: 生成的流量样本
            
        返回:
            梯度惩罚值
        """
        # 随机权重项
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(real_samples.device)
        alpha = alpha.expand_as(real_samples)
        
        # 插值样本
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates = interpolates.requires_grad_(True)
        
        # 计算判别器对插值样本的输出
        d_interpolates = self.forward(interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 修改这一行，使用.reshape()替代.view()
        gradients = gradients.reshape(gradients.size(0), -1)
        
        # 或者确保张量连续后再使用.view()
        # gradients = gradients.contiguous().view(gradients.size(0), -1)
        
        # 计算梯度惩罚
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

class TransformerFlowDiscriminator(nn.Module):
    def __init__(self, grid_size=32, time_steps=2880, direction_channels=2, 
                 d_model=128, nhead=8, num_layers=3, dim_feedforward=512,
                 temporal_sample_rate=60):
        """
        基于Transformer的流量判别器，用于捕捉时序依赖关系
        
        参数:
            grid_size: 网格大小，默认为32x32
            time_steps: 时间步数，默认为2880
            direction_channels: 方向通道数，默认为2（流入/流出）
            d_model: Transformer模型维度
            nhead: 多头注意力的头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            temporal_sample_rate: 时间降采样率，每隔多少步取一个样本
        """
        super(TransformerFlowDiscriminator, self).__init__()
        
        # 时间降采样设置
        self.temporal_sample_rate = temporal_sample_rate
        self.effective_time_steps = time_steps // temporal_sample_rate
        
        # 空间编码：将2D网格压缩为特征向量
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(direction_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # 计算空间编码后的特征大小
        spatial_feature_size = 128 * (grid_size // 8) * (grid_size // 8)
        
        # 投影到Transformer维度
        self.projection = nn.Linear(spatial_feature_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.effective_time_steps)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dim_feedforward=dim_feedforward,
                                                  batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 输出头
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入流量数据，形状为 [batch_size, time_steps, direction_channels, grid_size, grid_size]
        
        返回:
            未经sigmoid激活的critic值（对于WGAN-GP）
        """
        batch_size, time_steps, direction_channels, grid_h, grid_w = x.shape
        
        # 时间降采样
        indices = torch.linspace(0, time_steps-1, self.effective_time_steps).long()
        x = x[:, indices]
        
        # 向量化处理空间特征
        # 重塑为 [B*T', C, H, W]
        x = x.view(-1, direction_channels, grid_h, grid_w)
        
        # 批量应用空间编码器
        spatial_features = self.spatial_encoder(x)  # [B*T', 128, H/8, W/8]
        spatial_features = spatial_features.view(spatial_features.size(0), -1)  # [B*T', 128*(H/8)*(W/8)]
        
        # 批量投影到Transformer维度
        projected = self.projection(spatial_features)  # [B*T', d_model]
        
        # 重塑回 [B, T', d_model]
        sequence = projected.view(batch_size, self.effective_time_steps, -1)
        
        # 添加位置编码
        sequence = self.pos_encoder(sequence)
        
        # 应用Transformer编码器
        sequence = self.transformer_encoder(sequence)
        
        # 聚合时间维度信息 (使用平均池化+最后一步的组合)
        avg_pooled = torch.mean(sequence, dim=1)  # [B, d_model]
        last_step = sequence[:, -1]  # [B, d_model]
        combined = (avg_pooled + last_step) / 2  # 结合全局和最终时间步信息
        
        # 通过输出头获得最终结果
        output = self.fc(combined)  # [B, 1]
        
        return output
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        计算梯度惩罚项，用于WGAN-GP训练
        
        参数:
            real_samples: 真实流量样本
            fake_samples: 生成的流量样本
            
        返回:
            梯度惩罚值
        """
        # 随机权重项
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(real_samples.device)
        alpha = alpha.expand_as(real_samples)
        
        # 插值样本
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates = interpolates.requires_grad_(True)
        
        # 计算判别器对插值样本的输出
        d_interpolates = self.forward(interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.reshape(gradients.size(0), -1)
        
        # 计算梯度惩罚
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

class PositionalEncoding(nn.Module):
    """
    位置编码模块，为序列中的每个位置提供一个唯一的位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入中
        
        参数:
            x: 输入张量 [batch, seq_len, d_model]
            
        返回:
            添加了位置编码的张量
        """
        return x + self.pe[:, :x.size(1), :]
