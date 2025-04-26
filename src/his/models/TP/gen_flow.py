import torch
import torch.nn as nn
import numpy as np
import h5py
from models.TP.fastpredNF import fastpredNF_TP
from models.TP.dis_flow import FlowDiscriminator

class TrajectoryFlowGenerator(fastpredNF_TP):
    def __init__(self, cfg, args):
        """
        继承轨迹预测模型，并加入流量聚合模块
        """
        super(TrajectoryFlowGenerator, self).__init__(cfg, args)
        
        # 你可以在这里添加更多与流量生成相关的内容（例如，网格大小、流量计算方式等）
        self.grid_size = 32  # 设置网格大小，可以根据需求调整
        
        # self.max_lon = 116.646
        # self.min_lon = 116.10983
        # self.max_lat = 40.101093
        # self.min_lat = 39.710392
        
        self.max_lon = 116.50032165
        self.min_lon = 116.2503565
        self.max_lat = 39.9999216
        self.min_lat = 39.7997164
        
        # 初始化判别器
        self.discriminator = FlowDiscriminator(grid_size=self.grid_size)

        # 优化器
        self.g_optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 损失函数
        self.lambda_gp = 10  # 梯度惩罚系数
        self.lambda_adv = 0.1  # 对抗损失权重
        self.lambda_new_flow = 0.2  # 新生流量损失权重

        # 初始化新生流量预测器
        self.new_flow_predictor = NewFlowPredictor(grid_size=self.grid_size)
        
        # 加载模型权重
        # self.dual_modal_predictor.load_state_dict(torch.load("checkpoints/dual_flow/dual_model_final.pth"))
        
        # 加载模型
        # saved_model = torch.load("checkpoints/dual_flow/dual_model_final.pth")
        # if "state_dict" in saved_model:
        #     self.dual_modal_predictor.load_state_dict(saved_model["state_dict"])
        # else:
        #     self.dual_modal_predictor.load_state_dict(saved_model)
        
        # # 如果有历史轨迹数据，可以提前计算OD矩阵
        # self.od_matrix = None
        
        # # 历史流量数据 - 保持为numpy数组直到forward方法调用
        # self.historical_flow = None
        # self.historical_flow_tensor = None
        
        # self.load_historical_flow("mydata/his/BJ_MGF_17_19_100000_108.h5")
        
    def load_historical_flow(self, h5_path):
        """
        从h5文件加载历史流量数据
        
        参数:
            h5_path: h5文件路径
        """
        try:
            print(f"正在从 {h5_path} 加载历史流量数据")
            # 加载为numpy数组并存储
            self.historical_flow = self.dual_modal_predictor.load_historical_flow(h5_path)
            print(f"历史流量数据已加载，形状为: {self.historical_flow.shape}")
            
            # 清除之前可能存在的张量缓存
            self.historical_flow_tensor = None
            
            return self.historical_flow
        except Exception as e:
            print(f"加载历史流量数据失败: {e}")
            self.historical_flow = None
            self.historical_flow_tensor = None
            return None
        
    def to(self, device):
        """
        确保模型和所有子模块都移动到正确的设备上
        """
        super().to(device)
        self.new_flow_predictor = self.new_flow_predictor.to(device)
        
        # 仅在第一次调用时将历史流量数据转换为张量并移动到设备
        if self.historical_flow is not None and isinstance(self.historical_flow, np.ndarray):
            print("将历史流量数据转换为tensor并移动到设备")
            self.historical_flow_tensor = torch.from_numpy(self.historical_flow).float().to(device)
        return self

    def forward(self, pred_traj: torch.Tensor, real_traj: torch.Tensor, gt_ts_array: torch.Tensor) -> dict:
        """
        给定预测轨迹和真实轨迹，计算流量误差
        pred_traj: 预测的轨迹 (14432, 12, 2)
        real_traj: 真实轨迹 (14432, 12, 2)
        gt_ts_array: 真实轨迹的时间戳 (14432, 12)
        """
        # 计算真实流量
        real_flow = self.aggregate_flow(real_traj, gt_ts_array)
        
        # 1. 计算轨迹预测流量
        trajectory_flow = self.aggregate_flow(pred_traj, gt_ts_array)
        
        # 保存到npy文件
        # np.save("trajectory_flow_12_17.npy", trajectory_flow)
        
        
        # # 2. 使用历史流量数据
        # device = next(self.parameters()).device
        
        # # 如果历史流量数据已经转换为张量，则直接使用
        # if self.historical_flow_tensor is not None:
        #     historical_flow = self.historical_flow_tensor
        # # 如果历史流量数据为空，则使用轨迹流量
        # elif self.historical_flow is None:
        #     historical_flow = torch.from_numpy(trajectory_flow).float().to(device)
        # # 否则，将历史流量数据转换为张量
        # else:
        #     historical_flow = torch.from_numpy(self.historical_flow).float().to(device)
        #     # 缓存张量以供后续使用
        #     self.historical_flow_tensor = historical_flow
        
        # # 转换轨迹流量为张量
        # trajectory_flow = torch.from_numpy(trajectory_flow).float().to(device)
        
        # print(f"historical_flow.shape: {historical_flow.shape}")
        print(f"trajectory_flow.shape: {trajectory_flow.shape}")
        
        # # 调整维度顺序
        # historical_flow = historical_flow.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 2, time_steps, 32, 32]
        trajectory_flow = trajectory_flow.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 2, time_steps, 32, 32]
        
        print(f"historical_flow.shape: {historical_flow.shape}")
        print(f"trajectory_flow.shape: {trajectory_flow.shape}")
        
        # 使用双模态模型预测
        #total_flow = self.dual_modal_predictor(historical_flow, trajectory_flow)
        total_flow = trajectory_flow
        
        # 转换回原始格式 [time_steps, direction_channels, grid_size, grid_size]
        total_flow = total_flow.squeeze(0).permute(1, 0, 2, 3).cpu().detach().numpy()
        
        # 4. 计算流量误差（RMSE和MAE）
        outflow_rmse, outflow_mae = self.calculate_metrics(total_flow, real_flow, direction=0)
        inflow_rmse, inflow_mae = self.calculate_metrics(total_flow, real_flow, direction=1)
        
        # 计算真实流出量和预测流出量的均值
        real_flow_mean = np.mean(real_flow[:, 0])
        pred_flow_mean = np.mean(total_flow[:, 0])
        
        print(f"真实流出量: {real_flow_mean:.4f}, 预测流出量: {pred_flow_mean:.4f}")
        
        # 计算真实流入量和预测流入量的均值
        real_flow_mean = np.mean(real_flow[:, 1])
        pred_flow_mean = np.mean(total_flow[:, 1])
        
        print(f"真实流入量: {real_flow_mean:.4f}, 预测流入量: {pred_flow_mean:.4f}")
        
        # 5. 打印或返回误差
        return {
            "outflow_rmse": outflow_rmse,
            "outflow_mae": outflow_mae,
            "inflow_rmse": inflow_rmse,
            "inflow_mae": inflow_mae
        }
    
    def train_gan(self, pred_traj, real_traj, gt_ts_array, n_critic=5):
        """
        训练GAN模型，包括双模态流量预测
        """
        # 将NumPy数组转换为PyTorch张量
        device = next(self.parameters()).device
        
        # 计算轨迹流量
        trajectory_flow_np = self.aggregate_flow(pred_traj, gt_ts_array)
        real_flow_np = self.aggregate_flow(real_traj, gt_ts_array)
        
        # 使用历史流量数据
        # 如果历史流量数据已经转换为张量，则直接使用
        if self.historical_flow_tensor is not None:
            historical_flow = self.historical_flow_tensor
        # 如果历史流量数据为空，则使用轨迹流量
        elif self.historical_flow is None:
            historical_flow = torch.from_numpy(trajectory_flow_np).float().to(device)
        # 否则，将历史流量数据转换为张量
        else:
            historical_flow = torch.from_numpy(self.historical_flow).float().to(device)
            # 缓存张量以供后续使用
            self.historical_flow_tensor = historical_flow
        
        # 转换轨迹流量和真实流量为张量
        trajectory_flow = torch.from_numpy(trajectory_flow_np).float().to(device)
        real_flow = torch.from_numpy(real_flow_np).float().to(device)
        
        # 调整维度顺序 从 [time_steps, direction_channels, grid_size, grid_size] 转换为 [batch_size, direction_channels, time_steps, grid_size, grid_size]
        historical_flow = historical_flow.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 2, time_steps, 32, 32]
        trajectory_flow = trajectory_flow.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 2, time_steps, 32, 32]
        real_flow = real_flow.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 2, time_steps, 32, 32]
        
        # 使用双模态模型预测
        generated_flow = trajectory_flow
        
        # 训练判别器
        d_loss = 0
        for _ in range(n_critic):
            self.d_optimizer.zero_grad()
            
            real_validity = self.discriminator(real_flow)
            fake_validity = self.discriminator(generated_flow.detach())
            
            gradient_penalty = self.discriminator.compute_gradient_penalty(real_flow, generated_flow.detach())
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
            d_loss.backward()
            self.d_optimizer.step()
        
        # 训练生成器
        self.g_optimizer.zero_grad()
        
        # 生成器损失
        fake_validity = self.discriminator(generated_flow)
        g_loss = -torch.mean(fake_validity)
        
        # 轨迹预测损失
        traj_loss = self.trajectory_loss(pred_traj, real_traj)
        
        # 只考虑对抗损失和轨迹损失 (移除了 flow_loss)
        total_g_loss = (traj_loss + self.lambda_adv * g_loss)
        
        total_g_loss.backward()
        self.g_optimizer.step()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "traj_loss": traj_loss.item(),
            "total_g_loss": total_g_loss.item()
        }
    
    def trajectory_loss(self, pred_traj, real_traj):
        """
        计算轨迹预测损失
        
        参数:
            pred_traj: 预测的轨迹
            real_traj: 真实轨迹
        
        返回:
            轨迹预测损失
        """
        # 转换为张量
        device = next(self.parameters()).device
        if not isinstance(pred_traj, torch.Tensor):
            pred_traj = torch.from_numpy(pred_traj).float().to(device)
        if not isinstance(real_traj, torch.Tensor):
            real_traj = torch.from_numpy(real_traj).float().to(device)
        
        # 计算ADE (Average Displacement Error)
        ade = torch.mean(torch.sqrt(torch.sum((pred_traj - real_traj) ** 2, dim=-1)))
        
        # 计算FDE (Final Displacement Error)
        fde = torch.mean(torch.sqrt(torch.sum((pred_traj[:, -1] - real_traj[:, -1]) ** 2, dim=-1)))
        
        # 总损失
        loss = ade + fde
        return loss

    def aggregate_flow(self, traj: torch.Tensor, gt_ts_array: torch.Tensor) -> np.ndarray:
        """
        基于轨迹计算流量（通过网格聚合）
        只计算7号和8号的历史流量
        traj: 输入的轨迹，形状为 (B, T, 2)，B是batch size，T是轨迹长度，2是(x, y)坐标
        gt_ts_array: 时间戳数组，形状为 (B, T)
        """
        # time_start = 828
        # time_end = 1019
        # time_count = time_end - time_start + 1
        
        time_start = 1128
        time_end = 1139
        time_count = time_end - time_start + 1
        
        # 初始化流量矩阵 - 时间步, 方向(流出/流入), 网格行, 网格列
        flow = np.zeros((time_count, 2, self.grid_size, self.grid_size), dtype=np.float32)
        
        # 处理每个样本的轨迹
        for b in range(traj.shape[0]):
            sample_traj = traj[b]  # 获取单个样本的轨迹
            sample_gt_ts_array = gt_ts_array[b]
            
            # 如果第一个点的时间步不是1128，则跳过
            if not (time_start - 11 <= sample_gt_ts_array[0] <= time_end):
                continue
            
            # 从第二个点开始遍历
            for t in range(1, sample_traj.shape[0]):
                sample_gt_ts = sample_gt_ts_array[t]
                
                # 只处理7号和8号的数据（时间戳从8640到11520）
                if not (time_start <= sample_gt_ts <= time_end):
                    continue
                
                # 获取前一个点和当前点的坐标
                prev_x, prev_y = sample_traj[t-1, 0], sample_traj[t-1, 1]
                curr_x, curr_y = sample_traj[t, 0], sample_traj[t, 1]
                
                # 计算时间步 (从8640开始)
                time_step = int(sample_gt_ts) - time_start
                
                # 确保时间步在有效范围内
                if not (0 <= time_step < time_count):
                    continue
                
                # 映射到网格坐标
                prev_row, prev_col = self.map_to_grid(prev_x, prev_y)
                curr_row, curr_col = self.map_to_grid(curr_x, curr_y)
                
                # 只有当区域发生变化时才更新流量
                if (prev_row != curr_row) or (prev_col != curr_col):
                    flow[time_step, 0, prev_row, prev_col] += 1  # 流出量
                    flow[time_step, 1, curr_row, curr_col] += 1  # 流入量
        
        return flow
    
    def calculate_metrics(self, pred: np.ndarray, truth: np.ndarray, direction: int):
        """计算RMSE和MAE，direction=0表示流出量，direction=1表示流入量"""
        pred_flat = pred[:, direction].flatten()
        truth_flat = truth[:, direction].flatten()
        
        mse = np.mean((pred_flat - truth_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_flat - truth_flat))
        return rmse, mae

    def map_to_grid(self, lon, lat, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
        """将经纬度映射到32x32网格坐标"""
        # 如果没有提供边界值，使用类的属性
        if min_lon is None:
            min_lon = self.min_lon
        if max_lon is None:
            max_lon = self.max_lon
        if min_lat is None:
            min_lat = self.min_lat
        if max_lat is None:
            max_lat = self.max_lat
        
        grid_size = self.grid_size
        
        lat_step = (max_lat - min_lat) / grid_size
        lon_step = (max_lon - min_lon) / grid_size
        row = int((lat - min_lat) / lat_step)
        col = int((lon - min_lon) / lon_step)
        row = max(0, min(row, grid_size-1))
        col = max(0, min(col, grid_size-1))
        return row, col

    def compute_od_matrix_from_trajectories(self, trajectories):
        """
        从轨迹数据计算OD矩阵，并设置给新生流量预测器
        
        参数:
            trajectories: 轨迹数据列表，每个轨迹是一个(T, 2)的坐标序列
        """
        print("开始计算OD矩阵...")
        od_counts = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.grid_size))
        
        for traj in trajectories:
            if len(traj) < 2:
                continue
                
            # 获取起点和终点坐标
            start_point = traj[0]
            end_point = traj[-1]
            
            # 将坐标映射到网格
            start_row, start_col = self.map_to_grid(start_point[0], start_point[1])
            end_row, end_col = self.map_to_grid(end_point[0], end_point[1])
            
            # 更新OD计数
            od_counts[start_row, start_col, end_row, end_col] += 1
            
        # 将计数转换为概率分布
        total_count = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                total = np.sum(od_counts[i, j])
                if total > 0:
                    od_counts[i, j] = od_counts[i, j] / total
                    total_count += 1
                else:
                    # 如果该网格没有出发的车辆，就设置为留在原地
                    od_counts[i, j, i, j] = 1.0
        
        print(f"OD矩阵计算完成，共有{total_count}个网格有出发车辆")
        
        # 保存OD矩阵
        self.od_matrix = od_counts
        
        # 设置给新生流量预测器
        self.new_flow_predictor.set_od_matrix(od_counts)
        
        return od_counts

class NewFlowPredictor(nn.Module):
    def __init__(self, grid_size=32):
        super(NewFlowPredictor, self).__init__()
        self.grid_size = grid_size
        
        # 基础参数
        self.mu0 = nn.Parameter(torch.ones(grid_size, grid_size) * 0.1)  # 基础流量强度
        self.alpha = nn.Parameter(torch.ones(grid_size, grid_size) * 0.5)  # 自激强度
        self.beta = nn.Parameter(torch.ones(grid_size, grid_size) * 0.1)  # 衰减速率
        
        # 存储历史事件
        self.event_history = {}
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # 初始化OD矩阵 - 从网格(i,j)到网格(k,l)的转移概率
        self.od_matrix = self._initialize_od_matrix()
        
        self.max_lon = 116.50032165
        self.min_lon = 116.2503565
        self.max_lat = 39.9999216
        self.min_lat = 39.7997164
        
        self.a_k = nn.Parameter(torch.ones(2) * 0.1)
        self.b_k = nn.Parameter(torch.ones(2) * 0.1)
        
    def _initialize_od_matrix(self):
        """
        初始化OD矩阵 - 默认使用简单的假设：
        所有车辆只在出发网格内移动（对角矩阵）
        """
        # 安全地获取设备，避免依赖self.mu0
        device = torch.device('cpu')
        
        od_matrix = torch.zeros(self.grid_size, self.grid_size, 
                               self.grid_size, self.grid_size, device=device)
        
        # 设置对角元素为1.0 (所有车辆保持在原网格)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                od_matrix[i, j, i, j] = 1.0
                
        return od_matrix
        
    def set_od_matrix(self, new_od_matrix):
        """
        设置新的OD矩阵
        
        参数:
            new_od_matrix: 形状为 [grid_size, grid_size, grid_size, grid_size] 的张量或numpy数组
                          表示从网格(i,j)到网格(k,l)的转移概率
        """
        # 首先获取模型当前设备
        device = next(self.parameters()).device if hasattr(self, 'mu0') else torch.device('cpu')
        
        # 将numpy数组转换为torch张量
        if not isinstance(new_od_matrix, torch.Tensor):
            new_od_matrix = torch.tensor(new_od_matrix, dtype=torch.float32)
        
        # 将OD矩阵移动到正确的设备上
        self.od_matrix = new_od_matrix.to(device)
        
    def _compute_background_intensity(self, t):
        """
        计算背景强度μ(t)，使用傅里叶级数建模周期性
        t: 时间步（分钟）
        """
        # 确保t是PyTorch张量
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=self.mu0.device)
        
        total_window_minutes = 120  
        t_normalized = 2 * torch.pi * t / total_window_minutes  # 按预测窗口周期归一化
        
        # 调整K值以匹配小时级周期
        self.K = 2  # 使用基频（120分钟）和二次谐波（60分钟）
        background = self.mu0.clone()
        for k in range(1, self.K+1):  # k=1,2
            background += (self.a_k[k-1] * torch.sin(k * t_normalized) + 
                        self.b_k[k-1] * torch.cos(k * t_normalized))
        return background
        

    
    def _compute_excitation(self, t, row_idx, col_idx):
        """
        计算自激项R_t
        """
        # 确保t是PyTorch张量
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=self.mu0.device)
            
        if (row_idx, col_idx) not in self.event_history:
            return torch.zeros(1, device=self.mu0.device)
            
        events = self.event_history[(row_idx, col_idx)]
        excitation = torch.zeros(1, device=self.mu0.device)
        
        for event_time in events:
            if event_time < t:
                excitation += self.alpha[row_idx, col_idx] * \
                            torch.exp(-self.beta[row_idx, col_idx] * (t - event_time))
        
        return excitation
    
    def _hawkes_intensity(self, t, row_idx, col_idx):
        """
        计算完整的Hawkes过程强度函数
        """
        # 计算背景强度
        background = self._compute_background_intensity(t)
        
        # 计算自激项
        excitation = self._compute_excitation(t, row_idx, col_idx)
        
        return background[row_idx, col_idx] + excitation
    
    def _log_likelihood(self, events, row_idx, col_idx):
        """
        计算特定网格的Hawkes过程对数似然
        """
        if len(events) < 2:
            return torch.tensor(0.0, device=self.mu0.device)
            
        T = events[-1] - events[0]
        n = len(events)
        
        # 计算基础强度项
        base_term = -self.mu0[row_idx, col_idx] * T
        
        # 计算自激效应项
        excitation_term = 0
        for i in range(n):
            for j in range(i):
                excitation_term += torch.log(
                    self._hawkes_intensity(events[i], row_idx, col_idx)
                )
                
        return base_term + excitation_term
    
    def fit(self, df):
        """
        拟合Hawkes过程参数
        df: 包含sample_id, timestep, x, y等信息的DataFrame
        """
        # 1. 识别新生车辆
        new_vehicles = df[df['timestep'] == 0]
        
        # 2. 按网格统计新生车辆并存储事件历史
        for _, row in new_vehicles.iterrows():
            row_idx, col_idx = self.map_to_grid(row['x'], row['y'])
            if (row_idx, col_idx) not in self.event_history:
                self.event_history[(row_idx, col_idx)] = []
            self.event_history[(row_idx, col_idx)].append(row['real_timestamp'])
            
        # 3. 优化参数
        self._optimize_parameters()
        
        # 4. 如果有整个轨迹的数据，可以计算OD矩阵
        self._compute_od_matrix(df)
        
    def _compute_od_matrix(self, df):
        """
        从轨迹数据计算OD矩阵
        df: 包含轨迹信息的DataFrame
        """
        device = self.mu0.device
        od_counts = torch.zeros(self.grid_size, self.grid_size, 
                                self.grid_size, self.grid_size, 
                                device=device)
        
        # 按车辆ID分组，为每辆车找到其起点和终点
        for vehicle_id, vehicle_data in df.groupby('sample_id'):
            if len(vehicle_data) < 2:
                continue
                
            # 获取起点和终点坐标
            start_point = vehicle_data.iloc[0]
            end_point = vehicle_data.iloc[-1]
            
            # 将坐标映射到网格
            start_row, start_col = self.map_to_grid(start_point['x'], start_point['y'])
            end_row, end_col = self.map_to_grid(end_point['x'], end_point['y'])
            
            # 更新OD计数
            od_counts[start_row, start_col, end_row, end_col] += 1
            
        # 将计数转换为概率分布
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                total = od_counts[i, j].sum()
                if total > 0:
                    od_counts[i, j] = od_counts[i, j] / total
                else:
                    # 如果该网格没有出发的车辆，就设置为留在原地
                    od_counts[i, j, i, j] = 1.0
                    
        self.od_matrix = od_counts
        
    def _optimize_parameters(self):
        """
        优化Hawkes过程参数
        """
        for _ in range(100):  # 迭代次数
            self.optimizer.zero_grad()
            loss = self._compute_total_loss()
            loss.backward()
            self.optimizer.step()
            
            # 确保参数为正
            self.mu0.data.clamp_(min=0.001)
            self.alpha.data.clamp_(min=0.001)
            self.beta.data.clamp_(min=0.001)
            
    def _compute_total_loss(self):
        """
        计算所有网格的总损失
        """
        total_loss = 0
        for (row_idx, col_idx), events in self.event_history.items():
            if len(events) >= 2:
                events_tensor = torch.tensor(events, dtype=torch.float32).to(self.mu0.device)
                total_loss -= self._log_likelihood(events_tensor, row_idx, col_idx)
        return total_loss
        
    def predict(self, t_array):
        """
        预测整个时间序列的新生流量
        返回同时包含流出量和流入量的张量
        """
        device = next(self.parameters()).device
        
        # 确保t_array是PyTorch张量
        if not isinstance(t_array, torch.Tensor):
            t_array = torch.tensor(t_array, dtype=torch.float32, device=device)
        
        # 首先计算各网格的流出量 (Hawkes模型直接预测)
        outflow = self.predict_outflow(t_array)
        
        # 然后基于OD矩阵计算流入量
        inflow = self.predict_inflow(outflow)
        
        # 合并两类流量
        new_flow = torch.zeros((len(t_array), 2, self.grid_size, self.grid_size), device=device)
        new_flow[:, 0, :, :] = outflow  # 流出量
        new_flow[:, 1, :, :] = inflow   # 流入量
        
        return new_flow
    
    def predict_outflow(self, t_array):
        """
        预测各网格在各时间步的流出量 (Hawkes强度)
        返回形状为 [T, grid_size, grid_size] 的张量
        """
        device = next(self.parameters()).device
        
        # 确保t_array是PyTorch张量
        if not isinstance(t_array, torch.Tensor):
            t_array = torch.tensor(t_array, dtype=torch.float32, device=device)
            
        # 初始化流出量张量
        outflow = torch.zeros((len(t_array), self.grid_size, self.grid_size), device=device)
        
        # 计算每个时间步、每个网格的Hawkes强度作为流出量
        for t_idx, t in enumerate(t_array):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    outflow[t_idx, i, j] = self._hawkes_intensity(t, i, j)
        
        return outflow
        
    def predict_inflow(self, outflow):
        """
        基于流出量和OD矩阵计算流入量
        
        参数:
            outflow: 形状为 [T, grid_size, grid_size] 的张量，表示各网格在各时间步的流出量
            
        返回:
            形状为 [T, grid_size, grid_size] 的张量，表示各网格在各时间步的流入量
        """
        # 检查输入
        if len(outflow.shape) != 3:
            raise ValueError(f"outflow应为3维张量 [T, grid_size, grid_size]，但得到 {outflow.shape}")
            
        T = outflow.shape[0]  # 时间步数
        device = outflow.device
        
        # 确保OD矩阵在正确的设备上
        if self.od_matrix.device != device:
            self.od_matrix = self.od_matrix.to(device)
        
        # 使用einsum进行批量计算，一次性计算所有时间步
        # inflow[t, k, l] = sum_{i,j} outflow[t, i, j] * od_matrix[i, j, k, l]
        inflow = torch.einsum('tij,ijkl->tkl', outflow, self.od_matrix)
            
        return inflow
        
    def update(self, pred_flow, real_flow):
        """
        根据预测误差更新参数
        """
        # 确保输入是带有梯度的张量
        if not isinstance(pred_flow, torch.Tensor):
            pred_flow = torch.from_numpy(pred_flow).float()
        if not isinstance(real_flow, torch.Tensor):
            real_flow = torch.from_numpy(real_flow).float()
            
        # 移动到正确的设备
        device = next(self.parameters()).device
        pred_flow = pred_flow.to(device)
        real_flow = real_flow.to(device)
        
        # 计算损失
        loss = torch.mean((pred_flow - real_flow) ** 2)
        
        # 反向传播和更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def get_new_flow(self, t_array):
        """
        获取整个时间序列的新生流量预测
        """
        return self.predict(t_array)

    def map_to_grid(self, lon, lat, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
        """将经纬度映射到网格坐标"""
        if min_lon is None:
            min_lon = self.min_lon
        if max_lon is None:
            max_lon = self.max_lon
        if min_lat is None:
            min_lat = self.min_lat
        if max_lat is None:
            max_lat = self.max_lat
            
        lat_step = (max_lat - min_lat) / self.grid_size
        lon_step = (max_lon - min_lon) / self.grid_size
        row = int((lat - min_lat) / lat_step)
        col = int((lon - min_lon) / lon_step)
        row = max(0, min(row, self.grid_size-1))
        col = max(0, min(col, self.grid_size-1))
        return row, col
