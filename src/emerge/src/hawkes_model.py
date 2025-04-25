import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize

class HawkesGridModel:
    """
    基于Hawkes过程的网格车辆预测模型
    对每个网格单独训练一个Hawkes过程模型
    """
    def __init__(self, global_alpha=0.8, global_beta=1.0, global_mu=0.2, space_gamma=0.3, space_delta=1.0):
        """
        初始化模型
        
        参数:
        - global_alpha: 默认的alpha参数（历史事件的影响强度）
        - global_beta: 默认的beta参数（历史事件影响的衰减率）
        - global_mu: 默认的mu参数（基线强度）
        - space_gamma: 空间邻域影响强度
        - space_delta: 空间距离衰减率
        """
        self.grid_params = {}  # 存储每个网格的参数
        self.global_alpha = global_alpha
        self.global_beta = global_beta
        self.global_mu = global_mu
        self.space_gamma = space_gamma
        self.space_delta = space_delta
        self.grid_size = None
        self.grid_adjacency = None  # 存储网格邻接关系
    
    def _create_grid_adjacency(self, grid_size):
        """
        创建网格邻接关系矩阵
        
        参数:
        - grid_size: 网格边长
        
        返回:
        - 邻接关系列表，每个网格的相邻网格索引
        """
        adjacency = {}
        for i in range(grid_size):
            for j in range(grid_size):
                grid_idx = i * grid_size + j
                neighbors = []
                
                # 添加相邻网格（上下左右和对角线）
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # 跳过自身
                        
                        ni, nj = i + di, j + dj
                        # 检查边界
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            neighbor_idx = ni * grid_size + nj
                            dist = np.sqrt(di**2 + dj**2)  # 计算距离
                            neighbors.append((neighbor_idx, dist))
                
                adjacency[grid_idx] = neighbors
        
        return adjacency
    
    def intensity(self, t, grid_idx, history_times, history_grids=None):
        """
        计算特定时间和网格的强度函数值
        
        参数:
        - t: 时间点
        - grid_idx: 网格索引
        - history_times: 历史事件时间列表
        - history_grids: 历史事件对应的网格索引列表
        
        返回:
        - 强度函数值
        """
        # 获取网格的参数
        if grid_idx in self.grid_params:
            alpha, beta, mu = self.grid_params[grid_idx]
        else:
            alpha, beta, mu = self.global_alpha, self.global_beta, self.global_mu
        
        # 计算自身历史事件的影响
        intensity = mu
        
        for i, prev_t in enumerate(history_times):
            # 只考虑当前网格的历史事件
            if history_grids is None or history_grids[i] == grid_idx:
                time_diff = t - prev_t
                if time_diff > 0:  # 只考虑过去的事件
                    intensity += alpha * np.exp(-beta * time_diff)
        
        # 添加空间邻域影响
        if history_grids is not None and self.grid_adjacency and grid_idx in self.grid_adjacency:
            for i, prev_t in enumerate(history_times):
                prev_grid = history_grids[i]
                if prev_grid != grid_idx:  # 不考虑自身网格（已在上面计算）
                    # 检查该网格是否是邻居
                    for neighbor_idx, dist in self.grid_adjacency[grid_idx]:
                        if prev_grid == neighbor_idx:
                            time_diff = t - prev_t
                            if time_diff > 0:  # 只考虑过去的事件
                                # 基于空间距离的衰减影响
                                space_influence = self.space_gamma * np.exp(-self.space_delta * dist - beta * time_diff)
                                intensity += space_influence
                            break
        
        return intensity
    
    def log_likelihood(self, params, event_times, T):
        """
        计算对数似然函数
        
        参数:
        - params: (alpha, beta, mu)参数
        - event_times: 事件发生时间
        - T: 观察结束时间
        
        返回:
        - 负对数似然（用于最小化）
        """
        alpha, beta, mu = params
        
        # 参数约束
        if alpha < 0 or beta <= 0 or mu <= 0:
            return np.inf
        
        # 计算对数似然的第一部分：每个事件的强度贡献
        log_intensity_sum = 0
        for i, t in enumerate(event_times):
            history = event_times[:i]
            intensity_t = self.intensity(t, history, params)
            log_intensity_sum += np.log(intensity_t)
        
        # 计算对数似然的第二部分：积分项
        integral = mu * T
        for t in event_times:
            integral += (alpha / beta) * (1 - np.exp(-beta * (T - t)))
        
        # 负对数似然
        return -log_intensity_sum + integral
    
    def fit(self, train_data, time_steps=None):
        """
        拟合模型参数
        
        参数:
        - train_data: 训练数据，形状为(时间步数, 网格大小, 网格大小)
        - time_steps: 可选，时间步序列
        """
        n_steps, height, width = train_data.shape
        self.grid_size = height  # 假设高度和宽度相等
        
        # 创建网格邻接关系
        self.grid_adjacency = self._create_grid_adjacency(self.grid_size)
        
        # 如果没有提供时间步，则默认为连续整数
        if time_steps is None:
            time_steps = np.arange(n_steps)
        
        print(f"开始训练Hawkes模型，网格大小: {height}x{width}，时间步数: {n_steps}")
        
        # 统计非零单元格比例
        nonzero_count = np.count_nonzero(train_data)
        total_cells = train_data.size
        nonzero_ratio = nonzero_count / total_cells
        print(f"训练数据中非零单元格占比: {nonzero_ratio:.4f} ({nonzero_count}/{total_cells})")
        
        # 对每个网格单独拟合参数
        for i in tqdm(range(height), desc="拟合网格行"):
            for j in range(width):
                grid_idx = i * width + j
                
                # 提取当前网格的时间序列
                grid_series = train_data[:, i, j]
                
                # 找出事件发生的时间点（车辆数大于0的时间）
                event_indices = np.where(grid_series > 0)[0]
                
                # 如果该网格有足够的事件，单独拟合参数
                if len(event_indices) >= 3:  # 至少需要3个事件点来拟合参数
                    event_times = time_steps[event_indices]
                    event_counts = grid_series[event_indices]
                    
                    # 对于每个大于1的事件，我们需要复制多次来表示多个事件
                    all_event_times = []
                    for t_idx, count in zip(event_times, event_counts):
                        all_event_times.extend([t_idx] * int(count))
                    
                    # 只有当有足够的事件时才拟合
                    if len(all_event_times) >= 3:
                        try:
                            # 初始参数猜测
                            x0 = [self.global_alpha, self.global_beta, self.global_mu]
                            
                            # 约束条件：所有参数必须为正
                            bounds = [(0.001, 10), (0.001, 10), (0.001, 5)]
                            
                            # 拟合参数
                            T = time_steps[-1] + 1  # 观察结束时间
                            
                            result = minimize(
                                self.log_likelihood,
                                x0,
                                args=(all_event_times, T),
                                bounds=bounds,
                                method='L-BFGS-B'
                            )
                            
                            if result.success:
                                # 保存拟合的参数
                                self.grid_params[grid_idx] = tuple(result.x)
                            else:
                                # 拟合失败，使用默认参数
                                alpha = self.global_alpha
                                beta = self.global_beta
                                # 调整mu以匹配观察到的事件率
                                event_rate = len(all_event_times) / (T - all_event_times[0])
                                mu = max(0.01, event_rate / 2)  # 设置一个合理的下限
                                self.grid_params[grid_idx] = (alpha, beta, mu)
                                
                        except Exception as e:
                            print(f"网格({i},{j})拟合参数失败: {e}")
                            # 使用默认参数
                            self.grid_params[grid_idx] = (
                                self.global_alpha, 
                                self.global_beta, 
                                self.global_mu
                            )
                    else:
                        # 设置针对稀疏事件优化的参数
                        # 提高mu并降低alpha，使模型更容易捕捉稀疏事件
                        event_rate = len(event_indices) / n_steps
                        alpha = self.global_alpha * 0.8  # 减小历史依赖
                        beta = self.global_beta * 1.2   # 加快衰减
                        mu = max(0.05, event_rate * 2)  # 增大基线概率
                        self.grid_params[grid_idx] = (alpha, beta, mu)
                else:
                    # 对于极度稀疏的网格，设置专门的参数
                    if np.sum(grid_series) > 0:
                        # 有极少量事件的网格
                        event_rate = np.sum(grid_series) / n_steps
                        alpha = self.global_alpha * 0.5  # 大幅减小历史依赖
                        beta = self.global_beta * 1.5    # 大幅加快衰减
                        mu = max(0.01, event_rate * 3)   # 大幅增大基线概率
                    else:
                        # 完全没有事件的网格
                        alpha = self.global_alpha * 0.1
                        beta = self.global_beta * 2.0
                        mu = 0.001  # 极小的基线概率
                        
                    self.grid_params[grid_idx] = (alpha, beta, mu)
        
        # 统计拟合参数分布
        alphas = [params[0] for params in self.grid_params.values()]
        betas = [params[1] for params in self.grid_params.values()]
        mus = [params[2] for params in self.grid_params.values()]
        
        print("\n拟合参数统计:")
        print(f"Alpha - 均值: {np.mean(alphas):.4f}, 中位数: {np.median(alphas):.4f}, 范围: [{np.min(alphas):.4f}, {np.max(alphas):.4f}]")
        print(f"Beta - 均值: {np.mean(betas):.4f}, 中位数: {np.median(betas):.4f}, 范围: [{np.min(betas):.4f}, {np.max(betas):.4f}]")
        print(f"Mu - 均值: {np.mean(mus):.4f}, 中位数: {np.median(mus):.4f}, 范围: [{np.min(mus):.4f}, {np.max(mus):.4f}]")
        
        print(f"模型训练完成，共拟合 {len(self.grid_params)} 个网格参数")
    
    def predict(self, history_data, next_step, time_steps):
        """
        预测下一个时间步的网格车辆数
        
        参数:
        - history_data: 历史数据，形状为(历史时间步数, 网格大小, 网格大小)
        - next_step: 要预测的下一个时间步
        - time_steps: 历史数据对应的时间步列表
        
        返回:
        - 预测结果，形状为(网格大小, 网格大小)
        """
        # 获取历史数据的形状
        n_steps, height, width = history_data.shape
        
        # 创建结果数组
        predictions = np.zeros((height, width))
        
        # 对每个网格位置进行预测
        for x in range(height):
            for y in range(width):
                # 获取该网格的参数
                if (x, y) in self.grid_params:
                    alpha, beta, mu = self.grid_params[(x, y)]
                else:
                    alpha, beta, mu = self.global_alpha, self.global_beta, self.global_mu
                
                # 提取历史数据中该网格的计数
                counts = history_data[:, x, y]
                
                # 构建事件时间列表
                event_times = []
                for t_idx, count in enumerate(counts):
                    if count > 0:
                        t = time_steps[t_idx]
                        event_times.extend([t] * int(count))
                
                # 计算下一个时间步的强度
                intensity = self.intensity(next_step, (x, y), event_times)
                
                # 强度可以视为期望事件数
                predictions[x, y] = intensity
        
        return predictions
    
    def evaluate(self, test_data, train_data, time_steps, test_time_steps):
        """
        评估模型
        
        参数:
        - test_data: 测试数据，形状为(测试时间步数, 网格大小, 网格大小)
        - train_data: 训练数据，形状为(训练时间步数, 网格大小, 网格大小)
        - time_steps: 训练数据的时间步列表
        - test_time_steps: 测试数据的时间步列表
        
        返回:
        - mse: 均方误差
        - mae: 平均绝对误差
        """
        n_test_steps = len(test_data)
        predictions = []
        
        # 历史数据初始化为训练数据
        history_data = train_data
        history_times = time_steps
        
        # 对每个测试时间步进行预测
        for i in range(n_test_steps):
            # 下一个要预测的时间步
            next_step = test_time_steps[i]
            
            # 预测下一个时间步的网格车辆数
            pred = self.predict(history_data, next_step, history_times)
            predictions.append(pred)
            
            # 更新历史数据（添加真实的测试数据）
            if i < n_test_steps - 1:  # 除了最后一步，每次都需要更新历史
                history_data = np.vstack((history_data, test_data[i:i+1]))
                history_times = np.append(history_times, next_step)
        
        # 将预测结果转换为数组
        predictions = np.array(predictions)
        
        # 计算误差
        mse = np.mean((predictions - test_data) ** 2)
        mae = np.mean(np.abs(predictions - test_data))
        
        return mse, mae

if __name__ == "__main__":
    # 加载训练和测试数据
    train_data = np.load("train_data.npy")
    test_data = np.load("test_data.npy")
    time_steps = np.load("time_steps.npy")
    
    # 分割时间步
    train_time_steps = time_steps[:len(train_data)]
    test_time_steps = time_steps[len(train_data):]
    
    # 创建并训练Hawkes网格模型
    hawkes_model = HawkesGridModel()
    hawkes_model.fit(train_data, train_time_steps)
    
    # 评估模型
    mse, mae = hawkes_model.evaluate(test_data, train_data, train_time_steps, test_time_steps)
    print(f"Hawkes网格模型评估结果 - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # 保存模型参数
    np.save("hawkes_grid_params.npy", hawkes_model.grid_params) 