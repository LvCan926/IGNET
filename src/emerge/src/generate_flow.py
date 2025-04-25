import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
import h5py

def load_od_matrix(file_path):
    """
    加载OD矩阵
    
    参数:
    - file_path: OD矩阵文件路径
    
    返回:
    - od_matrix: OD矩阵数据
    - adjacency_map: 邻域映射
    """
    print(f"加载OD矩阵: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到OD矩阵文件: {file_path}")
    
    # 检查文件扩展名判断格式
    if file_path.endswith('.h5'):
        # 使用h5py加载h5格式
        with h5py.File(file_path, 'r') as f:
            # 加载OD矩阵
            od_matrix = f['od_matrix'][:]
            
            # 加载邻域映射
            adjacency_map = {}
            adj_group = f['adjacency_map']
            for key in adj_group.keys():
                grid_index = int(key)
                adjacency_map[grid_index] = list(adj_group[key][:])
            
            print(f"加载了h5格式OD矩阵，形状: {od_matrix.shape}")
            return od_matrix, adjacency_map
    else:
        # 使用numpy加载npy格式
        data = np.load(file_path, allow_pickle=True)
        
        # 检查是否是新格式的OD矩阵（包含邻域映射）
        if isinstance(data, np.ndarray) and data.dtype == np.dtype('O') and isinstance(data.item(), dict):
            data_dict = data.item()
            od_matrix = data_dict['od_matrix']
            
            # 适配新旧命名格式
            if 'adjacency_map' in data_dict:
                adjacency_map = data_dict['adjacency_map']
            elif 'adjacent_grids_map' in data_dict:
                adjacency_map = data_dict['adjacent_grids_map']
            else:
                adjacency_map = None
            
            print(f"加载了npy格式OD矩阵，形状: {od_matrix.shape}")
            return od_matrix, adjacency_map
        else:
            raise ValueError("不支持的OD矩阵格式。请使用包含邻域映射的新格式OD矩阵。")

def generate_flow_from_vehicles(new_vehicles, od_matrix_data, time_offset=1128):
    """
    根据新生车辆数和OD矩阵生成新生流量
    
    参数:
    - new_vehicles: 新生车辆数，形状为(time_steps, grid_size, grid_size)
    - od_matrix_data: OD矩阵数据，元组(od_matrix, adjacency_map)
    - time_offset: 时间步偏移量，用于映射新生车辆时间步到OD矩阵时间步
    
    返回:
    - flow: 新生流量，形状为(time_steps, 2, grid_size, grid_size)
    """
    print("根据新生车辆数和OD矩阵生成新生流量...")
    
    # 解析OD矩阵数据
    od_matrix, adjacency_map = od_matrix_data
    
    if adjacency_map is None:
        raise ValueError("缺少邻域映射。请确保使用了正确的OD矩阵格式。")
    
    time_steps, height, width = new_vehicles.shape
    grid_size = height  # 假设height和width相等
    
    # 检查OD矩阵格式
    od_time_steps, od_grid_size, n_neighbors, _ = od_matrix.shape
    assert od_grid_size == grid_size * grid_size, "OD矩阵网格大小与新生车辆网格大小不匹配"
    print(f"OD矩阵有 {n_neighbors} 个邻域索引(包括自身)")
    print("使用标准邻域流动模式")
    
    # 验证邻域映射表格式
    if isinstance(adjacency_map, dict):
        print("使用全局邻域映射字典")
        # 验证映射是否完整
        if len(adjacency_map) != grid_size * grid_size:
            print(f"警告: 邻域映射字典中网格数量 ({len(adjacency_map)}) 与实际网格数量 ({grid_size * grid_size}) 不匹配")
        
        # 检查邻域列表长度分布
        neighbor_lengths = {}
        for grid, neighbors in adjacency_map.items():
            length = len(neighbors)
            if length not in neighbor_lengths:
                neighbor_lengths[length] = 0
            neighbor_lengths[length] += 1
        
        print("邻域列表长度分布:")
        for length, count in sorted(neighbor_lengths.items()):
            print(f"  长度 {length}: {count} 个网格 ({100*count/(grid_size*grid_size):.1f}%)")
    else:
        raise ValueError("邻域映射不是字典格式。请使用正确的OD矩阵格式。")
    
    # 初始化新生流量：(时间步数, 2, 网格高度, 网格宽度)
    # 第二个维度表示流入和流出
    flow = np.zeros((time_steps, 2, grid_size, grid_size))
    
    # 对每个时间步计算流量分配
    for t in tqdm(range(time_steps), desc="计算流量"):
        # 映射到OD矩阵的时间步
        od_t = t % od_time_steps  # 确保时间步在OD矩阵范围内
        
        # 对每个网格处理新生车辆
        for i in range(grid_size):
            for j in range(grid_size):
                # 将2D网格索引转换为1D索引
                grid_idx = i * grid_size + j
                
                # 当前网格的新生车辆数
                vehicles = new_vehicles[t, i, j]
                
                if vehicles > 0:
                    # 初始化当前网格在当前时间步的流出量为0（不是所有车辆都立即流出）
                    # 实际流出量会根据不同目的地的出行时间在不同时间步累加
                    
                    # 处理每个邻域位置
                    for adj_idx in range(1, n_neighbors):  # 跳过索引0（自身）
                        # 获取比例和平均时间
                        ratio = od_matrix[od_t, grid_idx, adj_idx, 0]
                        avg_time = od_matrix[od_t, grid_idx, adj_idx, 1]
                        
                        if ratio > 0:
                            try:
                                # 获取目标网格索引
                                if grid_idx not in adjacency_map:
                                    # 网格不在邻域映射中
                                    continue
                                    
                                if adj_idx - 1 >= len(adjacency_map[grid_idx]):
                                    # 邻域索引超出范围
                                    continue
                                    
                                dest_idx = adjacency_map[grid_idx][adj_idx - 1]
                            except (IndexError, KeyError) as e:
                                # 如果邻域索引超出了该网格的邻域列表长度，或网格不存在于映射中，跳过
                                continue
                            
                            # 计算流向目标网格的车辆数
                            flow_count = vehicles * ratio
                            
                            # 将目标1D索引转换回2D索引
                            dest_i = dest_idx // grid_size
                            dest_j = dest_idx % grid_size
                            
                            # 计算出发和到达的时间步
                            departure_time = t
                            
                            # 根据平均时间确定到达时间
                            if avg_time > 0:
                                arrival_time = int(t + avg_time)
                            else:
                                # 如果平均时间为0，则认为车辆立即到达目的地
                                arrival_time = t + 1
                            
                            # 添加到流出量（在出发时间记录）
                            # 确保在预测范围内
                            if departure_time + 1 < time_steps:
                                flow[departure_time + 1, 0, i, j] += flow_count
                            
                            # 添加到流入量（在到达时间记录）
                            # 确保在预测范围内
                            if arrival_time < time_steps:
                                flow[arrival_time, 1, dest_i, dest_j] += flow_count
                                # 将到达的车辆添加到目标网格，以便继续生成流量
                                new_vehicles[arrival_time, dest_i, dest_j] += flow_count
    
    # 后处理：检查和调整全局流入流出总量
    total_out = np.sum(flow[:, 0])
    total_in = np.sum(flow[:, 1])
    
    print(f"流量检查 - 总流出量: {total_out:.2f}, 总流入量: {total_in:.2f}")
    
    # 如果流入量明显与流出量不同，检查OD矩阵中的比例值
    if abs(total_in - total_out) / total_out > 0.1:  # 差异超过10%
        print("警告: 流入量和流出量差异较大，可能是由于车辆在预测时间范围外到达")
        
        # 计算每个时间步的流入流出比例
        t_inflow_summary = []
        for t in range(time_steps):
            t_out = np.sum(flow[t, 0])
            t_in = np.sum(flow[t, 1])
            if t_out > 0:
                t_inflow_summary.append((t, t_out, t_in, t_in/t_out))
        
        # 打印最大差异的几个时间步
        print("差异最大的时间步:")
        sorted_summary = sorted(t_inflow_summary, key=lambda x: abs(x[3]-1), reverse=True)
        for t, t_out, t_in, ratio in sorted_summary[:5]:  # 打印前5个差异最大的
            print(f"  时间步 {t}: 流出量={t_out:.2f}, 流入量={t_in:.2f}, 比例={ratio:.4f}")
    
    print("新生流量生成完成！")
    return flow

def select_best_model(hawkes_mse, nn_mse):
    """
    根据MSE选择误差较小的模型
    
    参数:
    - hawkes_mse: Hawkes模型的MSE
    - nn_mse: 神经网络模型的MSE
    
    返回:
    - model_name: 选择的模型名称，'hawkes'或'nn'
    """
    if hawkes_mse <= nn_mse:
        print(f"Hawkes模型的MSE({hawkes_mse:.4f})较小，选择Hawkes模型")
        return 'hawkes'
    else:
        print(f"神经网络模型的MSE({nn_mse:.4f})较小，选择神经网络模型")
        return 'nn'

def generate_test_predictions(test_time_steps, train_data, train_time_steps, hawkes_model, nn_model, select_model='best'):
    """
    生成测试集的预测值
    
    参数:
    - test_time_steps: 测试集的时间步
    - train_data: 训练数据，形状为(训练时间步数, 网格大小, 网格大小)
    - train_time_steps: 训练数据的时间步
    - hawkes_model: Hawkes模型
    - nn_model: 神经网络模型
    - select_model: 'best','hawkes'或'nn'，表示选择的模型
    
    返回:
    - predictions: 测试集的预测值，形状为(测试时间步数, 网格大小, 网格大小)
    """
    print(f"使用{select_model}模型生成测试集预测...")
    
    # 只取1128-1139这12个时间步
    time_range_mask = (test_time_steps >= 1128) & (test_time_steps <= 1139)
    selected_test_time_steps = test_time_steps[time_range_mask]
    
    n_test_steps = len(selected_test_time_steps)
    print(f"选择的时间范围: 1128-1139，共{n_test_steps}个时间步")
    
    # 初始化历史数据
    history_data = train_data.copy()
    history_times = train_time_steps.copy()
    predictions = []
    
    # 分析训练数据中非零样本的分布
    nonzero_count = np.count_nonzero(train_data)
    total_count = train_data.size
    nonzero_ratio = nonzero_count / total_count
    print(f"训练数据中非零样本占比: {nonzero_ratio:.6f} ({nonzero_count}/{total_count})")
    
    # 计算训练数据中非零样本的平均值和标准差
    nonzero_values = train_data[train_data > 0]
    nonzero_mean = np.mean(nonzero_values) if len(nonzero_values) > 0 else 0
    nonzero_std = np.std(nonzero_values) if len(nonzero_values) > 0 else 0
    print(f"非零样本平均值: {nonzero_mean:.4f}, 标准差: {nonzero_std:.4f}")
    
    # 创建一个历史热点图，记录每个网格出现非零值的频率
    grid_size = train_data.shape[1]
    hotspot_map = np.zeros((grid_size, grid_size))
    for t in range(len(train_data)):
        hotspot_map += (train_data[t] > 0).astype(float)
    
    # 归一化热点图
    hotspot_map = hotspot_map / len(train_data)
    print(f"热点网格数量(频率>5%): {np.sum(hotspot_map > 0.05)}")
    
    # 设置概率阈值以确定预测是否为0
    existence_threshold = 0.02  # 低于此概率的预测为0
    
    # 对每个测试时间步进行预测
    for i in range(n_test_steps):
        next_step = selected_test_time_steps[i]
        
        # 根据选择的模型进行预测
        if select_model == 'hawkes':
            # Hawkes模型预测
            pred_raw = hawkes_model.predict(history_data, next_step, history_times)
            
            # 应用热点图进行校正
            pred = pred_raw.copy()
            for x in range(grid_size):
                for y in range(grid_size):
                    # 如果该网格不是热点且预测值很小，则设为0
                    if hotspot_map[x, y] < 0.03 and pred[x, y] < 0.1:
                        pred[x, y] = 0
                    
                    # 对于热点区域，提高预测值
                    if hotspot_map[x, y] > 0.1 and pred[x, y] > 0:
                        pred[x, y] *= (1 + hotspot_map[x, y])
                        
            # 使用存在性阈值将小值归零
            pred[pred < existence_threshold] = 0
                        
        elif select_model == 'nn':
            # 确保有足够的历史数据
            if len(history_data) >= 5:
                # 神经网络模型预测
                pred_raw = nn_model.predict(history_data[-5:])
                
                # 应用热点图进行校正
                pred = pred_raw.copy()
                for x in range(grid_size):
                    for y in range(grid_size):
                        # 如果该网格不是热点且预测值很小，则设为0
                        if hotspot_map[x, y] < 0.03 and pred[x, y] < 0.1:
                            pred[x, y] = 0
                        
                        # 对于热点区域，提高预测值
                        if hotspot_map[x, y] > 0.1 and pred[x, y] > 0:
                            pred[x, y] *= (1 + hotspot_map[x, y])
                
                # 使用存在性阈值将小值归零
                pred[pred < existence_threshold] = 0
            else:
                raise ValueError("神经网络模型需要至少5个历史时间步")
        else:  # 'best'
            # 混合模型策略
            # 1. 使用Hawkes模型获取基础预测
            hawkes_pred = hawkes_model.predict(history_data, next_step, history_times)
            
            # 2. 如果有足够的历史数据，也使用神经网络模型预测
            if len(history_data) >= 5:
                nn_pred = nn_model.predict(history_data[-5:])
                
                # 根据之前的MSE确定混合权重
                if i > 0:
                    last_actual = history_data[-1]
                    hawkes_mse = np.mean((hawkes_pred - last_actual) ** 2)
                    nn_mse = np.mean((nn_pred - last_actual) ** 2)
                    
                    # 计算两个模型的混合权重(权重与MSE成反比)
                    total_error = hawkes_mse + nn_mse
                    if total_error > 0:
                        hawkes_weight = 1 - (hawkes_mse / total_error)
                        nn_weight = 1 - (nn_mse / total_error)
                    else:
                        # 如果两个模型都完美预测，则平均
                        hawkes_weight = 0.5
                        nn_weight = 0.5
                    
                    # 归一化权重
                    sum_weights = hawkes_weight + nn_weight
                    hawkes_weight /= sum_weights
                    nn_weight /= sum_weights
                    
                    # 混合预测
                    pred_raw = hawkes_weight * hawkes_pred + nn_weight * nn_pred
                    print(f"时间步 {next_step}: Hawkes权重={hawkes_weight:.2f}, NN权重={nn_weight:.2f}")
                else:
                    # 第一个预测，平均两个模型的结果
                    pred_raw = (hawkes_pred + nn_pred) / 2
            else:
                # 没有足够的历史数据给神经网络，使用Hawkes
                pred_raw = hawkes_pred
            
            # 应用热点图和空间校正
            pred = pred_raw.copy()
            for x in range(grid_size):
                for y in range(grid_size):
                    # 如果该网格不是热点且预测值很小，则设为0
                    if hotspot_map[x, y] < 0.03 and pred[x, y] < 0.1:
                        pred[x, y] = 0
                    
                    # 对于热点区域，提高预测值
                    if hotspot_map[x, y] > 0.1 and pred[x, y] > 0:
                        pred[x, y] *= (1 + hotspot_map[x, y])
            
            # 使用存在性阈值将小值归零
            pred[pred < existence_threshold] = 0
        
        # 如果预测全为0，为热点区域添加少量随机新生车辆
        if np.sum(pred) == 0:
            print(f"警告: 时间步 {next_step} 的预测全为0，为热点区域添加随机值")
            hot_spots = np.where(hotspot_map > 0.1)
            if len(hot_spots[0]) > 0:
                # 随机选择一些热点并添加少量新生车辆
                num_spots = min(5, len(hot_spots[0]))
                random_indices = np.random.choice(len(hot_spots[0]), num_spots, replace=False)
                for idx in random_indices:
                    x, y = hot_spots[0][idx], hot_spots[1][idx]
                    # 添加一个符合历史非零样本分布的随机值
                    pred[x, y] = max(0.1, np.random.normal(nonzero_mean, nonzero_std/2))
        
        predictions.append(pred)
        
        # 更新历史数据
        # 注意：这里我们用预测值更新历史，因为没有真实值
        history_data = np.vstack((history_data, pred.reshape(1, *pred.shape)))
        history_times = np.append(history_times, next_step)
    
    predictions = np.array(predictions)
    
    # 最终统计
    final_nonzero = np.count_nonzero(predictions)
    final_total = predictions.size
    final_ratio = final_nonzero / final_total
    print(f"最终预测中非零样本占比: {final_ratio:.6f} ({final_nonzero}/{final_total})")
    print(f"预测的总新生车辆数: {np.sum(predictions):.2f}")
    
    return predictions

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成新生流量')
    parser.add_argument('--predicted-vehicles', type=str, default='Data/predicted_vehicles.npy', help='预测车辆数文件路径')
    parser.add_argument('--od-matrix', type=str, default='Data/enhanced_od_matrix.h5', help='OD矩阵文件路径')
    parser.add_argument('--output', type=str, default='Data/new_flow.npy', help='输出新生流量文件路径')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载预测车辆数
    print(f"加载预测车辆数: {args.predicted_vehicles}")
    predicted_vehicles = np.load(args.predicted_vehicles)
    print(f"预测车辆数形状: {predicted_vehicles.shape}")
    print(f"总的新生车辆数: {np.sum(predicted_vehicles)}")
    
    # 加载OD矩阵
    od_matrix_data = load_od_matrix(args.od_matrix)
    
    # 生成新生流量
    flow = generate_flow_from_vehicles(predicted_vehicles, od_matrix_data)
    
    # 保存新生流量
    print(f"保存新生流量到: {args.output}")
    np.save(args.output, flow)
    
    return flow

if __name__ == "__main__":
    main() 