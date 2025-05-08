import pandas as pd
import torch
import numpy as np

import argparse
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm, trange
from pathlib import Path
import random
import os


from utils import load_config_test, set_seeds
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics

import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch training & testing code for task-agnostic time-series prediction"
    )
    parser.add_argument(
        "--task", default="test", type=str, choices=['train', 'test', 'viz'],
        help="要执行的任务: train-训练, test-测试, viz-可视化"
    )
    # parser.add_argument("--config", type=str, default="cfg/config.yaml")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "tune"], default="test"
    )

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--save_model", action="store_true", help="save model")
    parser.add_argument(
        "--load_model", type=str, default=None, help="path of pre-trained model"
    )
    parser.add_argument("--logging_path", type=str, default=None)

    parser.add_argument(
        "--config_root",
        type=str,
        default="config/",
        help="root path to config file",
    )
    parser.add_argument("--scene", type=str, default="BJTaxi", help="scene name")

    parser.add_argument(
        "--aug_scene", action="store_true", help="trajectron++ augmentation"
    )
    parser.add_argument(
        "--w_mse", type=float, default=0, help="loss weight of mse_loss"
    )

    parser.add_argument("--clusterGMM", action="store_true")
    parser.add_argument(
        "--cluster_method", type=str, default="kmeans", help="clustering method"
    )
    parser.add_argument("--cluster_n", type=int, help="n cluster centers")
    parser.add_argument(
        "--cluster_name", type=str, default="", help="clustering model name"
    )
    parser.add_argument(
        "--manual_weights", nargs='+', default=None, type=int)

    parser.add_argument("--var_init", type=float, default=0.7, help="init var")
    parser.add_argument("--learnVAR", action="store_true")

    parser.add_argument("--use_clustering", default=False, action="store_true", help="使用轨迹聚类减少数据量")
    parser.add_argument("--cluster_count", type=int, default=2, help="每个场景每个时间步的聚类数量")

    return parser.parse_args()


def k_means(batch_x, ncluster=20, iter=5, disable_trange=True):
    """K-means聚类算法实现（批处理版本）
    
    参数:
        batch_x: 输入数据张量，形状为 [B, N, D] 
            B: batch大小, N: 每批样本数, D: 特征维度
        ncluster: 聚类中心数量 (默认20)
        iter: 迭代次数 (默认5)
        disable_trange: 是否禁用进度条 (默认True)
    
    返回:
        batch_c: 聚类中心张量，形状为 [B, ncluster, D]
    """
    
    # 获取输入数据的维度信息
    B, N, D = batch_x.size()  # B=batch数, N=样本数, D=特征维度
    
    # 初始化存储聚类中心的空张量（放在GPU上）
    batch_c = torch.Tensor().cuda()
    
    # 遍历每个batch（使用进度条显示）
    for i in trange(B, disable=disable_trange):
        # 取出当前batch的所有样本 [N, D]
        x = batch_x[i]
        
        # 初始化聚类中心：随机选择ncluster个样本作为初始中心
        c = x[torch.randperm(N)[:ncluster]]  # [ncluster, D]
        
        # K-means迭代优化
        for i in range(iter):
            # 步骤1: 分配样本到最近的中心
            # 计算所有样本到所有中心的距离 [N, ncluster]
            distances = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            # 获取每个样本最近的聚类中心索引 [N]
            a = distances.argmin(1)
            
            # 步骤2: 更新聚类中心
            # 计算每个簇的均值 [ncluster, D]
            c = torch.stack([
                x[a == k].mean(0)  # 对属于第k簇的样本求均值
                for k in range(ncluster)
            ])
            
            # 处理空簇问题（均值为NaN的簇）
            nanix = torch.any(torch.isnan(c), dim=1)  # 标记无效中心
            ndead = nanix.sum().item()  # 统计无效中心数量
            # 用随机样本替换无效中心
            c[nanix] = x[torch.randperm(N)[:ndead]]
        
        # 将当前batch的聚类中心添加到结果中 [1, ncluster, D]
        batch_c = torch.cat((batch_c, c.unsqueeze(0)), dim=0)
    
    return batch_c  # 返回所有batch的聚类中心 [B, ncluster, D]


def run_inference(cfg, model, metrics, data_loader, disable_tqdm=False):
    inference_start_time = time.time()
    model.eval()
    with torch.no_grad():
        pred_list = []
        gt_list = []
        obs_list = []

        obs_ts_list = []
        gt_ts_list = []
        
        # 新增：存储cluster_size数据
        cluster_size_list = []
        
        #print("开始加载数据...")
        for i, data_dict in enumerate(tqdm(data_loader, leave=False, disable=disable_tqdm)):
            #print(f"批次 {i} 数据字典的键:", data_dict.keys())
            if "obs_ts" in data_dict:
                batch_obs_ts = data_dict["obs_ts"]
                if isinstance(batch_obs_ts, torch.Tensor):
                    batch_obs_ts = batch_obs_ts.cpu().numpy()
                obs_ts_list.append(batch_obs_ts)
                
            if "gt_ts" in data_dict:
                batch_gt_ts = data_dict["gt_ts"]
                if isinstance(batch_gt_ts, torch.Tensor):
                    batch_gt_ts = batch_gt_ts.cpu().numpy()
                gt_ts_list.append(batch_gt_ts)
            
            # 新增：检查并获取cluster_size数据
            batch_cluster_size = None
            if "cluster_size" in data_dict:
                batch_cluster_size = data_dict["cluster_size"]
                if isinstance(batch_cluster_size, torch.Tensor):
                    batch_cluster_size = batch_cluster_size.cpu().numpy()
                    # 将cluster_size转换为整数
                    batch_cluster_size = batch_cluster_size.astype(np.int32)
                cluster_size_list.append(batch_cluster_size)
                # 调试用
                print(f"批次 {i} 包含cluster_size，形状: {batch_cluster_size.shape}")
            else:
                print(f"批次 {i} 不包含cluster_size")
            
            pred_list_i = []
            gt_list_i = []

            data_dict = {
                k: (
                    data_dict[k].cuda()
                    if isinstance(data_dict[k], torch.Tensor)
                    else data_dict[k]
                )
                for k in data_dict
            }
            
            #print(f"批次 {i} 数据形状:")
            #for k, v in data_dict.items():
            #    if isinstance(v, torch.Tensor):
            #        print(f"{k}: {v.shape}")
            
            dist_args = model.encoder(data_dict)

            if cfg.MGF.ENABLE:
                base_pos = model.get_base_pos(data_dict).clone()
            else:
                base_pos = (
                    model.get_base_pos(data_dict)[:, None]
                    .expand(-1, cfg.MGF.POST_CLUSTER, -1)
                    .clone()
                )  # (B, 20, 2)
            dist_args = dist_args[:, None].expand(-1, cfg.MGF.POST_CLUSTER, -1, -1)

            sampled_seq = model.flow.sample(
                base_pos, cond=dist_args, n_sample=cfg.MGF.POST_CLUSTER
            )

            dict_list = []
            for i in range(cfg.MGF.POST_CLUSTER):
                data_dict_i = deepcopy(data_dict)
                data_dict_i[("pred_st", 0)] = sampled_seq[:, i]
                if torch.sum(torch.isnan(data_dict_i[("pred_st", 0)])):
                    data_dict_i[("pred_st", 0)] = torch.where(
                        torch.isnan(data_dict_i[("pred_st", 0)]),
                        data_dict_i["obs_st"][:, 0, None, 2:4].expand(
                            data_dict_i[("pred_st", 0)].size()
                        ),
                        data_dict_i[("pred_st", 0)],
                    )
                dict_list.append(data_dict_i)

            dict_list = metrics.denormalize(dict_list)

            process_results_start_time = time.time()
            for data_dict in dict_list:
                obs_list_i = data_dict["obs"].cpu().numpy()
                pred_traj_i = data_dict[("pred", 0)].cpu().numpy()  # (B,12,2)
                pred_list_i.append(pred_traj_i)
                gt_list_i = data_dict["gt"].cpu().numpy()
            pred_list_i = np.array(pred_list_i).transpose(1, 0, 2, 3)

            pred_list.append(pred_list_i)
            gt_list.append(gt_list_i)
            obs_list.append(obs_list_i)

    # 检查列表是否为空
    if len(pred_list) == 0:
        print("警告: pred_list为空，没有预测数据")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None
    
    concat_start_time = time.time()
    pred_list = np.concatenate(pred_list, axis=0)
    gt_list = np.concatenate(gt_list, axis=0)
    obs_list = np.concatenate(obs_list, axis=0)
    
    # 拼接后：
    if obs_ts_list:
        obs_ts_array = np.concatenate(obs_ts_list, axis=0)
    else:
        obs_ts_array = np.array([])
        
    if gt_ts_list:
        gt_ts_array = np.concatenate(gt_ts_list, axis=0)
    else:
        gt_ts_array = np.array([])
    
    # 新增：处理cluster_size数据
    cluster_size_array = None
    if cluster_size_list:
        cluster_size_array = np.concatenate(cluster_size_list, axis=0)
    concat_time = time.time() - concat_start_time
    
    # 检查cluster_size_array中是否存在大于1的值
    if cluster_size_array is not None:
        print(f"cluster_size_array中存在大于1的值: {np.any(cluster_size_array > 1)}")
        print(f"cluster_size_array中大于1的值: {cluster_size_array[cluster_size_array > 1]}")

    pred_list_flatten_start_time = time.time()
    pred_list_flatten = torch.Tensor(
        pred_list.reshape(pred_list.shape[0], cfg.MGF.POST_CLUSTER, -1)
    ).cuda()
    pred_list_flatten_time = time.time() - pred_list_flatten_start_time

    kmeans_start_time = time.time()
    pred_list = (
        k_means(pred_list_flatten, disable_trange=disable_tqdm).cpu().numpy().reshape(pred_list.shape[0], 20, -1, 2)
    )
    kmeans_time = time.time() - kmeans_start_time
    print(f"K-means 聚类耗时: {kmeans_time:.4f} s")

    # 计算每条轨迹的ADE（Average Displacement Error）
    calc_ade_start_time = time.time()
    l2_dis = np.sqrt(((pred_list - gt_list[:, np.newaxis, :, :]) ** 2).sum(-1))  # 形状 (B, 20, 12)
    ade_per_traj = l2_dis.mean(axis=2)  # (B, 20)
    best_indices = np.argmin(ade_per_traj, axis=1)  # (B,)
    best_preds = pred_list[np.arange(len(pred_list)), best_indices]  # (B, 12, 2)
    calc_ade_time = time.time() - calc_ade_start_time
    
    print(f"best_preds.shape: {best_preds.shape}")

    inference_end_time = time.time()
    print(f"总推理耗时: {inference_end_time-inference_start_time:.4f} s")

    return obs_list, gt_list, pred_list, best_preds, obs_ts_array, gt_ts_array, cluster_size_array


def evaluate_metrics(args, gt_list, pred_list):
    metrics_start_time = time.time()
    l2_dis = np.sqrt(((pred_list - gt_list[:, np.newaxis, :, :]) ** 2).sum(-1))
    minade = l2_dis.mean(-1).min(-1)
    minfde = l2_dis[:, :, -1].min(-1)
    if args.scene == "eth":
        minade /= 0.6
        minfde /= 0.6
    elif args.scene == "sdd":
        minade *= 50
        minfde *= 50
    metrics_end_time = time.time()
    print(f"评估指标计算耗时: {metrics_end_time-metrics_start_time:.4f} s")
    return minade.mean(), minfde.mean()


def aggregate_flow(traj, gt_ts_array, grid_size=32, time_start=7200, time_end=10079, cluster_size_array=None):
    """
    基于轨迹计算流量（通过网格聚合）
    只计算特定时间范围的流量
    
    参数:
    traj: 输入的轨迹，形状为 (B, T, 2)，B是batch size，T是轨迹长度，2是(x, y)坐标
    gt_ts_array: 时间戳数组，形状为 (B, T)
    grid_size: 网格大小，默认32x32
    cluster_size_array: 簇大小数组，形状为 (B, T)，如果为None则默认所有流量为1
    
    返回:
    flow: 流量矩阵，形状为 (time_count, 2, grid_size, grid_size)
    """
    
    # 调试信息：查看cluster_size_array是否存在及其形状
    if cluster_size_array is not None:
        print(f"聚合流量：cluster_size_array存在，形状: {cluster_size_array.shape}")
        print(f"样本数量: {traj.shape[0]}, cluster_size_array样本数: {len(cluster_size_array)}")
        print(f"前5个样本的cluster_size值: {[cluster_size_array[i][0] if i < len(cluster_size_array) else None for i in range(min(5, len(cluster_size_array)))]}")
    else:
        print("聚合流量：未提供cluster_size_array，所有流量权重默认为1")
    
    # 北京出租车数据集边界
    min_lon = 116.2503565
    max_lon = 116.50032165
    min_lat = 39.7997164
    max_lat = 39.9999216
    
    time_count = time_end - time_start + 1
    
    # 初始化流量矩阵 - 时间步, 方向(流出/流入), 网格行, 网格列
    flow = np.zeros((time_count, 2, grid_size, grid_size), dtype=np.float32)
    
    # 记录预测时间步的最大最小值
    pred_time_max = 0
    pred_time_min = 1000000
    
    # 使用的实际样本数
    valid_samples = 0
    
    # 处理每个样本的轨迹
    for b in range(traj.shape[0]):
        sample_traj = traj[b]  # 获取单个样本的轨迹
        sample_gt_ts_array = gt_ts_array[b]
        
        # 获取该样本的cluster_size，如果可用
        sample_cluster_size = None
        if cluster_size_array is not None and b < len(cluster_size_array):
            sample_cluster_size = cluster_size_array[b]
            # 调试用：打印样本的cluster_size值
            if b < 5:  # 只打印前5个样本的信息
                print(f"样本 {b} 的 cluster_size: {sample_cluster_size[0]}")
        
        # 更新预测时间步的最大最小值
        pred_time_max = max(pred_time_max, sample_gt_ts_array.max())
        pred_time_min = min(pred_time_min, sample_gt_ts_array.min())
        
        # 从第二个点开始遍历
        for t in range(1, sample_traj.shape[0]):
            sample_gt_ts = sample_gt_ts_array[t]
            
            # 只处理指定时间范围内的数据
            if not (time_start <= sample_gt_ts <= time_end):
                continue
            
            # 获取前一个点和当前点的坐标
            prev_x, prev_y = sample_traj[t-1, 0], sample_traj[t-1, 1]
            curr_x, curr_y = sample_traj[t, 0], sample_traj[t, 1]
            
            # 计算时间步
            time_step = int(sample_gt_ts) - time_start
            
            # 确保时间步在有效范围内
            if not (0 <= time_step < time_count):
                continue
            
            # 计算该点的流量值（默认为1，如果有cluster_size则使用其值）
            flow_value = 1  # Default value
            if sample_cluster_size is not None:
                # 因为 sample_cluster_size 中所有值都相同 (代表该样本的簇大小)
                # 所以直接取第一个元素即可
                current_cluster_size = int(sample_cluster_size[0])
                if current_cluster_size > 1:  # 使用有效的簇大小
                    flow_value = current_cluster_size
                    if b < 5 and t == 1:  # 只对前5个样本的第一个轨迹点打印
                        print(f"样本 {b}, 时间步 {t}: 使用流量值 {flow_value}")
                        
            
            # 映射到网格坐标
            prev_row, prev_col = map_to_grid(prev_x, prev_y, min_lon, max_lon, min_lat, max_lat, grid_size)
            curr_row, curr_col = map_to_grid(curr_x, curr_y, min_lon, max_lon, min_lat, max_lat, grid_size)
            
            # 只有当区域发生变化时才更新流量
            if (prev_row != curr_row) or (prev_col != curr_col):
                flow[time_step, 0, prev_row, prev_col] += flow_value  # 流出量
                flow[time_step, 1, curr_row, curr_col] += flow_value  # 流入量
                valid_samples += 1
    
    print(f"预测时间步的最大最小值: {pred_time_max}/{pred_time_min}")
    print(f"有效样本数: {valid_samples}")
    
    return flow


def map_to_grid(lon, lat, min_lon, max_lon, min_lat, max_lat, grid_size):
    """将经纬度映射到网格坐标"""
    lat_step = (max_lat - min_lat) / grid_size
    lon_step = (max_lon - min_lon) / grid_size
    row = int((lat - min_lat) / lat_step)
    col = int((lon - min_lon) / lon_step)
    row = max(0, min(row, grid_size-1))
    col = max(0, min(col, grid_size-1))
    return row, col


def calculate_metrics(pred, truth, direction):
    """计算RMSE和MAE，direction=0表示流出量，direction=1表示流入量"""
    pred_flat = pred[:, direction].flatten()
    truth_flat = truth[:, direction].flatten()
    
    mse = np.mean((pred_flat - truth_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - truth_flat))
    return rmse, mae


def test(cfg, args, logger=None):
    """
    运行模型推理测试

    Args:
        cfg (CfgNode): 配置节点
        args (argparse.Namespace): 命令行参数
        logger (logging.Logger, optional): 日志记录器. Defaults to None.
    """
    # args.task在命令行解析时已设置，此处不再需要设置
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    load_config_start_time = time.time()
    # real_flow = np.load(f"mydata/real/real_BJ_2880_6-7.npy")
    # pred_flow = np.load(f"mydata/traj/pred_BJ_2880_6-7.npy")
    
    # print(f"real_flow.shape: {real_flow.shape}")
    # print(f"pred_flow.shape: {pred_flow.shape}")
    
    scene = args.scene
    args.load_model = f"./checkpoint/{scene}.ckpt"
    args.config_file = f"./config/{scene}.yml"
    cfg = load_config_test(args)

    args.task = "test"
    
    args.clusterGMM = cfg.MGF.ENABLE
    args.cluster_n = cfg.MGF.CLUSTER_N
    args.var_init = cfg.MGF.VAR_INIT
    args.learn_var = cfg.MGF.VAR_LEARNABLE
    load_config_end_time = time.time()

    model_build_start_time = time.time()
    model = Build_Model(cfg, args)
    model_build_end_time = time.time()
    print(f"模型构建耗时: {model_build_end_time-model_build_start_time:.4f} s")

    # 修改加载方式，使用strict=False允许部分加载
    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint["state"], strict=False)
    print("预训练模型已加载（不包含判别器部分）")
    
    dataloader_start_time = time.time()
    # 1. Setup dataloaders
    test_loader = unified_loader(
            cfg, 
            rand=False, 
            split="test",
            use_clustering=args.use_clustering, 
            cluster_count=args.cluster_count 
        )
    metrics = Build_Metrics(cfg)
    dataloader_end_time = time.time()
    print(f"数据加载器初始化耗时: {dataloader_end_time-dataloader_start_time:.4f} s")

    grid_size = 32  # 设置网格大小

    for i_trial in range(1):
        set_seeds(random.randint(0,1000))

        inference_start_time = time.time()
        obs_list, gt_list, pred_list, best_preds_list, obs_ts_array, gt_ts_array, cluster_size_array = run_inference(cfg, model, metrics, test_loader, disable_tqdm=True)
        inference_end_time = time.time()
        print(f"推理总耗时: {inference_end_time-inference_start_time:.4f} s")
        
        minade, minfde = evaluate_metrics(args, gt_list, pred_list)
        print(f"{args.scene} test {i_trial}:\n {minade}/{minfde}")
        
        # 直接聚合轨迹流量，使用cluster_size
        pred_flow = aggregate_flow(best_preds_list, gt_ts_array, grid_size, time_start=7200, time_end=10079, cluster_size_array=cluster_size_array)
        real_flow = aggregate_flow(gt_list, gt_ts_array, grid_size, time_start=7200, time_end=10079, cluster_size_array=cluster_size_array)
        
        print(f"pred_flow.shape: {pred_flow.shape}")
        print(f"real_flow.shape: {real_flow.shape}")
        
        # 保存流量矩阵到npy文件
        np.save(f"data/his_BJ_2880_6-7.npy", pred_flow)
        print(f"流量矩阵已保存到 data文件夹")
        
        # 计算流量误差（不依赖于模型）
        outflow_rmse, outflow_mae = calculate_metrics(pred_flow, real_flow, direction=0)
        inflow_rmse, inflow_mae = calculate_metrics(pred_flow, real_flow, direction=1)
        
        # 计算真实流出量和预测流出量的均值
        real_flow_mean = np.mean(real_flow[:, 0])
        pred_flow_mean = np.mean(pred_flow[:, 0])
        
        print(f"流量评估指标:")
        print(f"真实流出量: {real_flow_mean:.4f}, 预测流出量: {pred_flow_mean:.4f}")
        
        # 计算真实流入量和预测流入量的均值
        real_flow_mean = np.mean(real_flow[:, 1])
        pred_flow_mean = np.mean(pred_flow[:, 1])
        
        print(f"真实流入量: {real_flow_mean:.4f}, 预测流入量: {pred_flow_mean:.4f}")
        print(f"Outflow RMSE: {outflow_rmse:.4f}")
        print(f"Outflow MAE: {outflow_mae:.4f}")
        print(f"Inflow RMSE: {inflow_rmse:.4f}") 
        print(f"Inflow MAE: {inflow_mae:.4f}")

        # 在CSV输出中也包含cluster_size
        csv_start_time = time.time()
        csv_rows = []
        for sample_idx in range(best_preds_list.shape[0]):           
            trajectory = best_preds_list[sample_idx]  # 形状 (12, 2)
            # 假设每个样本的真实时间戳为一个标量
            sample_gt_ts = gt_ts_array[sample_idx]
            
            # 获取该样本的cluster_size，如果可用
            sample_cluster_size = None
            if cluster_size_array is not None and sample_idx < len(cluster_size_array):
                sample_cluster_size = cluster_size_array[sample_idx]
            
            for timestep in range(trajectory.shape[0]):
                x, y = trajectory[timestep]
                row_data = {
                    "sample_id": sample_idx, 
                    "timestep": timestep,
                    "x": x,
                    "y": y,
                    "real_timestamp": sample_gt_ts[timestep]  # 新增字段
                }
                
                # 如果有cluster_size数据，添加到CSV
                if sample_cluster_size is not None:
                    # 直接取第一个元素作为该样本的 cluster_size
                    cluster_size_value = int(sample_cluster_size[0])
                    if cluster_size_value <= 0:  # 确保最小值为1
                        cluster_size_value = 1
                    row_data["cluster_size"] = cluster_size_value
                
                csv_rows.append(row_data)
        
        # 保存为CSV文件（每个trial一个文件）
        df = pd.DataFrame(csv_rows)
        df.to_csv(f"./csv/{scene}_pred_trial_{i_trial}.csv", index=False)
        print("预测轨迹CSV saved.")
        
        # 构建CSV数据，保存真实轨迹，也包括cluster_size
        csv_rows_gt = []
        for sample_idx in range(gt_list.shape[0]):           
            trajectory = gt_list[sample_idx]  # 形状 (12, 2)
            # 使用相同的真实时间戳信息
            sample_gt_ts = gt_ts_array[sample_idx]
            
            # 获取该样本的cluster_size，如果可用
            sample_cluster_size = None
            if cluster_size_array is not None and sample_idx < len(cluster_size_array):
                sample_cluster_size = cluster_size_array[sample_idx]
                
            for timestep in range(trajectory.shape[0]):
                x, y = trajectory[timestep]
                row_data = {
                    "sample_id": sample_idx, 
                    "timestep": timestep,
                    "x": x,
                    "y": y,
                    "real_timestamp": sample_gt_ts[timestep]  # 新增字段
                }
                
                # 如果有cluster_size数据，添加到CSV
                if sample_cluster_size is not None:
                    # 直接取第一个元素作为该样本的 cluster_size
                    cluster_size_value = int(sample_cluster_size[0])
                    if cluster_size_value <= 0:  # 确保最小值为1
                        cluster_size_value = 1
                    row_data["cluster_size"] = cluster_size_value
                
                csv_rows_gt.append(row_data)
        
        df_gt = pd.DataFrame(csv_rows_gt)
        df_gt.to_csv(f"./csv/{scene}_truth_trial_{i_trial}.csv", index=False)
        print("真实轨迹CSV saved.")
        
        # 构建CSV数据，保存观测轨迹，也包括cluster_size
        csv_rows_obs = []
        for sample_idx in range(obs_list.shape[0]):
            trajectory = obs_list[sample_idx]
            sample_obs_ts = obs_ts_array[sample_idx]
            
            # 获取该样本的cluster_size，如果可用（观测部分也使用相同的cluster_size）
            sample_cluster_size = None
            if cluster_size_array is not None and sample_idx < len(cluster_size_array):
                sample_cluster_size = cluster_size_array[sample_idx]
                
            for timestep in range(trajectory.shape[0]):
                x, y = trajectory[timestep, 0:2]  # 提取位置
                row_data = {
                    "sample_id": sample_idx,
                    "timestep": timestep,
                    "x": x,
                    "y": y,
                    "real_timestamp": sample_obs_ts[timestep]
                }
                
                # 如果有cluster_size数据，添加到CSV
                if sample_cluster_size is not None:
                    # 直接取第一个元素作为该样本的 cluster_size
                    cluster_size_value = int(sample_cluster_size[0])
                    if cluster_size_value <= 0:  # 确保最小值为1
                        cluster_size_value = 1
                    row_data["cluster_size"] = cluster_size_value
                
                csv_rows_obs.append(row_data)
        
        df_obs = pd.DataFrame(csv_rows_obs)
        df_obs.to_csv(f"./csv/{scene}_obs_trial_{i_trial}.csv", index=False)
        print("观测轨迹CSV saved.")
        csv_end_time = time.time()
        print(f"CSV处理和保存耗时: {csv_end_time-csv_start_time:.4f} s")

if __name__ == "__main__":
    st_time = time.time()
    args = parse_args()
    args.scene = args.scene if hasattr(args, 'scene') else "eth"
    args.config_file = f"./config/{args.scene}.yml"  # 添加这一行
    cfg = load_config_test(args)
    test(cfg, args)
    end_time = time.time()
    print(f"总运行时间: {end_time-st_time:.4f} s")