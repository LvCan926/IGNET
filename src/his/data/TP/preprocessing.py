import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
from copy import deepcopy

container_abcs = collections.abc


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def dict_collate(batch):
    """
    自定义的collate函数，用于将一个批次的数据整理成一个字典结构。
    该函数特别适用于多个数据字段需要被同时处理并返回为字典格式的情况。
    
    :param batch: 输入的批次数据，通常是从DataLoader中获取的一批样本。
    :return: 一个包含所有数据字段的字典
    """
    # 使用 collate 函数处理数据，得到初步的批次数据
    batch = collate(batch)
    
    # 解包从批次中提取的字段
    (
        first_history_index,
        x_t,
        y_t,
        x_st_t,
        y_st_t,
        neighbors_data_st,
        neighbors_gt_st,
        neighbors_edge_value,
        robot_traj_st_t,
        map_tuple,
        dt,
        index,
        obs_ts,
        gt_ts,
        cluster_size,
    ) = batch

    # 将提取的字段打包成一个字典，返回这个字典
    out = {
        "index": index,  # 数据项的索引
        "obs": x_t,  # 当前时刻的观察值
        "gt": y_t,   # 当前时刻的地面真值
        "obs_st": x_st_t,  # 状态下的观察值
        "gt_st": y_st_t,   # 状态下的地面真值
        "neighbors_st": neighbors_data_st,  # 邻居数据（状态）
        "neighbors_gt_st": neighbors_gt_st,  # 邻居的地面真值（状态）
        "neighbors_edge": neighbors_edge_value,  # 邻居的边缘值
        "robot_traj_st": robot_traj_st_t,  # 机器人状态下的轨迹
        "map": map_tuple,  # 地图数据
        "dt": dt,  # 时间差
        "first_history_index": first_history_index,  # 第一个历史索引
        "obs_ts": obs_ts,  # 历史轨迹的真实时间戳
        "gt_ts": gt_ts,    # 未来轨迹的真实时间戳
        "cluster_size": cluster_size,  # 集群大小
    }

    return out  # 返回整理好的字典


def collate(batch):
    """
    用于将一个批次的数据进行整理（合并），以适应模型输入的需求。
    根据输入数据的类型（列表或字典），合并方式有所不同。
    
    :param batch: 输入的批次数据（一个列表，每个元素可以是元组、字典或其他类型）
    :return: 合并后的批次数据（通常是一个张量或者字典）
    """
    if len(batch) == 0:
        return batch  # 如果批次为空，直接返回

    elem = batch[0]  # 获取批次中的第一个元素

    if elem is None:
        return None  # 如果元素为None，则返回None

    elif isinstance(elem, container_abcs.Sequence):  # 如果元素是一个序列类型（例如list）
        if len(elem) == 4:  # 假设该元素包含地图、场景点、航向角和patch大小
            # 解包批次中的数据，分别是地图、场景点、航向角和patch大小
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            
            # 如果航向角为None，则设置为None，否则转换为Tensor
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            
            # 获取裁剪后的地图
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(
                scene_map,
                scene_pts=torch.Tensor(scene_pts),
                patch_size=patch_size[0],
                rotation=heading_angle,
            )
            return map  # 返回裁剪后的地图

        # 如果不是地图数据，则转置批次数据并递归调用collate处理
        transposed = zip(*batch)
        return [
            collate(samples) if not isinstance(samples[0], tuple) else samples
            for samples in transposed
        ]

    elif isinstance(elem, container_abcs.Mapping):  # 如果元素是字典类型
        # 针对字典中的邻居数据结构进行特殊处理
        # 在多进程环境下，需要将邻居数据结构序列化（dill）
        # 否则，每个tensor都会单独放在共享内存中，导致效率低下
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return (
            dill.dumps(neighbor_dict)  # 如果是在多进程中，则序列化邻居字典
            if torch.utils.data.get_worker_info()  # 判断当前是否在多进程中
            else neighbor_dict  # 否则直接返回字典
        )

    # 对于其他类型的数据，使用默认的collate函数进行处理
    return default_collate(batch)


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    """
    获取机器人相对于节点的轨迹。
    假设机器人轨迹是相对于某个节点位置的，这个函数将其转换为相对节点的轨迹。
    
    :param env: 环境对象
    :param state: 当前状态
    :param node_traj: 节点的轨迹
    :param robot_traj: 机器人轨迹
    :param node_type: 节点类型
    :param robot_type: 机器人类型
    :return: 处理后的机器人轨迹
    """
    # 获取标准化参数
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    # 调整标准化的前两项为环境设置的注意力半径
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    # 标准化机器人轨迹
    robot_traj_st = env.standardize(
        robot_traj, state[robot_type], node_type=robot_type, mean=node_traj, std=std
    )
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)
    
    return robot_traj_st_t  # 返回标准化后的机器人轨迹


def get_node_timestep_data(
    env,
    scene,
    t,
    node,
    state,
    pred_state,
    edge_types,
    max_ht,
    max_ft,
    hyperparams,
    scene_graph=None,
    normalize_direction=False,
):
    """
    为单个批次元素预处理数据：节点在指定时间点的状态，以及邻居数据。
    该函数主要用于根据给定的时间点、节点和场景获取节点及其邻居的历史和预测数据，并进行标准化。

    :param env: 环境对象
    :param scene: 场景对象
    :param t: 当前时间步
    :param node: 当前节点对象
    :param state: 节点的当前状态
    :param pred_state: 预测状态
    :param edge_types: 所有边的类型，用于邻居预处理
    :param max_ht: 最大历史时间步数
    :param max_ft: 最大未来时间步数（预测范围）
    :param hyperparams: 模型的超参数
    :param scene_graph: 如果场景图已经为该场景和时间计算过，可以在此传递
    :param normalize_direction: 是否归一化方向
    :return: 一个元组，包含了预处理后的批次数据
    """

    # 获取时间步范围：历史时间步（t-max_ht, t）和未来时间步（t+1, t+max_ft）
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])

    # 获取当前节点的历史和预测状态数据
    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    
    # 计算历史轨迹的第一个索引
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    # 获取标准化参数
    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    
    # 获取相对于节点的标准化状态
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)

    # 如果预测的是位置，则相对于当前的位置进行标准化
    if list(pred_state[node.type].keys())[0] == "position":
        y_st = env.standardize(
            y, pred_state[node.type], node.type, mean=rel_state[0:2], std=std[0:2]
        )
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    # 将数据转换为Tensor
    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # 获取cluster_size数据（如果存在）
    cluster_size_x = None
    cluster_size_y = None
    try:
        # 尝试获取历史轨迹的cluster_size
        cluster_size_x = node.get(timestep_range_x, {"cluster_size": [""]})
        # 尝试获取预测轨迹的cluster_size
        cluster_size_y = node.get(timestep_range_y, {"cluster_size": [""]})
        
        # 将numpy数组转换为tensor
        if cluster_size_x is not None and cluster_size_y is not None:
            # 确保是整数类型
            cluster_size_x = np.round(cluster_size_x).astype(np.int32)
            cluster_size_y = np.round(cluster_size_y).astype(np.int32)
            
            # 将numpy数组转换为tensor
            cluster_size_x = torch.tensor(cluster_size_x, dtype=torch.int32)
            cluster_size_y = torch.tensor(cluster_size_y, dtype=torch.int32)
            
            # 拼接历史和未来的cluster_size
            cluster_size = torch.cat([cluster_size_x, cluster_size_y], dim=0)
            
            # 确保最小值为1
            cluster_size = torch.where(cluster_size <= 0, torch.ones_like(cluster_size), cluster_size)
        else:
            cluster_size = None
    except (KeyError, ValueError):
        # 如果获取失败，设置为None
        cluster_size = None

    # 处理邻居数据
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams["edge_encoding"]:
        # 获取场景图数据
        scene_graph = (
            scene.get_scene_graph(
                t,
                env.attention_radius,
                hyperparams["edge_addition_filter"],
                hyperparams["edge_removal_filter"],
            )
            if scene_graph is None
            else scene_graph
        )

        neighbors_data_st = dict()
        neighbors_gt_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            neighbors_gt_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams["dynamic_edges"] == "yes":
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(
                    scene_graph.get_edge_scaling(node), dtype=torch.float
                )
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(
                    timestep_range_x, state[connected_node.type], padding=0.0
                )
                neighbor_gt_np = connected_node.get(
                    timestep_range_y, pred_state[connected_node.type], padding=0.0
                )

                # 标准化邻居状态
                _, std = env.get_standardize_params(
                    state[connected_node.type], node_type=connected_node.type
                )
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(
                    neighbor_state_np,
                    state[connected_node.type],
                    node_type=connected_node.type,
                    mean=rel_state,
                    std=std,
                )
                
                # 标准化邻居的预测状态
                _, std = env.get_standardize_params(
                    pred_state[connected_node.type], node_type=connected_node.type
                )
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_gt_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_gt_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_gt_np_st = env.standardize(
                    neighbor_gt_np,
                    pred_state[connected_node.type],
                    node_type=connected_node.type,
                    mean=rel_state,
                )

                # 转换为Tensor
                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbor_gt = torch.tensor(neighbor_gt_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)
                neighbors_gt_st[edge_type].append(neighbor_gt)

    # 机器人轨迹
    robot_traj_st_t = None
    timestep_range_r = np.array([t, t + max_ft])
    if hyperparams["incl_robot_node"]:
        x_node = node.get(timestep_range_r, state[node.type])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)
        robot_traj_st_t = get_relative_robot_traj(
            env, state, x_node, robot_traj, node.type, robot_type
        )

    # Map
    map_tuple = None
    if hyperparams["use_map_encoding"]:
        if node.type in hyperparams["map_encoder"]:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams["map_encoder"][node.type]
            if "heading_state_index" in me_hyp:
                heading_state_index = me_hyp["heading_state_index"]
                # We have to rotate the map in the opposit direction of the agent to match them
                if (
                    type(heading_state_index) is list
                ):  # infer from velocity or heading vector
                    heading_angle = (
                        -np.arctan2(
                            x[-1, heading_state_index[1]], x[-1, heading_state_index[0]]
                        )
                        * 180
                        / np.pi
                    )
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]

            patch_size = hyperparams["map_encoder"][node.type]["patch_size"]
            map_tuple = (scene_map, map_point, heading_angle, patch_size)

    # 如果需要归一化方向，则进行旋转
    if normalize_direction:
        # rotate
        x_t_rotate = torch.zeros_like(x_t)  # (8,6)
        y_t_rotate = torch.zeros_like(y_t)  # (12,2) vel
        x_st_t_rotate = torch.zeros_like(x_st_t)  # (8,6)
        y_st_t_rotate = torch.zeros_like(y_st_t)  # (12,2) vel
        neighbors_gt_st_rotate = deepcopy(neighbors_gt_st)

        current_vel = x_t[-1, 2:4]
        rotate_angle = -torch.arctan2(current_vel[1], current_vel[0])
        rotate_matrix = torch.tensor([torch.cos(rotate_angle), -torch.sin(rotate_angle),
                                        torch.sin(rotate_angle), torch.cos(rotate_angle)]).reshape(2,2).unsqueeze(0) # (1,2,2)
        
        x_t_rotate[:,0:2] = torch.bmm(rotate_matrix.repeat(8,1,1), (x_t[:,0:2] - x_t[-1,0:2]).unsqueeze(-1)).squeeze(-1) + x_t[-1,0:2]
        x_t_rotate[:,2:4] = torch.bmm(rotate_matrix.repeat(8,1,1), x_t[:,2:4].unsqueeze(-1)).squeeze(-1)
        x_t_rotate[:,4:6] = torch.bmm(rotate_matrix.repeat(8,1,1), x_t[:,4:6].unsqueeze(-1)).squeeze(-1)
        y_t_rotate = torch.bmm(rotate_matrix.repeat(12,1,1), y_t.unsqueeze(-1)).squeeze(-1)

        x_st_t_rotate[:,0:2] = torch.bmm(rotate_matrix.repeat(8,1,1), (x_st_t[:,0:2] - x_st_t[-1,0:2]).unsqueeze(-1)).squeeze(-1) + x_st_t[-1,0:2]
        x_st_t_rotate[:,2:4] = torch.bmm(rotate_matrix.repeat(8,1,1), x_st_t[:,2:4].unsqueeze(-1)).squeeze(-1)
        x_st_t_rotate[:,4:6] = torch.bmm(rotate_matrix.repeat(8,1,1), x_st_t[:,4:6].unsqueeze(-1)).squeeze(-1)
        y_st_t_rotate = torch.bmm(rotate_matrix.repeat(12,1,1), y_st_t.unsqueeze(-1)).squeeze(-1)

        if neighbors_gt_st[edge_type] is not None:
            for i, nb_fut in enumerate(neighbors_gt_st[edge_type]):
                neighbors_gt_st_rotate[edge_type][i] = torch.bmm(rotate_matrix.repeat(12,1,1), nb_fut.unsqueeze(-1)).squeeze(-1)

        x_t = x_t_rotate
        y_t = y_t_rotate
        x_st_t = x_st_t_rotate
        y_st_t = y_st_t_rotate
        neighbors_gt_st = neighbors_gt_st_rotate
    
    # 获取历史时间戳和未来时间戳
    obs_ts = np.arange(t - max_ht, t + 1)       # 历史时刻
    gt_ts = np.arange(t + 1, t + max_ft + 1)  # 未来时刻

    return (
        first_history_index,
        x_t,
        y_t,
        x_st_t,
        y_st_t,
        neighbors_data_st,
        neighbors_gt_st,
        neighbors_edge_value,
        robot_traj_st_t,
        map_tuple,
        scene.dt,
        (scene.name, t, "/".join([node.type.name, node.id])),
        obs_ts,  # 历史轨迹的时间戳数组
        gt_ts,   # 未来（ground truth/预测）的时间戳数组
        cluster_size,  # 新增返回cluster_size数据
    )

def get_timesteps_data(env, scene, t, node_type, state, pred_state,
                       edge_types, min_ht, max_ht, min_ft, max_ft, hyperparams):
    """
    为某个场景的某一时刻，收集所有节点的数据（包含节点本身、邻居、地图等）。
    
    :param env: 环境对象（包含标准化参数、注意力半径等）
    :param scene: 当前处理的场景
    :param t: 当前时间步
    :param node_type: 要处理的节点类型（例如行人、车辆）
    :param state: 状态的特征结构定义
    :param pred_state: 预测目标的结构定义（通常是位置）
    :param edge_types: 要处理的邻居边类型（如车-车、车-人等）
    :param min_ht: 节点所需的最小历史步数
    :param max_ht: 节点使用的最大历史步数
    :param min_ft: 节点所需的最小未来步数
    :param max_ft: 节点使用的最大未来步数
    :param hyperparams: 超参数配置
    :return: 一个元组：(批次数据, 节点列表, 时间步列表)
    """
    
    # 获取所有符合要求的节点：当前时刻有足够历史和未来轨迹的数据
    nodes_per_ts = scene.present_nodes(t,
                                       type=node_type,
                                       min_history_timesteps=min_ht,
                                       min_future_timesteps=max_ft,
                                       return_robot=not hyperparams['incl_robot_node']) # 如果模型包含机器人，则不返回机器人
    batch = list()          # 存放每个节点的数据
    nodes = list()          # 存放每个节点对象
    out_timesteps = list()  # 存放每个节点对应的时间步
    
    for timestep in nodes_per_ts.keys():
        # 获取当前时间步的场景图
        scene_graph = scene.get_scene_graph(timestep,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter'])
        # 获取该时间步的所有节点
        present_nodes = nodes_per_ts[timestep]
        for node in present_nodes:
            nodes.append(node)
            out_timesteps.append(timestep)
            # 获取该节点在该时间步的完整数据（历史/未来轨迹、邻居、地图、机器人等）
            batch.append(get_node_timestep_data(env, scene, timestep, node, state, pred_state,
                                                edge_types, max_ht, max_ft, hyperparams,
                                                scene_graph=scene_graph))
    if len(out_timesteps) == 0:
        return None
    
    # 返回拼接后的数据、节点列表、对应时间步列表
    return collate(batch), nodes, out_timesteps


def data_dict_to_next_step(data_dict, time_step):
    """
    将数据字典滚动到下一个时间步，把前一个时间段的 ground truth 合并到 observation 里。
    
    :param data_dict: 原始数据字典，格式为 dict_collate 输出结构
    :param time_step: 要前移的时间步数（通常为1）
    :return: 新的数据字典，obs 和 obs_st 中包含了更多“真实”未来信息
    """
    
    data_dict_ = deepcopy(data_dict)    # 拷贝原始数据，避免修改原对象
    obs = data_dict["obs"]              # 当前观测（历史）数据
    obs_st = data_dict["obs_st"]        # 标准化的历史数据
    gt = data_dict["gt"]                # 未来 ground truth 数据
    gt_st = data_dict["gt_st"]          # 标准化的 ground truth
    
    neighbors_st = data_dict["neighbors_st"]
    neighbors_gt_st = data_dict["neighbors_gt_st"]
    
    bs, n, d_o = obs.shape  # bs: 批次大小, n: 历史步数, d_o: 状态维度
    _, _, d_g = gt.shape    # d_g: 预测目标的维度
    
    # 将历史数据“右移”：用预测数据的前几步补齐末尾
    data_dict_["obs"][:, :-time_step] = obs[:, time_step:]
    data_dict_["obs"][:, -time_step:, :d_g] = gt[:, :time_step]
    
    # 标准化后的同样处理
    data_dict_["obs_st"][:, :-time_step] = obs_st[:, time_step:]
    data_dict_["obs_st"][:, -time_step:, :d_g] = gt_st[:, :time_step]
    
    # ==== 更新 cluster_size 字段（如果存在）====
    if "cluster_size" in data_dict and data_dict["cluster_size"] is not None:
        cluster_size = data_dict["cluster_size"]
        # 确保cluster_size是正确的形状
        if len(cluster_size.shape) >= 2:
            # 确保是整数类型
            if not torch.is_integral(cluster_size):
                cluster_size = torch.round(cluster_size).to(torch.int32)
                # 确保最小值为1
                cluster_size = torch.where(cluster_size <= 0, torch.ones_like(cluster_size), cluster_size)
                data_dict_["cluster_size"] = cluster_size
            
            # 更新cluster_size，与obs和gt保持一致
            data_dict_["cluster_size"][:, :-time_step] = cluster_size[:, time_step:]
            data_dict_["cluster_size"][:, -time_step:] = cluster_size[:, :time_step]
    
    """
    neighbors_st_ = []
    for b in range(bs):
        import pdb;pdb.set_trace()
        neighbors_st_.append(torch.cat([neighbors_st[b], neighbors_gt_st[b]], dim=1)[:, time_step:n+time_step])
    """
    data_dict_["neighbors_st"] = neighbors_st

    return data_dict_
