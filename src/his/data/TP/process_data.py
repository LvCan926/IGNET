"""
轨迹预测数据处理脚本
功能：处理ETH/UCY和Stanford Drone数据集，生成标准化格式并增强数据
"""

import sys

sys.path.append("./src")  # 添加自定义模块路径
import os
import numpy as np
import pandas as pd
import dill
import pickle

from data.TP.environment import Environment, Scene, Node, derivative_of

np.random.seed(123)

desired_max_time = 100    # 最大时间步长
pred_indices = [2, 3]     # 预测的坐标索引（可能是x,y）
state_dim = 6             # 状态维度（位置x,y + 速度x,y + 加速度x,y）
frame_diff = 10           # 原始帧间隔
desired_frame_diff = 1    # 目标帧间隔
dt = 0.4                  # 时间步长（秒）

standardization = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}


def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def augment_scene(scene, angle):
    """场景数据增强：旋转指定角度生成新场景
    Args:
        scene: 原始场景对象
        angle: 旋转角度（度）
    Returns:
        scene_aug: 增强后的新场景
    """

    def rotate_pc(pc, alpha):
        """坐标旋转函数"""
        M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])  # 旋转矩阵
        return M @ pc

    # 创建多级列索引（位置、速度、加速度 x/y）
    data_columns = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    # 创建新场景对象
    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180  # 角度转弧度

    # 处理场景中的每个行人节点
    for node in scene.nodes:
        # 获取原始位置数据
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        # 坐标旋转
        x, y = rotate_pc(np.array([x, y]), alpha)

        # 计算导数得到速度和加速度
        vx = derivative_of(x, scene.dt)  # x方向速度
        vy = derivative_of(y, scene.dt)  # y方向速度
        ax = derivative_of(vx, scene.dt)  # x方向加速度
        ay = derivative_of(vy, scene.dt)  # y方向加速度

        # 构建数据字典
        data_dict = {
            ("position", "x"): x,
            ("position", "y"): y,
            ("velocity", "x"): vx,
            ("velocity", "y"): vy,
            ("acceleration", "x"): ax,
            ("acceleration", "y"): ay,
        }

        # 创建DataFrame并生成新节点
        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(
            node_type=node.type,
            node_id=node.id,
            data=node_data,
            first_timestep=node.first_timestep,
        )

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    """随机选择增强后的场景"""
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug

# 初始化计数器（可能用于调试）
nl = 0
l = 0

# 路径配置
raw_path = "./src/data/TP/raw_data"
data_folder_name = "./src/data/TP/processed_data/"

maybe_makedirs(data_folder_name)  # 创建输出目录

# 创建多级列索引
data_columns = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration"], ["x", "y"]]
)


# ======================= 处理ETH-UCY数据集 =======================
for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'BJTaxi']:  # 遍历所有子数据集
    if desired_source != "BJTaxi":
        continue
    for data_class in ['train', 'val', 'test']:
        if data_class != "test":
            continue
        
        # 初始化环境对象
        env = Environment(
            node_type_list=["PEDESTRIAN"], standardization=standardization
        )
        # 设置注意力半径（行人之间3米范围内需要关注）
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius

        scenes = []  # 场景列表
        data_dict_path = os.path.join(
            data_folder_name, "_".join([desired_source, data_class]) + ".pkl"
        )

        # 遍历原始数据目录
        for subdir, dirs, files in os.walk(
            os.path.join(raw_path, desired_source, data_class)
        ):
            for file in files:
                if file.endswith(".txt"):  # 处理文本数据文件
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print("At", full_data_path)

                    # 读取原始数据
                    data = pd.read_csv(
                        full_data_path, sep="\t", index_col=False
                    )
                    
                    type = ""
                    
                    # 检查并适应新的数据格式（包含size列）
                    if 'cluster_id' in data.columns and 'time' in data.columns and 'lon' in data.columns and 'lat' in data.columns:
                        # 重命名列以匹配处理逻辑
                        data.rename(columns={
                            'cluster_id': 'track_id',
                            'time': 'frame_id',
                            'lon': 'pos_x',
                            'lat': 'pos_y'
                        }, inplace=True)
                        
                        # 保存size列，如果存在
                        if 'cluster_size' in data.columns:
                            data['cluster_size'] = data['cluster_size']  # 保留size列
                            
                        type = "BJTaxi"
                    else :
                        data.columns = ["frame_id", "track_id", "pos_x", "pos_y"]
                        type = "ETH"
                        
                    print(type)
                    
                    # 数据类型转换
                    data["frame_id"] = pd.to_numeric(
                        data["frame_id"], downcast="integer"
                    )
                    data["track_id"] = pd.to_numeric(
                        data["track_id"], downcast="integer"
                    )

                    if desired_source != "BJTaxi":
                        data["frame_id"] = data["frame_id"] // 10  # 降采样到0.4秒间隔（原始10FPS）

                    # data['frame_id'] -= data['frame_id'].min()

                    
                    data["node_type"] = "PEDESTRIAN"
                    data["node_id"] = data["track_id"].astype(str)

                    # 由于数据已经按entity_id分组和time排序，不需要再排序
                    if type == "ETH":
                        data.sort_values("frame_id", inplace=True)

                    # ETH测试集特殊处理（坐标缩放）
                    if desired_source == "eth" and data_class == "test":
                        data["pos_x"] = data["pos_x"] * 0.6
                        data["pos_y"] = data["pos_y"] * 0.6

                    # if data_class == "train":
                    #     #data_gauss = data.copy(deep=True)
                    #     data['pos_x'] = data['pos_x'] + 2 * np.random.normal(0,1)
                    #     data['pos_y'] = data['pos_y'] + 2 * np.random.normal(0,1)

                    # data = pd.concat([data, data_gauss])

                    # data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    # data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    # 创建场景对象
                    max_timesteps = data["frame_id"].max() + 1

                    scene = Scene(
                        timesteps=max_timesteps,
                        dt=dt,
                        name=file.rstrip(".txt"),
                        aug_func=augment if data_class == "train" else None,  # 仅训练集做增强
                    )

                    # 处理每个行人轨迹
                    for node_id in pd.unique(data["node_id"]):

                        node_df = data[data["node_id"] == node_id]

                        node_values = node_df[["pos_x", "pos_y"]].values

                        if node_values.shape[0] < 2:  # 跳过单点轨迹
                            continue

                        new_first_idx = node_df["frame_id"].iloc[0]

                        # 计算运动学参数
                        x = node_values[:, 0]
                        y = node_values[:, 1]
                        vx = derivative_of(x, scene.dt)  # 数值微分计算速度
                        vy = derivative_of(y, scene.dt)
                        ax = derivative_of(vx, scene.dt)  # 计算加速度
                        ay = derivative_of(vy, scene.dt)

                        # 构建数据字典
                        data_dict = {
                            ("position", "x"): x,
                            ("position", "y"): y,
                            ("velocity", "x"): vx,
                            ("velocity", "y"): vy,
                            ("acceleration", "x"): ax,
                            ("acceleration", "y"): ay,
                        }
                        
                        # 如果存在size列，添加到数据字典
                        if 'cluster_size' in node_df.columns:
                            data_dict[("cluster_size", "")] = node_df["cluster_size"].values

                        # 创建节点对象
                        node_data = pd.DataFrame(data_dict)
                        
                        # 确保数据列的正确顺序
                        if 'cluster_size' in node_df.columns:
                            # 更新data_columns包含size
                            size_columns = pd.MultiIndex.from_product([["cluster_size"], [""]])
                            combined_columns = data_columns.append(size_columns)
                            node_data = node_data.reindex(columns=combined_columns)
                        else:
                            node_data = node_data.reindex(columns=data_columns)
                            
                        node = Node(
                            node_type=env.NodeType.PEDESTRIAN,
                            node_id=node_id,
                            data=node_data,
                        )
                        node.first_timestep = new_first_idx  # 设置起始时间步

                        scene.nodes.append(node)
                        
                    # 数据增强（训练集每15度旋转生成新场景）
                    if data_class == "train":
                        scene.augmented = list()
                        angles = np.arange(0, 360, 15) if data_class == "train" else [0]  # 生成15度间隔的所有角度
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    print(scene)  # 打印场景信息
                    scenes.append(scene)
        print(f"Processed {len(scenes):.2f} scene for data class {data_class}")

        env.scenes = scenes

        # 保存处理后的数据
        if len(scenes) > 0:
            with open(data_dict_path, "wb") as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)  # 高效序列化
                pass

