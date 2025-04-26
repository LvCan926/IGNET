"""
多模态运动环境建模核心模块

功能概述：
1. Environment 类 - 多类型智能体交互环境容器
   - 管理异构节点类型(行人/车辆等)的交互关系与注意力范围
   - 提供标准化/反标准化数据管道(支持动态参数加载与缓存)
   - 场景概率采样控制(支持非均匀场景分布训练)

2. 核心能力：
   - 自动构建节点类型间交互拓扑(基于笛卡尔积)
   - 多维度数据标准化(处理NaN值保持数据完整性)
   - 场景资源动态加载与权重分配

典型应用场景：
- 自动驾驶轨迹预测模型的训练环境
- 社交机器人行为模拟的多智能体系统
- 城市交通流建模中的异构参与者交互
"""


import orjson
import numpy as np
from itertools import product
from .node_type import NodeTypeEnum


class Environment(object):
    def __init__(
        self,
        node_type_list,
        standardization,
        scenes=None,
        attention_radius=None,
        robot_type=None,
    ):
        self.scenes = scenes
        self.node_type_list = node_type_list
        self.attention_radius = attention_radius
        self.NodeType = NodeTypeEnum(node_type_list)
        self.robot_type = robot_type

        self.standardization = standardization
        self.standardize_param_memo = dict()

        self._scenes_resample_prop = None

        self.gt_dist = None  # for simulated data

    def get_edge_types(self):
        return list(product(self.NodeType, repeat=2))

    def get_standardize_params(self, state, node_type):
        # memo_key = (orjson.dumps(state), node_type)
        # if memo_key in self.standardize_param_memo:
        #    return self.standardize_param_memo[memo_key]

        standardize_mean_list = list()
        standardize_std_list = list()
        for entity, dims in state.items():
            for dim in dims:
                standardize_mean_list.append(
                    self.standardization[node_type][entity][dim]["mean"]
                )
                standardize_std_list.append(
                    self.standardization[node_type][entity][dim]["std"]
                )
        standardize_mean = np.stack(standardize_mean_list)
        standardize_std = np.stack(standardize_std_list)

        # self.standardize_param_memo[memo_key] = (standardize_mean, standardize_std)
        return standardize_mean, standardize_std

    def standardize(self, array, state, node_type, mean=None, std=None):
        if mean is None and std is None:
            mean, std = self.get_standardize_params(state, node_type)
        elif mean is None and std is not None:
            mean, _ = self.get_standardize_params(state, node_type)
        elif mean is not None and std is None:
            _, std = self.get_standardize_params(state, node_type)
        return np.where(np.isnan(array), np.array(np.nan), (array - mean) / std)

    def unstandardize(self, array, state, node_type, mean=None, std=None):
        if mean is None and std is None:
            mean, std = self.get_standardize_params(state, node_type)
        elif mean is None and std is not None:
            mean, _ = self.get_standardize_params(state, node_type)
        elif mean is not None and std is None:
            _, std = self.get_standardize_params(state, node_type)
        return array * std + mean

    @property
    def scenes_resample_prop(self):
        if self._scenes_resample_prop is None:
            self._scenes_resample_prop = np.array(
                [scene.resample_prob for scene in self.scenes]
            )
            self._scenes_resample_prop = self._scenes_resample_prop / np.sum(
                self._scenes_resample_prop
            )
        return self._scenes_resample_prop
