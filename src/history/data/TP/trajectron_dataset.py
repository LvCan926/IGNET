import torch
from torch.utils import data
import numpy as np
from data.TP.preprocessing import get_node_timestep_data
from copy import deepcopy
import warnings
import os
import pickle

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed, clustering functionality will not be available. Install with: pip install scikit-learn")


hypers = {
    "state_p": {"PEDESTRIAN": {"position": ["x", "y"]}},
    "state_v": {"PEDESTRIAN": {"velocity": ["x", "y"]}},
    "state_a": {"PEDESTRIAN": {"acceleration": ["x", "y"]}},
    "state_pva": {
        "PEDESTRIAN": {
            "position": ["x", "y"],
            "velocity": ["x", "y"],
            "acceleration": ["x", "y"],
        }
    },
    "batch_size": 256,
    "grad_clip": 1.0,
    "learning_rate_style": "exp",
    "min_learning_rate": 1e-05,
    "learning_decay_rate": 0.9999,
    "prediction_horizon": 12,
    "minimum_history_length": 1,
    "maximum_history_length": 7,
    "map_encoder": {
        "PEDESTRIAN": {
            "heading_state_index": 6,
            "patch_size": [50, 10, 50, 90],
            "map_channels": 3,
            "hidden_channels": [10, 20, 10, 1],
            "output_size": 32,
            "masks": [5, 5, 5, 5],
            "strides": [1, 1, 1, 1],
            "dropout": 0.5,
        }
    },
    "k": 1,
    "k_eval": 25,
    "kl_min": 0.07,
    "kl_weight": 100.0,
    "kl_weight_start": 0,
    "kl_decay_rate": 0.99995,
    "kl_crossover": 400,
    "kl_sigmoid_divisor": 4,
    "rnn_kwargs": {"dropout_keep_prob": 0.75},
    "MLP_dropout_keep_prob": 0.9,
    "enc_rnn_dim_edge": 128,
    "enc_rnn_dim_edge_influence": 128,
    "enc_rnn_dim_history": 128,
    "enc_rnn_dim_future": 128,
    "dec_rnn_dim": 128,
    "q_z_xy_MLP_dims": None,
    "p_z_x_MLP_dims": 32,
    "GMM_components": 1,
    "log_p_yt_xz_max": 6,
    "N": 1,
    "tau_init": 2.0,
    "tau_final": 0.05,
    "tau_decay_rate": 0.997,
    "use_z_logit_clipping": True,
    "z_logit_clip_start": 0.05,
    "z_logit_clip_final": 5.0,
    "z_logit_clip_crossover": 300,
    "z_logit_clip_divisor": 5,
    "dynamic": {
        "PEDESTRIAN": {"name": "SingleIntegrator", "distribution": False, "limits": {}}
    },
    "pred_state": {"PEDESTRIAN": {"velocity": ["x", "y"]}},
    "log_histograms": False,
    "dynamic_edges": "yes",
    "edge_state_combine_method": "sum",
    "edge_influence_combine_method": "attention",
    "edge_addition_filter": [0.25, 0.5, 0.75, 1.0],
    "edge_removal_filter": [1.0, 0.0],
    "offline_scene_graph": "yes",
    "incl_robot_node": False,
    "node_freq_mult_train": False,
    "node_freq_mult_eval": False,
    "scene_freq_mult_train": False,
    "scene_freq_mult_eval": False,
    "scene_freq_mult_viz": False,
    "edge_encoding": True,
    "use_map_encoding": False,
    "augment": True,
    "override_attention_radius": [],
    "learning_rate": 0.01,
    "npl_rate": 0.8,
    "K": 80,
    "tao": 0.4,
}


class EnvironmentDataset(object):
    def __init__(
        self,
        env,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        augment=False,
        normalize_direction=False,
        **kwargs
    ):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams["maximum_history_length"]
        self.max_ft = kwargs["min_future_timesteps"]
        self.node_type_datasets = list()
        self._augment = augment
        for node_type in env.NodeType:
            if node_type not in hyperparams["pred_state"]:
                continue
            self.node_type_datasets.append(
                NodeTypeDataset(
                    env,
                    node_type,
                    state,
                    pred_state,
                    node_freq_mult,
                    scene_freq_mult,
                    hyperparams,
                    augment=augment,
                    normalize_direction=normalize_direction,
                    **kwargs
                )
            )

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):
    def __init__(
        self,
        env,
        node_type,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        augment=False,
        normalize_direction=False,
        **kwargs
    ):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams["maximum_history_length"]
        self.max_ft = kwargs["min_future_timesteps"]

        self.augment = augment
        self.normalize_direction = normalize_direction

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [
            edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type
        ]

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(
                np.arange(0, scene.timesteps), type=self.node_type, **kwargs
            )
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += (
                        [(scene, t, node)]
                        * (scene.frequency_multiplier if scene_freq_mult else 1)
                        * (node.frequency_multiplier if node_freq_mult else 1)
                    )

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()  # randomly choose a angle, on-the-fly augmenting
            node = scene.get_node_by_id(node.id)
        
        return get_node_timestep_data(
            self.env,
            scene,
            t,
            node,
            self.state,
            self.pred_state,
            self.edge_types,
            self.max_ht,
            self.max_ft,
            self.hyperparams,
            normalize_direction=self.normalize_direction,
        )


class ClusteredNodeTypeDataset(NodeTypeDataset):
    """
    将每个场景和时间步的节点进行聚类的数据集
    每个聚类选择一个代表性节点，并记录该节点代表的数量
    """
    def __init__(
        self,
        env,
        node_type,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        augment=False,
        normalize_direction=False,
        cluster_count=10,
        dataset_name_for_cache: str = None,
        cache_dir: str = None,
        **kwargs
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for clustering. Install with: pip install scikit-learn")
            
        super().__init__(
            env, node_type, state, pred_state, node_freq_mult, 
            scene_freq_mult, hyperparams, augment, normalize_direction, **kwargs
        )
        self.cluster_count = cluster_count
        self.dataset_name_for_cache = dataset_name_for_cache
        self.cache_dir = cache_dir

        loaded_from_cache = False
        if self.cache_dir and self.dataset_name_for_cache:
            self.clustered_index, self.cluster_info = self._load_cluster_data()
            
            # print(f"self.clustered_index: {self.clustered_index}")
            # print(f"self.cluster_info: {self.cluster_info}")
            
            if self.clustered_index is not None and self.cluster_info is not None:
                loaded_from_cache = True
                self.len = len(self.clustered_index)
                print(f"Loaded pre-clustered data for {self.node_type.name} from cache. Cluster count: {self.cluster_count}. Points: {self.len}")

        if not loaded_from_cache:
            print("需计算聚类")
            
            # 按场景和时间步组织索引
            self.scene_timestep_index = self._organize_indices()
            # 创建聚类后的新索引
            self.clustered_index, self.cluster_info = self._compute_clusters_and_info()
            self.len = len(self.clustered_index)
            print(f"Computed clusters for {self.node_type.name}. Original points: {len(self.index)}, Clustered points: {self.len}")

            if self.cache_dir and self.dataset_name_for_cache:
                self._save_cluster_data()
                print(f"Saved clustered data for {self.node_type.name} to cache.")
        
    def _get_cache_filenames(self):
        print(f"node_type.name: {self.node_type.name}")
        
        if not self.cache_dir or not self.dataset_name_for_cache or not hasattr(self.node_type, 'name'):
            return None, None
        
        base_filename = f"{self.dataset_name_for_cache}_{self.node_type.name}_k{self.cluster_count}"
        index_filename = os.path.join(self.cache_dir, f"{base_filename}_indices.pkl")
        info_filename = os.path.join(self.cache_dir, f"{base_filename}_info.pkl")
        return index_filename, info_filename

    def _load_cluster_data(self):
        index_filename, info_filename = self._get_cache_filenames()
        
        if index_filename is None or not os.path.exists(index_filename) or not os.path.exists(info_filename):
            return None, None
        
        try:
            with open(index_filename, 'rb') as f:
                # Loads list of (scene_name, t, node_id)
                loaded_clustered_node_ids = pickle.load(f)
            with open(info_filename, 'rb') as f:
                # Loads dict of (scene_name, t, node_id) -> cluster_size
                loaded_cluster_info_with_names = pickle.load(f)

            # Reconstruct self.clustered_index from names/IDs
            node_map = {(s.name, ts, n.id): (s, ts, n) for s, ts, n in self.index}
            
            reconstructed_clustered_index = []
            for s_name, t_val, n_id in loaded_clustered_node_ids:
                key = (s_name, t_val, n_id)
                if key in node_map:
                    reconstructed_clustered_index.append(node_map[key])
                else:
                    warnings.warn(f"Node ID {n_id} at scene_name '{s_name}', time {t_val} not found in current environment index. Skipping.")
            
            # Convert keys of cluster_info from (scene_name, t, node_id) to (scene_obj, t, node_id)
            scene_name_to_obj_map = {s.name: s for s in self.env.scenes}
            
            final_cluster_info = {}
            for (s_name, t_val, n_id), size in loaded_cluster_info_with_names.items():
                scene_obj = scene_name_to_obj_map.get(s_name)
                if scene_obj:
                    final_cluster_info[(scene_obj, t_val, n_id)] = size
                else:
                    warnings.warn(f"Scene name '{s_name}' from cluster info cache not found in environment. Skipping info for node {n_id} at time {t_val}.")
            
            return reconstructed_clustered_index, final_cluster_info
        except Exception as e:
            warnings.warn(f"Error loading cached cluster data: {e}")
            return None, None

    def _save_cluster_data(self):
        index_filename, info_filename = self._get_cache_filenames()
        if index_filename is None:
            warnings.warn("Cache directory or dataset name not configured. Skipping saving cluster data.")
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Convert clustered_index (list of (scene_obj, t_val, node_obj)) to list of (scene_name, t_val, node_id)
        serializable_clustered_index = [(s.name, t_val, n.id) for s, t_val, n in self.clustered_index]
        
        # Convert self.cluster_info keys from (scene_obj, t_val, node_id_val) to (scene_name, t_val, node_id_val)
        serializable_cluster_info = {}
        for (scene_obj, t_val, node_id_val), size in self.cluster_info.items():
            serializable_cluster_info[(scene_obj.name, t_val, node_id_val)] = size

        try:
            with open(index_filename, 'wb') as f:
                pickle.dump(serializable_clustered_index, f)
            with open(info_filename, 'wb') as f:
                pickle.dump(serializable_cluster_info, f)
        except Exception as e:
            warnings.warn(f"Error saving cluster data to cache: {e}")
        
    def _organize_indices(self):
        """将索引按场景和时间步组织"""
        scene_timestep_dict = {}
        for idx, (scene, t, node) in enumerate(self.index):
            key = (scene, t)
            if key not in scene_timestep_dict:
                scene_timestep_dict[key] = []
            scene_timestep_dict[key].append((idx, node))
        return scene_timestep_dict
    
    def _compute_clusters_and_info(self):
        """为每个场景和时间步创建聚类后的索引"""
        clustered_index = []
        cluster_info = {}  # 存储聚类信息, key: (scene_obj, t, node.id)
        
        for (scene, t), node_indices in self.scene_timestep_index.items():
            # 如果节点数量少于聚类数，直接使用所有节点
            if len(node_indices) <= self.cluster_count:
                for idx, node in node_indices:
                    clustered_index.append((scene, t, node))
                    # 每个节点自己形成一个簇
                    cluster_info[(scene, t, node.id)] = 1
                continue
                
            # 获取所有节点的历史轨迹用于聚类
            history_features = []
            original_indices = []
            all_nodes = []
            
            for idx, node in node_indices:
                # 获取节点数据但不进行增强
                orig_augment = self.augment
                self.augment = False
                node_data = super().__getitem__(idx)
                self.augment = orig_augment
                
                # 提取历史轨迹特征用于聚类
                x_t = node_data[1]  # 历史轨迹
                history_features.append(x_t.flatten().numpy())
                original_indices.append(idx)
                all_nodes.append(node)
            
            # 执行K-means聚类
            kmeans = KMeans(n_clusters=min(self.cluster_count, len(node_indices)), random_state=42)
            cluster_labels = kmeans.fit_predict(np.array(history_features))
            
            # 为每个簇选择最接近中心的样本
            cluster_centers = kmeans.cluster_centers_
            closest_indices, _ = pairwise_distances_argmin_min(cluster_centers, np.array(history_features))
            
            # 计算每个簇的大小（成员数量）
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            label_to_count = dict(zip(unique_labels, counts))
            
            # 添加到新索引
            for center_idx in closest_indices:
                node = all_nodes[center_idx]
                cluster_id = cluster_labels[center_idx]
                cluster_size = label_to_count[cluster_id]
                
                clustered_index.append((scene, t, node))
                cluster_info[(scene, t, node.id)] = int(cluster_size)
        
        return clustered_index, cluster_info
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        original_scene, original_t, original_node = self.clustered_index[i]
        
        current_scene_for_data = original_scene
        current_node_for_data = original_node
        if self.augment:
            current_scene_for_data = original_scene.augment()
            current_node_for_data = current_scene_for_data.get_node_by_id(original_node.id)
            if current_node_for_data is None: # Safety check if node not found in augmented scene
                warnings.warn(f"Node {original_node.id} not found in augmented scene {current_scene_for_data.name}. Using original node and scene for data.")
                current_node_for_data = original_node
                current_scene_for_data = original_scene
        
        # Get node data using the potentially augmented scene and node, but original timestep
        node_data = get_node_timestep_data(
            self.env, current_scene_for_data, original_t, current_node_for_data, 
            self.state, self.pred_state, 
            self.edge_types, self.max_ht, self.max_ft, self.hyperparams,
            normalize_direction=self.normalize_direction
        )
        
        # Get cluster size using the original scene, t, and node.id for the lookup
        # self.cluster_info is now consistently keyed by (original_scene_obj, t, node_id)
        key_for_cluster_info = (original_scene, original_t, original_node.id)
        cluster_size = self.cluster_info.get(key_for_cluster_info, 1) # Default to 1 if not found
        
        # 创建cluster_size张量
        x_len = node_data[1].shape[0]  # 历史轨迹长度
        y_len = node_data[2].shape[0]  # 未来轨迹长度
        cluster_size_tensor = torch.full((x_len + y_len,), cluster_size, dtype=torch.int32)
        
        # 将cluster_size添加到node_data
        # 确保node_data的最后一个位置是None或cluster_size
        if node_data[-1] is None:
            modified_data = node_data[:-1] + (cluster_size_tensor,)
        else:
            modified_data = node_data + (cluster_size_tensor,)
        
        return modified_data


class ClusteredEnvironmentDataset(EnvironmentDataset):
    """
    使用聚类方法的环境数据集，对每个场景和时间步的节点进行聚类
    """
    def __init__(
        self,
        env,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        cluster_count=10,
        augment=False,
        normalize_direction=False,
        dataset_name_for_cache: str = None,
        cache_dir: str = None,
        **kwargs
    ):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams["maximum_history_length"]
        self.max_ft = kwargs["min_future_timesteps"]
        self.node_type_datasets = list()
        self._augment = augment
        
        for node_type in env.NodeType:
            if node_type not in hyperparams["pred_state"]:
                continue
            # 使用聚类数据集
            self.node_type_datasets.append(
                ClusteredNodeTypeDataset(
                    env,
                    node_type,
                    state,
                    pred_state,
                    node_freq_mult,
                    scene_freq_mult,
                    hyperparams,
                    cluster_count=cluster_count,
                    augment=augment,
                    normalize_direction=normalize_direction,
                    dataset_name_for_cache=dataset_name_for_cache,
                    cache_dir=cache_dir,
                    **kwargs
                )
            )
