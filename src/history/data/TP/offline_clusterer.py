import argparse
import os
import pickle
import warnings
import numpy as np
import torch
import time

# Assuming these modules are accessible from the script's location
# Adjust paths if necessary, or ensure this script is run from a context where these imports work.
from environment import Environment, Scene, Node
from trajectron_dataset import NodeTypeDataset, SKLEARN_AVAILABLE, hypers as default_hyperparams
from preprocessing import get_node_timestep_data

if SKLEARN_AVAILABLE:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
else:
    warnings.warn("scikit-learn not installed, clustering functionality will not be available. Install with: pip install scikit-learn")
    KMeans = None
    pairwise_distances_argmin_min = None


def compute_clusters_for_node_type(
    nt_dataset: NodeTypeDataset,
    cluster_count: int,
    hyperparams: dict
):
    """
    Computes clustered index and cluster info for a given NodeTypeDataset.
    This function is adapted from ClusteredNodeTypeDataset._compute_clusters_and_info.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for clustering.")

    # 1. Organize indices by scene and timestep (similar to _organize_indices)
    scene_timestep_dict = {}
    for idx, (scene, t, node) in enumerate(nt_dataset.index):
        key = (scene, t)
        if key not in scene_timestep_dict:
            scene_timestep_dict[key] = []
        scene_timestep_dict[key].append((idx, node)) # Store original index and node

    clustered_index_tuples = []  # To store (scene_id, t, node_id) for representatives
    cluster_info_map = {}      # To store (scene_id, t, node_id_repr) -> cluster_size

    print(f"Processing node type: {nt_dataset.node_type.name}")
    for (scene, t), node_indices_with_orig_idx in scene_timestep_dict.items():
        # node_indices_with_orig_idx is a list of (original_nt_dataset_idx, node_object)
        
        if len(node_indices_with_orig_idx) <= cluster_count:
            for orig_idx, node in node_indices_with_orig_idx:
                clustered_index_tuples.append((scene.name, t, node.id))
                cluster_info_map[(scene.name, t, node.id)] = 1
            continue

        history_features = []
        original_indices_for_scene_time = [] # Stores original_nt_dataset_idx
        all_nodes_for_scene_time = []

        for orig_idx, node in node_indices_with_orig_idx:
            # Get node data (history) using NodeTypeDataset's __getitem__
            # This internally calls get_node_timestep_data
            # Temporarily disable augmentation for feature extraction if nt_dataset has it
            # (Assuming nt_dataset.augment is not used or handled if it affects history)
            node_data = nt_dataset[orig_idx] # This uses the __getitem__ of NodeTypeDataset

            x_t = node_data[1]  # History tensor
            if isinstance(x_t, torch.Tensor):
                history_features.append(x_t.flatten().cpu().numpy())
            else: # Assuming numpy array
                history_features.append(x_t.flatten())
                
            original_indices_for_scene_time.append(orig_idx)
            all_nodes_for_scene_time.append(node)

        if not history_features:
            continue
            
        history_features_np = np.array(history_features)
        
        # Ensure n_clusters is not more than n_samples
        current_n_clusters = min(cluster_count, len(history_features_np))
        if current_n_clusters == 0 :
             continue


        kmeans = KMeans(n_clusters=current_n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(history_features_np)
        
        cluster_centers = kmeans.cluster_centers_
        # pairwise_distances_argmin_min finds the closest original sample to each cluster center
        closest_original_sample_indices_in_batch, _ = pairwise_distances_argmin_min(cluster_centers, history_features_np)
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        label_to_count = dict(zip(unique_labels, counts))

        for representative_batch_idx in closest_original_sample_indices_in_batch:
            # node is the representative node object
            node = all_nodes_for_scene_time[representative_batch_idx]
            # cluster_id for this representative node
            cluster_id_of_representative = cluster_labels[representative_batch_idx]
            # size of the cluster this node represents
            size_of_cluster = label_to_count[cluster_id_of_representative]

            clustered_index_tuples.append((scene.name, t, node.id))
            cluster_info_map[(scene.name, t, node.id)] = int(size_of_cluster)
            
    return clustered_index_tuples, cluster_info_map


def main():
    parser = argparse.ArgumentParser(description="Offline clusterer for trajectory data.")
    parser.add_argument("--env_path", default="data/processed/BJTaxi_test.pkl", help="Path to the environment data file (e.g., .pkl).")
    parser.add_argument("--dataset_name", default="BJTaxi", help="Name of the dataset (e.g., 'eth', 'sdd'), used for naming cache files.")
    parser.add_argument("--cache_dir", default="data/processed/cache", help="Directory to save the_cluster cache files.")
    parser.add_argument("--cluster_count", type=int, default=2, help="Number of clusters (K for K-Means).")
    parser.add_argument("--min_history_timesteps", type=int, default=8, help="Minimum history timesteps for nodes.")
    parser.add_argument("--min_future_timesteps", type=int, default=12, help="Minimum future timesteps for nodes.")
    # Add more arguments if necessary, e.g., for hyperparams, node_freq_mult, scene_freq_mult

    args = parser.parse_args()

    if not SKLEARN_AVAILABLE:
        print("scikit-learn is not installed. Clustering cannot proceed.")
        return

    print(f"Loading environment from: {args.env_path}")
    with open(args.env_path, 'rb') as f:
        env = pickle.load(f)
    print("Environment loaded.")

    # Use default_hyperparams or allow loading from a config file
    hyperparams = default_hyperparams 
    # Potentially override hyperparams['pred_state'], state, pred_state from args if needed
    # For simplicity, using defaults from trajectron_dataset.py for now
    state = hyperparams['state_pva'] 
    pred_state = hyperparams['pred_state']

    os.makedirs(args.cache_dir, exist_ok=True)

    for node_type in env.NodeType:
        if node_type not in pred_state:
            print(f"Skipping node type {node_type.name} as it's not in pred_state.")
            continue

        print(f"Processing node type: {node_type.name} for dataset: {args.dataset_name}")

        # Instantiate NodeTypeDataset to get the full index and access to __getitem__
        # These kwargs are passed to NodeTypeDataset constructor
        # and then to index_env
        # TODO: Expose node_freq_mult, scene_freq_mult, etc., as script arguments if they vary.
        nt_dataset_kwargs = {
            'min_history_timesteps': args.min_history_timesteps,
            'min_future_timesteps': args.min_future_timesteps,
            # Add other necessary kwargs for NodeTypeDataset and its index_env method
        }

        node_type_dataset = NodeTypeDataset(
            env,
            node_type,
            state, # state configuration for node features
            pred_state, # pred_state configuration (used by ClusteredNodeTypeDataset, maybe not directly by NodeTypeDataset for indexing)
            node_freq_mult=False, # Example, make this configurable
            scene_freq_mult=False, # Example, make this configurable
            hyperparams=hyperparams,
            augment=False, # Typically False for offline processing
            normalize_direction=False, # Typically False unless features are normalized pre-clustering
            **nt_dataset_kwargs
        )
        
        if not node_type_dataset.index:
            print(f"No data found for node type {node_type.name}. Skipping.")
            continue

        clustered_indices, cluster_info = compute_clusters_for_node_type(
            node_type_dataset,
            args.cluster_count,
            hyperparams
        )

        # Save the results
        base_filename = f"{args.dataset_name}_{node_type.name}_k{args.cluster_count}"
        index_cache_path = os.path.join(args.cache_dir, f"{base_filename}_indices.pkl")
        info_cache_path = os.path.join(args.cache_dir, f"{base_filename}_info.pkl")

        try:
            with open(index_cache_path, 'wb') as f:
                pickle.dump(clustered_indices, f)
            print(f"Saved clustered indices to: {index_cache_path}")

            with open(info_cache_path, 'wb') as f:
                pickle.dump(cluster_info, f)
            print(f"Saved cluster info to: {info_cache_path}")
        except IOError as e:
            print(f"Error saving cache files for {node_type.name}: {e}")

    print("Offline clustering process completed.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Offline clustering process completed. Time taken: {end_time - start_time:.2f} seconds")