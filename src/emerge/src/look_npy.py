import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import h5py

def look_npy():
    # 加载文件
    print("Loading data files...")
    # 使用h5py加载OD矩阵
    with h5py.File('Data/enhanced_od_matrix.h5', 'r') as f:
        od_matrix = f['od_matrix'][:]
        # 可以根据需要加载adjacency_map
        # adj_group = f['adjacency_map']
        # adjacency_map = {int(k): adj_group[k][:] for k in adj_group.keys()}
    
    predicted_vehicles = np.load('Data/predicted_vehicles.npy')
    new_flow = np.load('Data/new_flow.npy')

    # 分析和显示OD矩阵
    print("\n" + "=" * 50)
    print("Data/enhanced_od_matrix.h5:")
    print(f"Shape: {od_matrix.shape}")
    print(f"Data type: {od_matrix.dtype}")
    print(f"Min value: {np.min(od_matrix)}")
    print(f"Max value: {np.max(od_matrix)}")
    print(f"Mean value: {np.mean(od_matrix)}")
    print(f"Non-zero elements: {np.count_nonzero(od_matrix)}")
    print(f"Non-zero percentage: {np.count_nonzero(od_matrix) / od_matrix.size * 100:.4f}%")
    print(f"First 10 non-zero elements: {od_matrix[od_matrix > 0].flatten()[:10]}")
    print("=" * 50)

    # 分析和显示预测车辆
    print("\n" + "=" * 50)
    print("Data/predicted_vehicles.npy:")
    print(f"Shape: {predicted_vehicles.shape}")
    print(f"Data type: {predicted_vehicles.dtype}")
    print(f"Min value: {np.min(predicted_vehicles)}")
    print(f"Max value: {np.max(predicted_vehicles)}")
    print(f"Mean value: {np.mean(predicted_vehicles)}")
    print(f"Standard deviation: {np.std(predicted_vehicles)}")
    print(f"Median: {np.median(predicted_vehicles)}")
    print(f"First 10 elements: {predicted_vehicles.flatten()[:10]}")
    print("=" * 50)

    # 分析和显示新流量
    print("\n" + "=" * 50)
    print("Data/new_flow.npy:")
    print(f"Shape: {new_flow.shape}")
    print(f"Data type: {new_flow.dtype}")
    print(f"Min value: {np.min(new_flow)}")
    print(f"Max value: {np.max(new_flow)}")
    print(f"Mean value: {np.mean(new_flow)}")
    print(f"Non-zero elements: {np.count_nonzero(new_flow)}")
    print(f"Non-zero percentage: {np.count_nonzero(new_flow) / new_flow.size * 100:.4f}%")
    print(f"First 10 non-zero elements: {new_flow[new_flow > 0].flatten()[:10]}")
    print("=" * 50)
    
if __name__ == "__main__":
    look_npy()
    
# # 创建保存图像的目录
# os.makedirs('npy_visualizations', exist_ok=True)



# # 可视化
# print("\nGenerating visualizations...")

# # 可视化OD矩阵 (选择第一个时间步)
# for t in range(min(3, od_matrix.shape[0])):  # 仅显示前3个时间步
#     plt.figure(figsize=(12, 5))
    
#     # 绘制第一个通道
#     plt.subplot(1, 2, 1)
#     plt.imshow(od_matrix[t, :, :, 0], cmap='viridis')
#     plt.colorbar()
#     plt.title(f'OD Matrix - Timestep {t} - Channel 0')
    
#     # 绘制第二个通道
#     plt.subplot(1, 2, 2)
#     plt.imshow(od_matrix[t, :, :, 1], cmap='viridis')
#     plt.colorbar()
#     plt.title(f'OD Matrix - Timestep {t} - Channel 1')
    
#     plt.tight_layout()
#     plt.savefig(f'npy_visualizations/od_matrix_t{t}.png')
#     plt.close()

# # 可视化预测车辆
# for t in range(min(3, predicted_vehicles.shape[0])):  # 仅显示前3个时间步
#     plt.figure(figsize=(8, 6))
#     plt.imshow(predicted_vehicles[t], cmap='hot')
#     plt.colorbar()
#     plt.title(f'Predicted Vehicles - Timestep {t}')
#     plt.tight_layout()
#     plt.savefig(f'npy_visualizations/predicted_vehicles_t{t}.png')
#     plt.close()

# # 可视化新流量
# for t in range(min(3, new_flow.shape[0])):  # 仅显示前3个时间步
#     plt.figure(figsize=(12, 5))
    
#     # 绘制第一个通道 (可能是x方向流量)
#     plt.subplot(1, 2, 1)
#     plt.imshow(new_flow[t, 0], cmap='coolwarm')
#     plt.colorbar()
#     plt.title(f'New Flow - Timestep {t} - Channel 0 (X-dir)')
    
#     # 绘制第二个通道 (可能是y方向流量)
#     plt.subplot(1, 2, 2)
#     plt.imshow(new_flow[t, 1], cmap='coolwarm')
#     plt.colorbar()
#     plt.title(f'New Flow - Timestep {t} - Channel 1 (Y-dir)')
    
#     plt.tight_layout()
#     plt.savefig(f'npy_visualizations/new_flow_t{t}.png')
#     plt.close()

# print(f"\nVisualizations saved to 'npy_visualizations' directory")
# print("Analysis complete!") 