import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from hawkes_model import HawkesGridModel
from neural_network_model import NNGridModel

def main():
    """
    比较Hawkes过程模型和神经网络模型的性能
    """
    print("加载数据...")
    
    # 加载处理后的数据
    train_data = np.load("train_data.npy")
    test_data = np.load("test_data.npy")
    time_steps = np.load("time_steps.npy")
    
    # 分割时间步
    train_time_steps = time_steps[:len(train_data)]
    test_time_steps = time_steps[len(train_data):]
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    print(f"训练时间步: {train_time_steps}")
    print(f"测试时间步: {test_time_steps}")
    
    # 训练Hawkes过程模型
    print("\n===== Hawkes过程模型 =====")
    hawkes_start_time = time.time()
    
    hawkes_model = HawkesGridModel()
    hawkes_model.fit(train_data, train_time_steps)
    
    # 评估Hawkes模型
    hawkes_mse, hawkes_mae = hawkes_model.evaluate(
        test_data, 
        train_data, 
        train_time_steps, 
        test_time_steps
    )
    
    hawkes_end_time = time.time()
    hawkes_training_time = hawkes_end_time - hawkes_start_time
    
    print(f"Hawkes模型评估结果 - MSE: {hawkes_mse:.4f}, MAE: {hawkes_mae:.4f}")
    print(f"Hawkes模型训练时间: {hawkes_training_time:.2f}秒")
    
    # 训练神经网络模型
    print("\n===== 神经网络模型 =====")
    nn_start_time = time.time()
    
    nn_model = NNGridModel(
        seq_length=5,
        hidden_dim=64,
        num_layers=2,
        batch_size=16,
        epochs=30,
        lr=0.001
    )
    
    nn_model.fit(train_data)
    
    # 评估神经网络模型
    nn_mse, nn_mae = nn_model.evaluate(test_data, train_data)
    
    nn_end_time = time.time()
    nn_training_time = nn_end_time - nn_start_time
    
    print(f"神经网络模型评估结果 - MSE: {nn_mse:.4f}, MAE: {nn_mae:.4f}")
    print(f"神经网络模型训练时间: {nn_training_time:.2f}秒")
    
    # 比较结果
    print("\n===== 模型比较 =====")
    comparison_data = {
        "模型": ["Hawkes过程", "神经网络"],
        "MSE": [hawkes_mse, nn_mse],
        "MAE": [hawkes_mae, nn_mae],
        "训练时间(秒)": [hawkes_training_time, nn_training_time]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    
    # 绘制比较图
    # plot_comparison(comparison_df)
    
    # 可视化预测结果
    # visualize_predictions(hawkes_model, nn_model, test_data, train_data, 
    #                       train_time_steps, test_time_steps)

def plot_comparison(comparison_df):
    """绘制模型比较图"""
    # 性能对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE对比
    axes[0].bar(comparison_df["模型"], comparison_df["MSE"], color=['blue', 'orange'])
    axes[0].set_title('MSE对比')
    axes[0].set_ylabel('均方误差')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAE对比
    axes[1].bar(comparison_df["模型"], comparison_df["MAE"], color=['blue', 'orange'])
    axes[1].set_title('MAE对比')
    axes[1].set_ylabel('平均绝对误差')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 训练时间对比
    axes[2].bar(comparison_df["模型"], comparison_df["训练时间(秒)"], color=['blue', 'orange'])
    axes[2].set_title('训练时间对比')
    axes[2].set_ylabel('训练时间(秒)')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def visualize_predictions(hawkes_model, nn_model, test_data, train_data, 
                         train_time_steps, test_time_steps):
    """可视化两个模型的预测结果"""
    # 选择第一个测试时间步进行可视化
    test_idx = 0
    test_step = test_time_steps[test_idx]
    
    # Hawkes模型预测
    hawkes_pred = hawkes_model.predict(train_data, test_step, train_time_steps)
    
    # 神经网络模型预测
    nn_pred = nn_model.predict(train_data[-5:])
    
    # 真实值
    actual = test_data[test_idx]
    
    # 计算每个模型在每个网格上的绝对误差
    hawkes_error = np.abs(hawkes_pred - actual)
    nn_error = np.abs(nn_pred - actual)
    
    # 创建可视化图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 真实值热图
    im0 = axes[0, 0].imshow(actual, cmap='viridis')
    axes[0, 0].set_title(f'真实值 (时间步 {test_step})')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Hawkes预测热图
    im1 = axes[0, 1].imshow(hawkes_pred, cmap='viridis')
    axes[0, 1].set_title('Hawkes模型预测')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Hawkes误差热图
    im2 = axes[0, 2].imshow(hawkes_error, cmap='Reds')
    axes[0, 2].set_title('Hawkes模型绝对误差')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 真实值热图(重复，用于比较)
    im3 = axes[1, 0].imshow(actual, cmap='viridis')
    axes[1, 0].set_title(f'真实值 (时间步 {test_step})')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 神经网络预测热图
    im4 = axes[1, 1].imshow(nn_pred, cmap='viridis')
    axes[1, 1].set_title('神经网络模型预测')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 神经网络误差热图
    im5 = axes[1, 2].imshow(nn_error, cmap='Reds')
    axes[1, 2].set_title('神经网络模型绝对误差')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.close()
    
    # 绘制误差分布直方图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(hawkes_error.flatten(), bins=50, alpha=0.7, label='Hawkes')
    plt.hist(nn_error.flatten(), bins=50, alpha=0.7, label='神经网络')
    plt.xlabel('绝对误差')
    plt.ylabel('频率')
    plt.title('绝对误差分布')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([hawkes_error.flatten(), nn_error.flatten()], 
                labels=['Hawkes', '神经网络'])
    plt.ylabel('绝对误差')
    plt.title('误差箱线图')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()

if __name__ == "__main__":
    main() 