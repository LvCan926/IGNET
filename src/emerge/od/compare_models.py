import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hawkes_model import HawkesProcess, load_and_process_data as hawkes_load_data
from neural_network_model import NeuralNetworkPredictor, load_and_process_data as nn_load_data

def main():
    """
    比较Hawkes过程模型和神经网络模型的性能
    """
    print("开始加载数据...")
    
    # 加载数据
    train_data = hawkes_load_data('train.txt')
    val_data = hawkes_load_data('val.txt')
    test_data = hawkes_load_data('test.txt')
    
    print(f"数据加载完成。训练数据: {len(train_data)}行, 验证数据: {len(val_data)}行, 测试数据: {len(test_data)}行")
    
    # 训练Hawkes过程模型
    print("\n=== Hawkes过程模型 ===")
    hawkes_model = HawkesProcess()
    hawkes_model.fit(train_data)
    
    # 在验证集和测试集上评估Hawkes模型
    hawkes_val_mse, hawkes_val_mae = hawkes_model.evaluate(val_data)
    print(f"Hawkes模型在验证集上的结果 - MSE: {hawkes_val_mse:.4f}, MAE: {hawkes_val_mae:.4f}")
    
    hawkes_test_mse, hawkes_test_mae = hawkes_model.evaluate(test_data)
    print(f"Hawkes模型在测试集上的结果 - MSE: {hawkes_test_mse:.4f}, MAE: {hawkes_test_mae:.4f}")
    
    # 训练神经网络模型
    print("\n=== 神经网络模型 ===")
    
    # 设置模型参数
    seq_length = 5
    hidden_size = 64
    num_layers = 2
    batch_size = 64
    epochs = 30
    lr = 0.001
    
    nn_model = NeuralNetworkPredictor(
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr
    )
    
    # 训练神经网络模型
    nn_model.fit(train_data, val_data)
    
    # 绘制损失曲线
    nn_model.plot_loss()
    
    # 在测试集上评估神经网络模型
    nn_test_mse, nn_test_mae = nn_model.evaluate(test_data)
    print(f"神经网络模型在测试集上的结果 - MSE: {nn_test_mse:.4f}, MAE: {nn_test_mae:.4f}")
    
    # 比较两个模型
    print("\n=== 模型比较 ===")
    comparison_data = {
        "模型": ["Hawkes过程", "神经网络"],
        "测试MSE": [hawkes_test_mse, nn_test_mse],
        "测试MAE": [hawkes_test_mae, nn_test_mae]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    
    # 绘制对比图
    plot_comparison(comparison_df)
    
    # 实体级别的比较
    print("\n=== 实体级别的对比 ===")
    entity_comparison(hawkes_model, nn_model, test_data)

def plot_comparison(comparison_df):
    """绘制模型比较图"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE比较
    ax[0].bar(comparison_df["模型"], comparison_df["测试MSE"], color=['blue', 'orange'])
    ax[0].set_title('模型MSE对比')
    ax[0].set_ylabel('均方误差 (MSE)')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAE比较
    ax[1].bar(comparison_df["模型"], comparison_df["测试MAE"], color=['blue', 'orange'])
    ax[1].set_title('模型MAE对比')
    ax[1].set_ylabel('平均绝对误差 (MAE)')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def entity_comparison(hawkes_model, nn_model, test_data):
    """在实体级别比较两个模型的性能"""
    # 按实体分组
    grouped = test_data.groupby('entity_id')
    
    entity_results = []
    
    # 对每个实体进行评估
    for entity_id, group in grouped:
        # 确保数据按时间排序
        group = group.sort_values('time_step')
        
        # 只选择有足够数据的实体
        if len(group) <= 5:
            continue
            
        # Hawkes模型预测
        hawkes_predictions = []
        actuals = []
        
        for i in range(1, len(group)):
            history_times = group['time_step'].values[:i]
            pred = hawkes_model.predict(entity_id, group['time_step'].values[i], history_times)
            hawkes_predictions.append(pred)
            actuals.append(group['count'].values[i] - 1)  # 减去1得到新生车辆数
        
        hawkes_mse = np.mean((np.array(hawkes_predictions) - np.array(actuals)) ** 2)
        hawkes_mae = np.mean(np.abs(np.array(hawkes_predictions) - np.array(actuals)))
        
        # 神经网络模型预测
        nn_predictions = []
        
        for i in range(5, len(group)):
            history = group.iloc[i-5:i]
            pred = nn_model.predict(history)
            nn_predictions.append(pred)
        
        # 确保比较相同长度的预测结果
        actuals = actuals[4:]  # 跳过前5个，因为神经网络模型需要5个历史数据点
        
        nn_mse = np.mean((np.array(nn_predictions) - np.array(actuals)) ** 2)
        nn_mae = np.mean(np.abs(np.array(nn_predictions) - np.array(actuals)))
        
        entity_results.append({
            'entity_id': entity_id,
            'hawkes_mse': hawkes_mse,
            'hawkes_mae': hawkes_mae,
            'nn_mse': nn_mse,
            'nn_mae': nn_mae,
            'sample_count': len(actuals)
        })
    
    # 创建结果DataFrame
    entity_df = pd.DataFrame(entity_results)
    
    # 输出前10个实体的结果
    print("前10个实体的模型性能对比:")
    print(entity_df.head(10))
    
    # 计算每个模型在多少个实体上表现更好
    hawkes_better_mse = (entity_df['hawkes_mse'] < entity_df['nn_mse']).sum()
    nn_better_mse = (entity_df['nn_mse'] < entity_df['hawkes_mse']).sum()
    
    hawkes_better_mae = (entity_df['hawkes_mae'] < entity_df['nn_mae']).sum()
    nn_better_mae = (entity_df['nn_mae'] < entity_df['hawkes_mae']).sum()
    
    print(f"\nHawkes模型在{hawkes_better_mse}个实体上MSE表现更好，神经网络在{nn_better_mse}个实体上MSE表现更好")
    print(f"Hawkes模型在{hawkes_better_mae}个实体上MAE表现更好，神经网络在{nn_better_mae}个实体上MAE表现更好")
    
    # 绘制散点图比较
    plt.figure(figsize=(10, 8))
    plt.scatter(entity_df['hawkes_mse'], entity_df['nn_mse'], alpha=0.6)
    
    # 添加对角线(y=x)
    max_val = max(entity_df['hawkes_mse'].max(), entity_df['nn_mse'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.xlabel('Hawkes模型 MSE')
    plt.ylabel('神经网络模型 MSE')
    plt.title('按实体比较两个模型的MSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('entity_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 