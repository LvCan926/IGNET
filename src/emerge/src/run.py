import os
import numpy as np
import time
from tqdm import tqdm
import look_npy

def run_data_processing():
    """运行数据处理脚本"""
    print("===== 第1步：数据处理 =====")
    import data_process
    
    # 检查已处理的数据文件是否存在
    if (os.path.exists("NewBornFlow/Data/train_data.npy") and 
        os.path.exists("NewBornFlow/Data/test_data.npy") and 
        os.path.exists("NewBornFlow/Data/time_steps.npy")):
        print("数据文件已存在，跳过处理步骤")
        return
    
    # 否则，运行数据处理
    start_time = time.time()
    
    # 加载并处理数据
    data, time_steps, time_to_idx = data_process.load_and_process_data(
        "Data/MGF/txt/BJ_MGF_processed_10080.txt",
        time_start=0,
        time_end=10079
    )
    
    # 分割训练集和测试集
    train_data, test_data = data_process.split_train_test(data, time_steps)
    
    # 保存处理后的数据
    np.save("NewBornFlow/Data/train_data.npy", train_data)
    np.save("NewBornFlow/Data/test_data.npy", test_data)
    np.save("NewBornFlow/Data/time_steps.npy", np.array(time_steps))
    
    end_time = time.time()
    print(f"数据处理完成，用时: {end_time - start_time:.2f}秒")

def run_hawkes_model():
    """训练和评估Hawkes过程模型"""
    print("\n===== 第2步：Hawkes过程模型 =====")
    from hawkes_model import HawkesGridModel
    
    # 加载处理后的数据
    train_data = np.load("NewBornFlow/Data/train_data.npy")
    test_data = np.load("NewBornFlow/Data/test_data.npy")
    time_steps = np.load("NewBornFlow/Data/time_steps.npy")
    
    # 分割时间步
    train_time_steps = time_steps[:len(train_data)]
    test_time_steps = time_steps[len(train_data):]
    
    # 创建并训练Hawkes模型
    start_time = time.time()
    
    hawkes_model = HawkesGridModel()
    
    # 检查是否有已保存的模型参数
    if os.path.exists("NewBornFlow/Data/hawkes_grid_params.npy"):
        hawkes_model.grid_params = np.load("NewBornFlow/Data/hawkes_grid_params.npy", allow_pickle=True).item()
        print("已加载保存的Hawkes模型参数")
    else:
        hawkes_model.fit(train_data, train_time_steps)
        # 保存模型参数
        np.save("NewBornFlow/Data/hawkes_grid_params.npy", hawkes_model.grid_params)
    
    # 评估模型
    mse, mae = hawkes_model.evaluate(test_data, train_data, train_time_steps, test_time_steps)
    
    end_time = time.time()
    
    # 打印结果
    print(f"Hawkes模型训练和评估完成，用时: {end_time - start_time:.2f}秒")
    print(f"评估结果 - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    return hawkes_model, mse, mae

def run_neural_network_model():
    """训练和评估神经网络模型"""
    print("\n===== 第3步：神经网络模型 =====")
    from neural_network_model import NNGridModel
    
    # 加载处理后的数据
    train_data = np.load("NewBornFlow/Data/train_data.npy")
    test_data = np.load("NewBornFlow/Data/test_data.npy")
    
    # 创建并训练神经网络模型
    start_time = time.time()
    
    nn_model = NNGridModel(
        seq_length=5,
        hidden_dim=64,
        num_layers=2,
        batch_size=16,
        epochs=30,
        lr=0.001
    )
    
    # 检查是否有已保存的模型
    if os.path.exists("NewBornFlow/Data/nn_grid_model.pth"):
        nn_model.load_model("NewBornFlow/Data/nn_grid_model.pth")
        print("已加载保存的神经网络模型")
    else:
        # 训练模型
        nn_model.fit(train_data)
        # 保存模型
        nn_model.save_model("NewBornFlow/Data/nn_grid_model.pth")
    
    
    # 评估模型
    mse, mae = nn_model.evaluate(test_data, train_data)
    
    end_time = time.time()
    
    # 打印结果
    print(f"神经网络模型训练和评估完成，用时: {end_time - start_time:.2f}秒")
    print(f"评估结果 - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    return nn_model, mse, mae

def run_model_comparison(hawkes_model, hawkes_mse, hawkes_mae, nn_model, nn_mse, nn_mae):
    """比较两个模型的性能"""
    print("\n===== 第4步：模型比较 =====")
    import compare_models
    
    # 打印比较结果
    print("\n模型比较结果:")
    comparison_data = {
        "模型": ["Hawkes过程", "神经网络"],
        "MSE": [hawkes_mse, nn_mse],
        "MAE": [hawkes_mae, nn_mae]
    }
    
    # 使用pandas打印表格
    try:
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df)
    except ImportError:
        print("无法导入pandas，使用普通格式打印")
        print("模型      MSE      MAE")
        print(f"Hawkes过程  {hawkes_mse:.4f}  {hawkes_mae:.4f}")
        print(f"神经网络    {nn_mse:.4f}  {nn_mae:.4f}")
    
    # 确定更好的模型
    if hawkes_mse <= nn_mse:
        print("Hawkes过程模型的性能更好")
        better_model = "hawkes"
    else:
        print("神经网络模型的性能更好")
        better_model = "nn"
    
    return better_model

def run_flow_generation(hawkes_model, nn_model, better_model):
    """根据选择的模型生成新生流量"""
    print("\n===== 第5步：生成新生流量 =====")
    
    # 检查OD矩阵文件是否存在
    od_matrix_file = "NewBornFlow/Data/enhanced_od_matrix.h5"
    if not os.path.exists(od_matrix_file):
        print(f"警告: OD矩阵文件不存在: {od_matrix_file}")
        print("正在生成OD矩阵...")
        
        # 生成OD矩阵
        try:
            if os.path.exists("OD/enhanced_od_matrix.py"):
                import subprocess
                subprocess.run(["python", "OD/enhanced_od_matrix.py"])
                print("OD矩阵生成完成")
            else:
                print(f"错误: 找不到OD矩阵生成脚本: OD/enhanced_od_matrix.py")
                return
        except Exception as e:
            print(f"生成OD矩阵时出错: {e}")
            return
    
    # 检查预测车辆数据文件是否存在
    predictions_file = "NewBornFlow/Data/predicted_vehicles.npy"
    if not os.path.exists(predictions_file):
        print(f"预测车辆数据文件不存在: {predictions_file}")
        print("正在使用选定模型生成预测数据...")
        
        # 加载训练数据
        train_data = np.load("NewBornFlow/Data/train_data.npy")
        time_steps = np.load("NewBornFlow/Data/time_steps.npy")
        train_time_steps = time_steps[:len(train_data)]
        
        # 要预测的时间步范围（1128-1139）
        test_time_steps = np.arange(7200, 10080)
        
        # 使用选择的模型生成预测
        from generate_flow import generate_test_predictions
        predictions = generate_test_predictions(
            test_time_steps=test_time_steps, 
            train_data=train_data, 
            train_time_steps=train_time_steps, 
            hawkes_model=hawkes_model, 
            nn_model=nn_model, 
            select_model=better_model
        )
        
        # 保存预测数据
        print(f"保存预测车辆数据到: {predictions_file}")
        np.save(predictions_file, predictions)
        print(f"预测数据形状: {predictions.shape}")
    else:
        print(f"使用已存在的预测车辆数据文件: {predictions_file}")
    
    # 运行流量生成脚本
    import final_flow_generator
    final_flow_generator.main()
    
    print("新生流量生成完成！")

def look_npy():
    # 加载文件
    print("Loading data files...")
    # 使用h5py加载OD矩阵
    import h5py
    with h5py.File('NewBornFlow/Data/enhanced_od_matrix.h5', 'r') as f:
        od_matrix = f['od_matrix'][:]
        # 可以根据需要加载adjacency_map
        # adj_group = f['adjacency_map']
        # adjacency_map = {int(k): adj_group[k][:] for k in adj_group.keys()}
    
    predicted_vehicles = np.load('NewBornFlow/Data/predicted_vehicles.npy', allow_pickle=True)
    new_flow = np.load('NewBornFlow/Data/new_flow_forced.npy', allow_pickle=True)

    # 分析和显示OD矩阵
    print("\n" + "=" * 50)
    print("NewBornFlow/Data/enhanced_od_matrix.h5:")
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
    print("NewBornFlow/Data/predicted_vehicles.npy:")
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
    print("NewBornFlow/Data/new_flow_forced.npy:")
    print(f"Shape: {new_flow.shape}")
    print(f"Data type: {new_flow.dtype}")
    print(f"Min value: {np.min(new_flow)}")
    print(f"Max value: {np.max(new_flow)}")
    print(f"Mean value: {np.mean(new_flow)}")
    print(f"Non-zero elements: {np.count_nonzero(new_flow)}")
    print(f"Non-zero percentage: {np.count_nonzero(new_flow) / new_flow.size * 100:.4f}%")
    print(f"First 10 non-zero elements: {new_flow[new_flow > 0].flatten()[:10]}")
    print("=" * 50)

def main():
    """主运行函数"""
    print("开始运行车辆预测模型实验...")
    
    # 步骤1：数据处理
    run_data_processing()
    
    # 步骤2：训练和评估Hawkes过程模型
    hawkes_model, hawkes_mse, hawkes_mae = run_hawkes_model()
    
    # 步骤3：训练和评估神经网络模型
    nn_model, nn_mse, nn_mae = run_neural_network_model()
    
    # 步骤4：比较两个模型
    better_model = run_model_comparison(
        hawkes_model, hawkes_mse, hawkes_mae, 
        nn_model, nn_mse, nn_mae
    )
    
    # 步骤5：根据选择的模型生成新生流量
    run_flow_generation(hawkes_model, nn_model, nn_model)
    
    # 步骤6：查看npy内容
    look_npy()
    
    print("\n实验完成！请查看生成的图表和结果文件。")

if __name__ == "__main__":
    main() 