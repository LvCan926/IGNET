import torch
import numpy as np
from dual_modal_flow import DualModalFlowPredictor
import os
from tqdm import tqdm

def load_flow_data(data_path):
    """
    加载流量数据
    
    参数:
        data_path: npy文件路径
    返回:
        加载的数据
    """
    try:
        data = np.load(data_path)
        print(f"从 {data_path} 加载数据，形状: {data.shape}")
        return torch.FloatTensor(data)
    except Exception as e:
        print(f"加载数据失败 {data_path}: {e}")
        return None

def train_model(
    hist_flow_path,
    traj_flow_path,
    real_flow_path,
    num_epochs=100,
    batch_size=32,
    save_dir='checkpoints',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    训练双模态流量预测模型
    
    参数:
        hist_flow_path: 历史流量数据路径
        traj_flow_path: 轨迹流量数据路径
        real_flow_path: 真实流量数据路径
        num_epochs: 训练轮数
        batch_size: 批次大小
        save_dir: 模型保存目录
        device: 训练设备
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    hist_flow = load_flow_data(hist_flow_path)
    traj_flow = load_flow_data(traj_flow_path)
    real_flow = load_flow_data(real_flow_path)
    
    if hist_flow is None or traj_flow is None or real_flow is None:
        print("数据加载失败，退出训练")
        return
    
    # 确保数据维度正确
    # 历史流量: [batch_size, 2, 108, 32, 32]
    # 轨迹流量: [batch_size, 2, 12, 32, 32]
    # 真实流量: [batch_size, 2, 12, 32, 32]
    
    # 初始化模型
    model = DualModalFlowPredictor()
    model = model.to(device)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 使用tqdm显示进度
        pbar = tqdm(range(0, len(hist_flow), batch_size), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i in pbar:
            # 获取当前批次
            batch_hist = hist_flow[i:i+batch_size].to(device)
            batch_traj = traj_flow[i:i+batch_size].to(device)
            batch_real = real_flow[i:i+batch_size].to(device)
            
            # 前向传播
            pred_flow = model(batch_hist, batch_traj)
            
            # 计算损失
            loss = model.update(pred_flow, batch_real)
            
            total_loss += loss
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}')
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存到: {save_path}')

if __name__ == '__main__':
    # 示例用法
    hist_flow_path = 'path/to/historical_flow.npy'
    traj_flow_path = 'path/to/trajectory_flow.npy'
    real_flow_path = 'path/to/real_flow.npy'
    
    train_model(
        hist_flow_path=hist_flow_path,
        traj_flow_path=traj_flow_path,
        real_flow_path=real_flow_path,
        num_epochs=100,
        batch_size=32
    ) 