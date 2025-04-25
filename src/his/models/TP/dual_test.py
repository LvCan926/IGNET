from dual_modal_flow import DualModalFlowPredictor
import torch

model = DualModalFlowPredictor()

# 构造虚拟输入数据
hist_flow = torch.randn(1, 2, 108, 32, 32)  # 符合(108,2,32,32)的历史流量
traj_flow = torch.randn(1, 2, 12, 32, 32)   # 轨迹聚合流量

# 多次预测对比
pred1 = model.forward(hist_flow, traj_flow)
pred2 = model.forward(hist_flow, traj_flow)
print("预测结果差异:", torch.mean((pred1 - pred2)**2))  # 输出接近0.0
