import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
import numpy as np
import matplotlib.pyplot as plt

# 假设数据准备
lif = neuron.LIFNode(tau=100.)
x = torch.rand(size=[32]) * 4  # 输入
T = 50  # 时间步长
s_list = []  # 脉冲
v_list = []  # 膜电位
for t in range(T):
    s_list.append(lif(x).unsqueeze(0))
    v_list.append(lif.v.unsqueeze(0))
s_array = torch.cat(s_list).numpy()  # [T, N]
v_array = torch.cat(v_list).numpy()  # [T, N]

# 1. plot_2d_heatmap 示例
plt.figure()
visualizing.plot_2d_heatmap(
    array=v_array,
    title='Membrane Potentials',
    xlabel='Simulating Step',
    ylabel='Neuron Index',
    int_x_ticks=True,
    x_max=T,
    dpi=200
)
plt.show()

# 2. plot_2d_bar_in_3d 示例
plt.figure()
visualizing.plot_2d_bar_in_3d(
    array=s_array,
    title='Spiking Rates',
    xlabel='Neuron Index',
    ylabel='Simulating Step',
    zlabel='Spike Count',
    int_x_ticks=True,
    int_y_ticks=True,
    dpi=200
)
plt.show()

# 3. plot_1d_spikes 示例
plt.figure()
visualizing.plot_1d_spikes(
    spikes=s_array,
    title='Spike Events',
    xlabel='Simulating Step',
    ylabel='Neuron Index',
    int_x_ticks=True,
    plot_firing_rate=True,
    dpi=200
)
plt.show()

# 4. plot_2d_feature_map 示例（假设卷积层输出）
C, W, H = 16, 8, 8
conv_spikes = (np.random.rand(C, W, H) > 0.8).astype(float)
plt.figure()
visualizing.plot_2d_feature_map(
    x3d=conv_spikes,
    nrows=4,
    ncols=4,
    space=2,
    title='Convolutional Feature Maps',
    dpi=200
)
plt.show()

# 5. plot_one_neuron_v_s 示例（单个神经元）
# 数据准备
single_lif = neuron.LIFNode(tau=100.)
x_single = torch.tensor([2.0])  # 输入标量
T = 150
s_single = []
v_single = []
for t in range(T):
    spike = single_lif(x_single)  # 获取脉冲输出
    s_single.append(spike.item())  # 转换为标量
    v_single.append(single_lif.v.item())  # 转换为标量
s_array = np.array(s_single)  # [T]
v_array = np.array(v_single)  # [T]

# 调试打印
print("s_array shape:", s_array.shape)
print("v_array shape:", v_array.shape)

# 调用可视化函数
visualizing.plot_one_neuron_v_s(
    v=v_array,
    s=s_array,
    v_threshold=single_lif.v_threshold,
    v_reset=single_lif.v_reset,
    title='Single Neuron Dynamics',
    dpi=200
)
plt.show()