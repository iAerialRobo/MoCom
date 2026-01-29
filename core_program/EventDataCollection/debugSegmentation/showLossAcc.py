import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置要绘制的前n个点
n = 100  # 你可以修改这个值来选择前多少个点

pathFileTrA = r'C:\Users\ZhuanZ.DESKTOP-LJJ6K3U\Desktop\newPicture\trainAcc.csv'
pathFileTrL = r'C:\Users\ZhuanZ.DESKTOP-LJJ6K3U\Desktop\newPicture\trainLoss.csv'
pathFileTeA = r'C:\Users\ZhuanZ.DESKTOP-LJJ6K3U\Desktop\newPicture\testAcc.csv'
pathFileTeL = r'C:\Users\ZhuanZ.DESKTOP-LJJ6K3U\Desktop\newPicture\testLoss.csv'

# 读取数据
train_acc = pd.read_csv(pathFileTrA)    # 第一个文档数据
train_loss = pd.read_csv(pathFileTrL)  # 第二个文档数据
test_acc = pd.read_csv(pathFileTeA)      # 第三个文档数据
test_loss = pd.read_csv(pathFileTeL)    # 第四个文档数据


# 定义平滑函数（移动平均）
def smooth_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 设置参数
n = 40  # 只取前40个点
window_size = 5  # 平滑窗口大小，可以调整这个值

# 只取前n个点的数据
train_acc = train_acc[:n]
train_loss = train_loss[:n]
test_acc = test_acc[:n]
test_loss = test_loss[:n]

# 平滑数据
train_acc_smooth = smooth_data(train_acc['Value'], window_size)
train_loss_smooth = smooth_data(train_loss['Value'], window_size)
test_acc_smooth = smooth_data(test_acc['Value'], window_size)
test_loss_smooth = smooth_data(test_loss['Value'], window_size)

# 因为移动平均会减少数据点数，调整对应的Step
steps = train_acc['Step'][window_size-1:].values

# 创建子图：1行2列
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 绘制Accuracy曲线
ax1.plot(steps, train_acc_smooth, 'g-', label='Train Accuracy')
ax1.plot(steps, test_acc_smooth, 'r-', label='Test Accuracy')
ax1.set_title(f'Training and Testing Accuracy (First {n} Steps, Smoothed)')
ax1.set_xlabel('Step')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# 绘制Loss曲线
ax2.plot(steps, train_loss_smooth, 'g-', label='Train Loss ')
ax2.plot(steps, test_loss_smooth, 'r-', label='Test Loss')
ax2.set_title(f'Training and Testing Loss (First {n} Steps, Smoothed)')
ax2.set_xlabel('Step')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()