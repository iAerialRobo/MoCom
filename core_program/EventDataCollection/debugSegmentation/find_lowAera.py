import numpy as np
import matplotlib.pyplot as plt

# 读取数据
file_path = 'D:\\eventVision\\collecting\\3_3\\3\\UAV_event_statistics.txt'

frame_indices = []
event_counts = []
with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        frame_idx = int(parts[0])
        event_count = int(parts[3])
        frame_indices.append(frame_idx)
        event_counts.append(event_count)

# 转换为NumPy数组
frame_indices = np.array(frame_indices)
event_counts = np.array(event_counts)

# 绘制原始折线图
plt.figure(figsize=(12, 6))
plt.plot(frame_indices, event_counts, label='Event Count per Frame', color='blue', alpha=0.5)
plt.xlabel('Frame Index (33ms per frame)')
plt.ylabel('Event Count')
plt.title('Event Counts Over Time with Valleys')
plt.grid(True)

# 方法1：固定阈值法
threshold = 3000  # 手动设置阈值
valleys_threshold = frame_indices[event_counts < threshold]
# plt.scatter(valleys_threshold, event_counts[event_counts < threshold], color='red', label='Valleys (Threshold < 3000)', s=50)

# 方法2：局部最小值法
valleys_local = []
window_size = 5  # 邻域窗口大小
for i in range(window_size, len(event_counts) - window_size):
    if event_counts[i] == min(event_counts[i-window_size:i+window_size+1]):
        valleys_local.append(frame_indices[i])
plt.scatter(valleys_local, event_counts[np.searchsorted(frame_indices, valleys_local)], color='green', label='Local Valleys', s=50)

# 方法3：滑动窗口最小值法
window_size = 10  # 调整窗口大小
valleys_sliding = []
for i in range(window_size, len(event_counts) - window_size):
    if event_counts[i] == min(event_counts[i-window_size:i+window_size+1]):
        valleys_sliding.append(frame_indices[i])
# plt.scatter(valleys_sliding, event_counts[np.searchsorted(frame_indices, valleys_sliding)], color='orange', label='Sliding Window Valleys', s=50)

# 方法4：一阶差分法（零交叉检测）
diff = np.diff(event_counts)
valleys_diff = []
for i in range(1, len(diff)):
    if diff[i-1] < 0 and diff[i] >= 0:  # 下降后上升
        valleys_diff.append(frame_indices[i])
# plt.scatter(valleys_diff, event_counts[np.searchsorted(frame_indices, valleys_diff)], color='purple', label='Diff Valleys', s=50)

# 添加图例
plt.legend()
plt.show()

# 输出波谷位置
print("Valleys detected by Threshold (< 3000):", valleys_threshold)
print("Valleys detected by Local Minima:", valleys_local)
print("Valleys detected by Sliding Window:", valleys_sliding)
print("Valleys detected by Difference Method:", valleys_diff)