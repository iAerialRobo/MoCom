import numpy as np
import matplotlib.pyplot as plt

# 假设数据保存在文件 'event_data.txt' 中
file_path = 'D:\\eventVision\\collecting\\3_3\\3\\UAV_event_statistics.txt'

# 读取数据
frame_indices = []
event_counts = []
with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        frame_idx = int(parts[0])      # 第一列：帧索引
        event_count = int(parts[3])    # 第四列：事件总数
        frame_indices.append(frame_idx)
        event_counts.append(event_count)#

# c = 500
# frame_indices = frame_indices[0:c]
# event_counts = event_counts[0:c]
# 转换为NumPy数组
frame_indices = np.array(frame_indices)
event_counts = np.array(event_counts)

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(frame_indices, event_counts, label='Event Count per Frame', color='blue')
plt.xlabel('Frame Index (33ms per frame)')
plt.ylabel('Event Count')
plt.title('Event Counts Over Time')
plt.grid(True)
plt.legend()
plt.show()