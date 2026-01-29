import numpy as np
import matplotlib.pyplot as plt

# 假设数据已读取
file_path = 'D:\\eventVision\\collecting\\3_3\\1\\UAV_event_statistics.txt'

frame_indices = []
event_counts = []
with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        frame_idx = int(parts[0])
        event_count = int(parts[3])
        frame_indices.append(frame_idx)
        event_counts.append(event_count)

frame_indices = np.array(frame_indices)
event_counts = np.array(event_counts)

# 绘制原始折线图
plt.figure(figsize=(12, 6))
plt.plot(frame_indices, event_counts, label='Event Count per Frame', color='blue', alpha=0.5)
plt.xlabel('Frame Index (33ms per frame)')
plt.ylabel('Event Count')
plt.title('Event Counts with Bottom Valley Points')
plt.grid(True)

# 参数设置
window_size = 5      # 局部最小值窗口大小
bottom_threshold_factor = 1.5  # 底部定义为全局最小值的1.5倍
flat_threshold = 100  # 平坦区域变化阈值

# 找到全局最小值
global_min = np.min(event_counts)
bottom_threshold = global_min * bottom_threshold_factor  # 定义底部阈值

# 检测底部平坦区域的最低值
valleys_bottom = []
i = 0
while i < len(event_counts) - 1:
    # 只在底部区域内检测
    if event_counts[i] <= bottom_threshold:
        if abs(event_counts[i+1] - event_counts[i]) <= flat_threshold:
            # 找到平坦区域的起始点
            start_idx = i
            while i < len(event_counts) - 1 and event_counts[i] <= bottom_threshold and abs(event_counts[i+1] - event_counts[i]) <= flat_threshold:
                i += 1
            end_idx = i
            # 找到平坦区域的最小值和对应帧
            min_value = min(event_counts[start_idx:end_idx + 1])
            min_frame = frame_indices[start_idx + np.argmin(event_counts[start_idx:end_idx + 1])]
            valleys_bottom.append(min_frame)
        else:
            i += 1
    else:
        i += 1

# 过滤，确保最低点是局部最小值（仅在底部区域内）
valleys_final = []
for valley_frame in valleys_bottom:
    idx = np.searchsorted(frame_indices, valley_frame)
    if idx - window_size >= 0 and idx + window_size < len(event_counts):
        if event_counts[idx] <= min(event_counts[idx-window_size:idx+window_size+1]):
            valleys_final.append(valley_frame)

# 去除重复点（取第一个最低值）
valleys_final = sorted(list(dict.fromkeys(valleys_final)))

# 绘制波谷
plt.scatter(valleys_final, event_counts[np.searchsorted(frame_indices, valleys_final)],
            color='red', label='Bottom Valley Points', s=50)

# 添加图例
plt.legend()
plt.show()

# 输出波谷位置
print("Detected Bottom Valley Points at frames:", valleys_final)
print(f"Global Minimum: {global_min}, Bottom Threshold: {bottom_threshold}")