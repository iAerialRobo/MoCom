import dv
import numpy as np
import matplotlib.pyplot as plt

# 读取aedat4文件
file_path = 'D:/eventVision/collecting/3_3/1.aedat4'  # 你的文件路径
with dv.AedatFile(file_path) as f:
    events = []
    for i, event in enumerate(f['events']):
        # 显式转换所有字段
        timestamp = float(event.timestamp)
        x = int(event.x)  # 先转为int，检查范围
        y = int(event.y)
        polarity = bool(event.polarity)

        # 检查字段是否超出预期范围
        if x > 65535 or y > 65535:
            print(f"Warning at event {i}: x={x} or y={y} exceeds uint16 range")
        if x < 0 or y < 0:
            print(f"Warning at event {i}: x={x} or y={y} is negative")

        # 添加到列表
        events.append([timestamp, x, y, polarity])

# 转换为结构化数组
events = np.array(events, dtype=[('t', np.float64), ('x', np.uint16), ('y', np.uint16), ('p', bool)])

# 参数设置
time_window = 33000  # 33毫秒 = 33000微秒
spatial_window = 3  # 空间窗（像素）
min_neighbors = 5  # 最小邻居数
frame_duration = 33000  # 每帧33ms


# 滤波函数
def filter_small_events(events, time_window, spatial_window, min_neighbors):
    filtered_events = []
    for event in events:
        t = event['t']
        t_min = t - time_window
        t_max = t + time_window
        x_min = event['x'] - spatial_window
        x_max = event['x'] + spatial_window
        y_min = event['y'] - spatial_window
        y_max = event['y'] + spatial_window

        neighbors = events[
            (events['t'] >= t_min) & (events['t'] <= t_max) &
            (events['x'] >= x_min) & (events['x'] <= x_max) &
            (events['y'] >= y_min) & (events['y'] <= y_max)
            ]

        if len(neighbors) >= min_neighbors:
            filtered_events.append(event)

    return np.array(filtered_events, dtype=events.dtype)


# 应用滤波
filtered_events = filter_small_events(events, time_window, spatial_window, min_neighbors)


# 按33ms分割成帧并计算每帧事件数
def get_frame_event_counts(events, frame_duration):
    # 获取时间范围
    t_min = events['t'].min()
    t_max = events['t'].max()
    num_frames = int(np.ceil((t_max - t_min) / frame_duration))  # 总帧数

    # 初始化帧事件计数
    frame_counts = np.zeros(num_frames, dtype=int)

    # 对每个事件，计算它属于哪一帧
    for event in events:
        frame_idx = int((event['t'] - t_min) / frame_duration)
        if frame_idx < num_frames:  # 防止越界
            frame_counts[frame_idx] += 1

    return frame_counts


# 获取滤波后每帧事件数
filtered_frame_counts = get_frame_event_counts(filtered_events, frame_duration)

# 生成时间轴（帧索引）
time_frames = np.arange(len(filtered_frame_counts))

# 绘制每帧事件数时间变化图
plt.figure(figsize=(10, 6))
plt.plot(time_frames, filtered_frame_counts, label='Filtered Event Count per Frame', color='red')
plt.xlabel('Frame Index (33ms per frame)')
plt.ylabel('Event Count')
plt.title('Filtered Event Counts Over Time (33ms Frames)')
plt.grid(True)
plt.legend()
plt.show()

# 可选：保存每帧事件数到文件
np.save('filtered_frame_counts.npy', filtered_frame_counts)