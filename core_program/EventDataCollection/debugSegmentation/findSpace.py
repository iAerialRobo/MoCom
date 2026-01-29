import numpy as np
import matplotlib.pyplot as plt


# 从 TXT 文件提取数据
def extract_frame_data(file_path):
    negative_events = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split(', ')
            neg_str = parts[3].split(': ')[1].strip()  # Negative Events: <number>
            neg = int(neg_str)
            negative_events.append(neg)
    frames = np.arange(len(negative_events))
    return frames, np.array(negative_events)


# 检测信号区域
def detect_signal_regions(negative_events, threshold=500, min_duration_frames=50, smooth_window=5):
    # 平滑信号
    smooth_signal = np.convolve(negative_events, np.ones(smooth_window) / smooth_window, mode='same')

    # 阈值分割
    above_threshold = smooth_signal > threshold
    boundaries = np.where(np.diff(above_threshold.astype(int)))[0]

    # 提取区域
    regions = []
    for i in range(0, len(boundaries), 2):
        if i + 1 < len(boundaries):
            start = boundaries[i] + 1
            end = boundaries[i + 1]
            duration = end - start + 1
            if duration >= min_duration_frames:  # 确保区域足够长
                regions.append((start, end))

    return regions


# 主程序
file_path = 'D:\\eventVision\\collecting\\3_3\\result\\filtered\\event_stats.txt'
frames, negative_events = extract_frame_data(file_path)

# 检测信号区域
threshold = 500  # 负事件数量阈值，根据图像调整
min_duration_frames = 50  # 最小持续帧数，约1.5秒
smooth_window = 5  # 平滑窗口大小
signal_regions = detect_signal_regions(negative_events, threshold, min_duration_frames, smooth_window)

# 输出结果
print(f"检测到 {len(signal_regions)} 个信号区域：")
for i, (start, end) in enumerate(signal_regions):
    duration_ms = (end - start + 1) * 33  # 每帧33ms
    print(f"Region {i + 1}: Frame {start} to {end}, Duration {duration_ms} ms")

# 可视化
plt.figure(figsize=(15, 5))
plt.plot(frames, negative_events, label='Negative Events')
for start, end in signal_regions:
    plt.axvspan(start, end, color='red', alpha=0.3)
plt.xlabel('Frame Number (Line Number)')
plt.ylabel('Number of Negative Events')
plt.title('Detected Signal Regions (Negative Events)')
plt.legend()
plt.grid(True)
plt.savefig('detected_signal_regions.png')
plt.show()