import numpy as np
import matplotlib.pyplot as plt


# 从 TXT 文件提取数据，使用行号作为 frame_num
def extract_frame_data(file_path):
    positive_events = []
    negative_events = []
    total_events = []

    with open(file_path, 'r') as f:
        for line in f:
            # 分割每一行，提取 Positive 和 Negative Events 的数量
            parts = line.split(', ')
            pos_str = parts[2].split(': ')[1]  # Positive Events: <number>
            neg_str = parts[3].split(': ')[1].strip()  # Negative Events: <number>

            pos = int(pos_str)
            neg = int(neg_str)
            total = pos + neg

            positive_events.append(pos)
            negative_events.append(neg)
            total_events.append(total)

    # 使用行号作为帧号
    frames = np.arange(len(positive_events))
    return frames, np.array(positive_events), np.array(negative_events), np.array(total_events)


# 计算特征
def compute_features(total_events, positive_events, window_size=10):
    # 正负比例 (Positive / Total)
    pos_ratio = positive_events / (total_events + 1e-6)  # 避免除以零

    # 滑动窗口方差
    variance = np.zeros(len(total_events))
    half_window = window_size // 2
    for i in range(len(total_events)):
        start = max(0, i - half_window)
        end = min(len(total_events), i + half_window + 1)
        variance[i] = np.var(total_events[start:end])

    return pos_ratio, variance


# 分割动作
def segment_actions(pos_ratio, variance, total_events, min_action_length=10):
    # 平滑数据
    pos_ratio_smooth = np.convolve(pos_ratio, np.ones(5) / 5, mode='same')
    variance_smooth = np.convolve(variance, np.ones(5) / 5, mode='same')

    # 阈值设定（需根据数据调整）
    pos_ratio_threshold = 0.5  # 示例值，需分析信号灯闪烁比例
    variance_threshold = np.median(variance) * 0.5  # 示例值，动态调整

    # 标记静止帧
    static_frames = (pos_ratio_smooth < pos_ratio_threshold) & (variance_smooth < variance_threshold)

    # 找到动作边界
    boundaries = np.where(np.diff(static_frames.astype(int)))[0]
    action_segments = []
    start = 0
    for boundary in boundaries:
        if not static_frames[start]:  # 非静止期结束
            if boundary - start >= min_action_length:  # 确保动作段足够长
                action_segments.append((start, boundary))
        start = boundary + 1
    if start < len(total_events) and not static_frames[start]:
        if len(total_events) - start >= min_action_length:
            action_segments.append((start, len(total_events) - 1))

    return action_segments


# 主程序
file_path = 'D:\\eventVision\\collecting\\3_3\\result2\\filtered\\event_stats.txt'  # 修改为你的文件路径
frames, positive_events, negative_events, total_events = extract_frame_data(file_path)
pos_ratio, variance = compute_features(total_events, positive_events, window_size=10)
action_segments = segment_actions(pos_ratio, variance, total_events, min_action_length=10)

# 输出结果
print(f"分割出 {len(action_segments)} 个动作段：")
for i, (start, end) in enumerate(action_segments):
    duration_ms = (end - start + 1) * 33  # 每帧33ms
    print(f"Action {i + 1}: Frame {start} to {end}, Duration {duration_ms} ms")

# 可视化
plt.figure(figsize=(15, 5))
plt.plot(frames, total_events, label='Total Events')
for start, end in action_segments:
    plt.axvspan(start, end, color='green', alpha=0.3)
plt.xlabel('Frame Number (Line Number)')
plt.ylabel('Event Count')
plt.title('Action Segments')
plt.legend()
plt.grid(True)
plt.savefig('action_segments.png')
plt.show()

# 验证
if len(action_segments) != 9:
    print(f"警告：期望9个动作，实际得到 {len(action_segments)} 个，请调整阈值或窗口大小")