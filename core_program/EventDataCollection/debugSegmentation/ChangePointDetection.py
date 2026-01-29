import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

# 从 TXT 文件提取数据，使用行号作为 frame_num
def extract_frame_data(file_path):
    positive_events = []
    negative_events = []
    total_events = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split(', ')
            pos_str = parts[2].split(': ')[1]  # Positive Events: <number>
            neg_str = parts[3].split(': ')[1].strip()  # Negative Events: <number>

            pos = int(pos_str)
            neg = int(neg_str)
            total = pos + neg

            positive_events.append(pos)
            negative_events.append(neg)
            total_events.append(total)

    frames = np.arange(len(positive_events))
    return frames, np.array(positive_events), np.array(negative_events), np.array(total_events)

# CPD 分割动作（优化版）
def segment_actions_cpd(total_events, penalty=10000, smooth_window=50, min_segment_length=30):
    # 平滑数据以减少噪声
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        total_events_smooth = np.convolve(total_events, kernel, mode='same')
    else:
        total_events_smooth = total_events

    # 使用 Pelt 算法检测变化点
    algo = rpt.Pelt(model="l2").fit(total_events_smooth)
    change_points = algo.predict(pen=penalty)  # penalty 控制变化点数量

    # 将变化点转换为动作段，并加入最小长度约束
    action_segments = []
    start = 0
    for cp in change_points[:-1]:  # 最后一个是序列末尾，不作为边界
        if cp - start >= min_segment_length:  # 仅保留长度足够的段
            action_segments.append((start, cp - 1))
        start = cp
    # 处理最后一段
    if len(total_events) - start >= min_segment_length:
        action_segments.append((start, len(total_events) - 1))

    return action_segments, change_points, total_events_smooth

# 主程序
file_path = 'D:\\eventVision\\collecting\\3_3\\result3\\filtered\\event_stats.txt'
frames, positive_events, negative_events, total_events = extract_frame_data(file_path)

# CPD 分割（优化参数）
penalty_value = 10000      # 大幅增加 penalty
smooth_window = 50         # 增强平滑效果
min_segment_length = 30    # 最小段长约1秒（30帧 * 33ms = 990ms）
action_segments_cpd, change_points, total_events_smooth = segment_actions_cpd(
    total_events, penalty=penalty_value, smooth_window=smooth_window, min_segment_length=min_segment_length
)

# 输出结果
print(f"数据总帧数: {len(total_events)}")
print(f"CPD 分割出 {len(action_segments_cpd)} 个动作段：")
for i, (start, end) in enumerate(action_segments_cpd):
    duration_ms = (end - start + 1) * 33  # 每帧33ms
    print(f"Action {i + 1}: Frame {start} to {end}, Duration {duration_ms} ms")

# 可视化
plt.figure(figsize=(15, 5))
plt.plot(frames, total_events, label='Total Events (Raw)', alpha=0.5)
plt.plot(frames, total_events_smooth, label='Total Events (Smoothed)', color='blue')
for start, end in action_segments_cpd:
    plt.axvspan(start, end, color='purple', alpha=0.3)
plt.xlabel('Frame Number (Line Number)')
plt.ylabel('Event Count')
plt.title(f'CPD Action Segments (Penalty={penalty_value}, Smooth Window={smooth_window}, Min Length={min_segment_length} frames)')
plt.legend()
plt.grid(True)
plt.savefig('cpd_action_segments_optimized.png')
plt.show()

# 验证
if len(action_segments_cpd) != 9:
    print(f"警告：期望9个动作，实际得到 {len(action_segments_cpd)} 个，请调整penalty值或smooth_window（当前为 {penalty_value}, {smooth_window}）")