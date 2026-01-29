import numpy as np
import matplotlib.pyplot as plt


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


# 计算特征
def compute_features(total_events, positive_events, window_size=10):
    pos_ratio = positive_events / (total_events + 1e-6)  # 避免除以零
    variance = np.zeros(len(total_events))
    half_window = window_size // 2
    for i in range(len(total_events)):
        start = max(0, i - half_window)
        end = min(len(total_events), i + half_window + 1)
        variance[i] = np.var(total_events[start:end])

    return pos_ratio, variance


# 分割动作
def segment_actions(pos_ratio, variance, total_events, min_action_length=10):
    pos_ratio_smooth = np.convolve(pos_ratio, np.ones(5) / 5, mode='same')
    variance_smooth = np.convolve(variance, np.ones(5) / 5, mode='same')

    pos_ratio_threshold = 0.5  # 示例值，需分析信号灯闪烁比例
    variance_threshold = np.median(variance) * 0.5  # 示例值，动态调整

    static_frames = (pos_ratio_smooth < pos_ratio_threshold) & (variance_smooth < variance_threshold)
    boundaries = np.where(np.diff(static_frames.astype(int)))[0]
    action_segments = []
    start = 0
    for boundary in boundaries:
        if not static_frames[start]:  # 非静止期结束
            if boundary - start >= min_action_length:
                action_segments.append((start, boundary))
        start = boundary + 1
    if start < len(total_events) and not static_frames[start]:
        if len(total_events) - start >= min_action_length:
            action_segments.append((start, len(total_events) - 1))

    return action_segments


# 后处理1：过滤小于1秒的动作
def filter_short_actions(action_segments, min_duration_frames=30):
    filtered_segments = []
    for start, end in action_segments:
        duration_frames = end - start + 1
        if duration_frames >= min_duration_frames:  # 至少30帧（约1秒）
            filtered_segments.append((start, end))
    return filtered_segments


# 后处理2：合并相邻的动作
def merge_close_actions(action_segments, max_gap_frames=10):
    if not action_segments:
        return []

    merged_segments = []
    current_start, current_end = action_segments[0]

    for i in range(1, len(action_segments)):
        next_start, next_end = action_segments[i]
        gap = next_start - current_end - 1  # 计算两段之间的间隔帧数

        if gap <= max_gap_frames:  # 如果间隔小于等于阈值，合并
            current_end = next_end
        else:  # 否则结束当前动作，开启新动作
            merged_segments.append((current_start, current_end))
            current_start = next_start
            current_end = next_end

    merged_segments.append((current_start, current_end))
    return merged_segments


# 后处理3：过滤小于3秒的动作
def filter_less_than_3s_actions(action_segments, min_duration_frames_3s=91):
    filtered_segments = []
    for start, end in action_segments:
        duration_frames = end - start + 1
        if duration_frames >= min_duration_frames_3s:  # 至少91帧（约3秒）
            filtered_segments.append((start, end))
    return filtered_segments


# 主程序
# file_path = 'D:\\eventVision\\collecting\\3_3\\result2\\filtered\\event_stats.txt'
# file_path = 'D:\\eventVision\\collecting\\3_3\\final\\event_stats_2_5.txt'
# file_path = 'D:\\eventVision\\collecting\\3_3\\final\\event_stats_3_0.txt'
# file_path = 'D:\\eventVision\\collecting\\3_3\\final\\event_stats_3_5.txt'
# file_path = 'D:\\eventVision\\collecting\\4_16\\1\\UAV_event_statistics.txt'
file_path = 'D:\\eventVision\\collecting\\4_16\\result1\\filtered\\event_stats.txt'
# GT 结果（去掉第一个 1）
#gt_sequence_3_5 = [165,245,    340,585,   683,915,    1005,1240,   1330,1555,    1655,1880,  1975,2220,   2315,2566,  2660,2799]
#num_actions_3_5 = 9
#gt_sequence_3_0 = [230,320,    405, 640,   730, 950,  1035, 1265,  1355, 1570,   1645,1873,  1954,2200,   2285,2525, 2610, 2750]
#num_actions_3_0 = 9
#gt_sequence_2_5 = [100,230,     285,540,    600,830,   900,1120,     1185,1410,    1470,1710,   1770,2025,  2084,2320,  2395,2540]
#num_actions_2_5 = 9
gt_result1 = [165,245,    340,585,   683,915,    1005,1240,   1330,1555,    1655,1880,  1975,2220,   2315,2566,  2660,2799]
num_actions_3_5 = 8
gt_result2 = [230,320,    405, 640,   730, 950,  1035, 1265,  1355, 1570,   1645,1873,  1954,2200,   2285,2525, 2610, 2750]
num_actions_3_0 = 8
gt_result3 = [100,230,     285,540,    600,830,   900,1120,     1185,1410,    1470,1710,   1770,2025,  2084,2320,  2395,2540]
num_actions_2_5 = 8


frames, positive_events, negative_events, total_events = extract_frame_data(file_path)
pos_ratio, variance = compute_features(total_events, positive_events, window_size=10)
action_segments = segment_actions(pos_ratio, variance, total_events, min_action_length=10)

# 后处理1：过滤小于1秒的动作
min_duration_frames = 30  # 1秒 ≈ 30帧 (1000ms / 33ms ≈ 30.3)
filtered_action_segments = filter_short_actions(action_segments, min_duration_frames)

# 后处理2：合并相邻动作
max_gap_frames = 10  # 最大间隔10帧（约330ms），可调整
merged_action_segments = merge_close_actions(filtered_action_segments, max_gap_frames)

# 后处理3：过滤小于3秒的动作
min_duration_frames_3s = 91  # 3秒 ≈ 91帧 (3000ms / 33ms ≈ 90.9)
final_action_segments = filter_less_than_3s_actions(merged_action_segments, min_duration_frames_3s)

# 输出结果
print(f"原始分割出 {len(action_segments)} 个动作段")
print(f"过滤小于1秒后剩余 {len(filtered_action_segments)} 个动作段")
print(f"合并相邻动作后剩余 {len(merged_action_segments)} 个动作段")
print(f"过滤小于3秒后剩余 {len(final_action_segments)} 个动作段：")
for i, (start, end) in enumerate(final_action_segments):
    duration_ms = (end - start + 1) * 33  # 每帧33ms
    print(f"Action {i + 1}: Frame {start} to {end}, Duration {duration_ms} ms")

# 可视化
plt.figure(figsize=(15, 5))
plt.plot(frames, total_events, label='Total Events')
for start, end in final_action_segments:
    plt.axvspan(start, end, color='green', alpha=0.3)
plt.xlabel('Frame Number (Line Number)')
plt.ylabel('Event Count')
plt.title('Final Action Segments (Duration >= 3s, Merged if Gap <= 330ms)')
plt.legend()
plt.grid(True)
plt.savefig('final_action_segments.png')
plt.show()

# 验证
if len(final_action_segments) != 8:
    print(f"警告：期望9个动作，实际得到 {len(final_action_segments)} 个，请调整阈值或合并间隔")