import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

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

# HMM 分割动作
def segment_actions_hmm(total_events, n_components=2, min_action_length=10):
    # 重塑数据为 (n_samples, n_features)，这里仅使用 total_events 作为特征
    X = total_events.reshape(-1, 1)

    # 初始化并训练 HMM 模型
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
    model.fit(X)

    # 预测隐藏状态序列
    hidden_states = model.predict(X)

    # 分割动作段：假设状态 1 为“动作”，状态 0 为“静止”
    # （状态编号可能不同，需根据均值判断）
    state_means = model.means_.flatten()
    action_state = np.argmax(state_means)  # 假设事件数多的状态为“动作”
    static_state = np.argmin(state_means)  # 事件数少的状态为“静止”

    # 找到动作边界
    action_frames = (hidden_states == action_state)
    boundaries = np.where(np.diff(action_frames.astype(int)))[0]

    action_segments = []
    start = 0
    for boundary in boundaries:
        if action_frames[start]:  # 当前段是动作
            if boundary - start >= min_action_length:
                action_segments.append((start, boundary))
        start = boundary + 1
    # 处理最后一个段
    if start < len(total_events) and action_frames[start]:
        if len(total_events) - start >= min_action_length:
            action_segments.append((start, len(total_events) - 1))

    return action_segments, hidden_states, model

# 后处理：与你的方法保持一致
def filter_short_actions(action_segments, min_duration_frames=30):
    filtered_segments = []
    for start, end in action_segments:
        duration_frames = end - start + 1
        if duration_frames >= min_duration_frames:
            filtered_segments.append((start, end))
    return filtered_segments

def merge_close_actions(action_segments, max_gap_frames=10):
    if not action_segments:
        return []

    merged_segments = []
    current_start, current_end = action_segments[0]

    for i in range(1, len(action_segments)):
        next_start, next_end = action_segments[i]
        gap = next_start - current_end - 1

        if gap <= max_gap_frames:
            current_end = next_end
        else:
            merged_segments.append((current_start, current_end))
            current_start = next_start
            current_end = next_end

    merged_segments.append((current_start, current_end))
    return merged_segments

def filter_less_than_3s_actions(action_segments, min_duration_frames_3s=91):
    filtered_segments = []
    for start, end in action_segments:
        duration_frames = end - start + 1
        if duration_frames >= min_duration_frames_3s:
            filtered_segments.append((start, end))
    return filtered_segments

# 主程序
file_path = 'D:\\eventVision\\collecting\\3_3\\result3\\filtered\\event_stats.txt'
frames, positive_events, negative_events, total_events = extract_frame_data(file_path)

# HMM 分割
action_segments_hmm, hidden_states, hmm_model = segment_actions_hmm(total_events, n_components=2, min_action_length=10)

# 后处理1：过滤小于1秒的动作
filtered_action_segments_hmm = filter_short_actions(action_segments_hmm, min_duration_frames=30)

# 后处理2：合并相邻动作
merged_action_segments_hmm = merge_close_actions(filtered_action_segments_hmm, max_gap_frames=10)

# 后处理3：过滤小于3秒的动作
final_action_segments_hmm = filter_less_than_3s_actions(merged_action_segments_hmm, min_duration_frames_3s=91)

# 输出结果
print(f"HMM 原始分割出 {len(action_segments_hmm)} 个动作段")
print(f"HMM 过滤小于1秒后剩余 {len(filtered_action_segments_hmm)} 个动作段")
print(f"HMM 合并相邻动作后剩余 {len(merged_action_segments_hmm)} 个动作段")
print(f"HMM 过滤小于3秒后剩余 {len(final_action_segments_hmm)} 个动作段：")
for i, (start, end) in enumerate(final_action_segments_hmm):
    duration_ms = (end - start + 1) * 33  # 每帧33ms
    print(f"Action {i + 1}: Frame {start} to {end}, Duration {duration_ms} ms")

# 可视化
plt.figure(figsize=(15, 5))
plt.plot(frames, total_events, label='Total Events')
for start, end in final_action_segments_hmm:
    plt.axvspan(start, end, color='orange', alpha=0.3)
plt.xlabel('Frame Number (Line Number)')
plt.ylabel('Event Count')
plt.title('HMM Action Segments (Duration >= 3s, Merged if Gap <= 330ms)')
plt.legend()
plt.grid(True)
plt.savefig('hmm_action_segments.png')
plt.show()

# 验证
if len(final_action_segments_hmm) != 9:
    print(f"警告：期望9个动作，实际得到 {len(final_action_segments_hmm)} 个，请调整HMM参数（例如n_components）")