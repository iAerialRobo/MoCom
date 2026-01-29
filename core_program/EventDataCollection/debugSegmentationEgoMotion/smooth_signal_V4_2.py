import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def extract_frame_data(file_path):
    positive_events, negative_events, total_events = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split(', ')
            pos = int(parts[2].split('=')[1])
            neg = int(parts[3].split('=')[1].strip())
            total = pos + neg
            positive_events.append(pos)
            negative_events.append(neg)
            total_events.append(total)
    frames = np.arange(len(positive_events))
    return frames, np.array(positive_events), np.array(negative_events), np.array(total_events)

def compute_features(total_events_smoothed, positive_events_smoothed, window_size=10):
    pos_ratio = positive_events_smoothed / (total_events_smoothed + 1e-6)
    variance = np.zeros(len(total_events_smoothed))
    half_window = window_size // 2
    for i in range(len(total_events_smoothed)):
        start = max(0, i - half_window)
        end = min(len(total_events_smoothed), i + half_window + 1)
        variance[i] = np.var(total_events_smoothed[start:end])
    return pos_ratio, variance

def segment_actions(pos_ratio, variance, total_events, min_action_length=10):
    pos_ratio_smooth = np.convolve(pos_ratio, np.ones(5) / 5, mode='same')
    variance_smooth = np.convolve(variance, np.ones(5) / 5, mode='same')

    pos_ratio_threshold = 0.99
    variance_threshold = np.median(variance) * 0.99
    # (pos_ratio_smooth < pos_ratio_threshold) &
    static_frames = (variance_smooth < variance_threshold) & (total_events < 2000)
    boundaries = np.where(np.diff(static_frames.astype(int)))[0]

    action_segments, start = [], 0
    for boundary in boundaries:
        if not static_frames[start] and boundary - start >= min_action_length:
            action_segments.append((start, boundary))
        start = boundary + 1
    if start < len(total_events) and not static_frames[start]:
        if len(total_events) - start >= min_action_length:
            action_segments.append((start, len(total_events) - 1))

    # 👇返回 pos_ratio_smooth, variance_smooth 和 threshold 用于可视化
    return action_segments, pos_ratio_smooth, variance_smooth, pos_ratio_threshold, variance_threshold

def filter_short_actions(segments, min_frames=30):
    return [(s, e) for s, e in segments if e - s + 1 >= min_frames]

def merge_close_actions(segments, max_gap=10):
    if not segments:
        return []
    merged = []
    cs, ce = segments[0]
    for ns, ne in segments[1:]:
        if ns - ce - 1 <= max_gap:
            ce = ne
        else:
            merged.append((cs, ce))
            cs, ce = ns, ne
    merged.append((cs, ce))
    return merged

def filter_long_enough(segments, min_frames=91):
    return [(s, e) for s, e in segments if e - s + 1 >= min_frames]


# === 主流程 ===
# file_path = r'E:\IEEE_tro_compareExperiment\11_13\2\filtered\event_stats.txt'
file_path = r'E:\IEEE_tro_compareExperiment\11_14\1\png\filtered\event_stats.txt'
# file_path = r'E:\IEEE_tro_compareExperiment\11_13\2\filtered\event_stats.txt'
frames, pos, neg, total = extract_frame_data(file_path)

start_frame = 505
pos = pos[start_frame:]
neg = neg[start_frame:]
total = total[start_frame:]
frames = np.arange(1, len(pos) + 1)

total_smooth = gaussian_filter1d(total, sigma=5)
pos_smooth = gaussian_filter1d(pos, sigma=5)

pos_ratio, variance = compute_features(total_smooth, pos_smooth)
initial_segments, pos_ratio_smooth, variance_smooth, pos_ratio_threshold, variance_threshold = segment_actions(
    pos_ratio, variance, total_smooth
)

# === 可视化初始动作段 ===
# fig, ax1 = plt.subplots(figsize=(15, 6))

# 左边y轴：事件总数
# ax1.plot(frames, total_smooth, color='red', label='Smoothed Total Events')
# ax1.set_xlabel('Frame')
# ax1.set_ylabel('Total Events', color='red')
# ax1.tick_params(axis='y', labelcolor='red')
# ax1.grid(True, linestyle='--', alpha=0.5)

# 在 total 图中标注初始动作段
# for s, e in initial_segments:
#     ax1.axvspan(s, e, color='green', alpha=0.3)

# 右边y轴：pos_ratio_smooth 和 variance_smooth
# ax2 = ax1.twinx()
# ax2.plot(frames, pos_ratio_smooth, color='blue', label='Pos Ratio (Smoothed)', alpha=0.7)
# ax2.plot(frames, variance_smooth, color='orange', label='Variance (Smoothed)', alpha=0.7)
# ax2.axhline(variance_threshold, color='purple', linestyle='--', label=f'Variance Threshold = {variance_threshold:.2f}')
# ax2.set_ylabel('Pos Ratio / Variance', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

# 合并图例
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines + lines2, labels + labels2, loc='upper right')

# plt.title('Initial Action Segments with Pos Ratio and Variance')
# plt.tight_layout()
# plt.savefig('initial_action_segments_with_ratio_variance.png')
# plt.show()

# 后续处理
filtered_short_segments = filter_short_actions(initial_segments, 30)
merged_segments = merge_close_actions(filtered_short_segments, 30)

# === 可视化合并后的动作段 ===
#fig, ax1 = plt.subplots(figsize=(15, 6))

# ax1.plot(frames, total_smooth, color='red', label='Smoothed Total Events')
# ax1.set_xlabel('Frame')
# ax1.set_ylabel('Total Events', color='red')
# ax1.tick_params(axis='y', labelcolor='red')
# ax1.grid(True, linestyle='--', alpha=0.5)

# for s, e in merged_segments:
#     ax1.axvspan(s, e, color='green', alpha=0.3)

# ax2 = ax1.twinx()
# ax2.plot(frames, pos_ratio_smooth, color='blue', label='Pos Ratio (Smoothed)', alpha=0.7)
# ax2.plot(frames, variance_smooth, color='orange', label='Variance (Smoothed)', alpha=0.7)
# ax2.axhline(variance_threshold, color='purple', linestyle='--', label=f'Variance Threshold = {variance_threshold:.2f}')
# ax2.set_ylabel('Pos Ratio / Variance', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines + lines2, labels + labels2, loc='upper right')

# plt.title('Merged Action Segments with Pos Ratio and Variance')
# plt.tight_layout()
# plt.savefig('merged_action_segments_with_ratio_variance.png')
# plt.show()

# 继续后续处理
segments = filter_long_enough(merged_segments, 70)

print(f"检测到 {len(segments)} 个动作段：")
for i, (s, e) in enumerate(segments):
    print(f"Action {i+1}: Frame {s}-{e}, Duration {(e-s+1)*33} ms")

# === 可视化最终段 ===
fig, ax1 = plt.subplots(figsize=(15, 6))

ax1.plot(frames, total, color='red', label='Smoothed Total Events')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Total Events', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True, linestyle='--', alpha=0.5)

for s, e in segments:
    ax1.axvspan(s, e, color='green', alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(frames, pos_ratio_smooth, color='blue', label='Pos Ratio (Smoothed)', alpha=0.7)
ax2.plot(frames, variance_smooth, color='orange', label='Variance (Smoothed)', alpha=0.7)
ax2.axhline(variance_threshold, color='purple', linestyle='--', label=f'Variance Threshold = {variance_threshold:.2f}')
ax2.set_ylabel('Pos Ratio / Variance', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Detected Action Segments with Pos Ratio and Variance')
plt.tight_layout()
plt.savefig('final_action_segments_with_ratio_variance.png')
plt.show()