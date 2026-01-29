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
    static_frames = (pos_ratio_smooth < pos_ratio_threshold) & (variance_smooth < variance_threshold)
    boundaries = np.where(np.diff(static_frames.astype(int)))[0]
    action_segments, start = [], 0
    for boundary in boundaries:
        if not static_frames[start] and boundary - start >= min_action_length:
            action_segments.append((start, boundary))
        start = boundary + 1
    if start < len(total_events) and not static_frames[start]:
        if len(total_events) - start >= min_action_length:
            action_segments.append((start, len(total_events) - 1))
    return action_segments

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
file_path = r'E:\IEEE_tro_compareExperiment\11_13\3\filtered\event_stats.txt'

frames, pos, neg, total = extract_frame_data(file_path)

start_frame = 700  # 3 filtered
# frames = frames[start_frame:]
pos = pos[start_frame:]
neg = neg[start_frame:]
total = total[start_frame:]
frames = np.arange(1, len(pos) + 1)

total_smooth = gaussian_filter1d(total, sigma=5)
pos_smooth = gaussian_filter1d(pos, sigma=5)

pos_ratio, variance = compute_features(total_smooth, pos_smooth)

segments = segment_actions(pos_ratio, variance, total_smooth)
segments = filter_short_actions(segments, 30)
segments = merge_close_actions(segments, 10)
segments = filter_long_enough(segments, 91)

print(f"检测到 {len(segments)} 个动作段：")
for i, (s, e) in enumerate(segments):
    print(f"Action {i+1}: Frame {s}-{e}, Duration {(e-s+1)*33} ms")

plt.figure(figsize=(15, 5))
plt.plot(frames, total_smooth, color='red', label='Smoothed Total Events')
for s, e in segments:
    plt.axvspan(s, e, color='green', alpha=0.3)
plt.xlabel('Frame')
plt.ylabel('Event Count')
plt.title('Detected Action Segments')
plt.legend()
plt.grid(True)
plt.savefig('final_action_segments_simplified.png')
plt.show()
