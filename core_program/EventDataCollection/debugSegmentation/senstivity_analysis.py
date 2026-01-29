import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


# 分割动作（修改为接受阈值参数）
def segment_actions(pos_ratio, variance, total_events, pos_threshold, var_mult, min_action_length=10):
    pos_ratio_smooth = np.convolve(pos_ratio, np.ones(5) / 5, mode='same')
    variance_smooth = np.convolve(variance, np.ones(5) / 5, mode='same')

    variance_threshold = np.median(variance) * var_mult

    static_frames = (pos_ratio_smooth < pos_threshold) & (variance_smooth < variance_threshold)
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


# 后处理 1：过滤小于 1 秒的动作
def filter_short_actions(action_segments, min_duration_frames=30):
    filtered_segments = []
    for start, end in action_segments:
        duration_frames = end - start + 1
        if duration_frames >= min_duration_frames:  # 至少 30 帧（约 1 秒）
            filtered_segments.append((start, end))
    return filtered_segments


# 后处理 2：合并相邻的动作
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


# 后处理 3：过滤小于 3 秒的动作
def filter_less_than_3s_actions(action_segments, min_duration_frames_3s=91):
    filtered_segments = []
    for start, end in action_segments:
        duration_frames = end - start + 1
        if duration_frames >= min_duration_frames_3s:  # 至少 91 帧（约 3 秒）
            filtered_segments.append((start, end))
    return filtered_segments


# 计算单个区间的IOU
def compute_iou(pred_start, pred_end, gt_start, gt_end):
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start + 1)
    union = (pred_end - pred_start + 1) + (gt_end - gt_start + 1) - intersection
    return intersection / union if union > 0 else 0


# 计算中心点误差
def compute_center_error(pred_start, pred_end, gt_start, gt_end):
    pred_center = (pred_start + pred_end) / 2
    gt_center = (gt_start + gt_end) / 2
    return abs(pred_center - gt_center)


# 计算整体指标（假设预测和GT数量相同，一一对应）
def compute_metrics(pred_segments, gt_segments):
    if len(pred_segments) != len(gt_segments):
        return 0.0, float('inf')  # 如果数量不匹配，返回低IOU和高误差

    avg_iou = 0.0
    avg_center_error = 0.0
    for pred, gt in zip(pred_segments, gt_segments):
        pred_start, pred_end = pred
        gt_start, gt_end = gt
        avg_iou += compute_iou(pred_start, pred_end, gt_start, gt_end)
        avg_center_error += compute_center_error(pred_start, pred_end, gt_start, gt_end)

    avg_iou /= len(pred_segments)
    avg_center_error /= len(pred_segments)
    return avg_iou, avg_center_error


# 主程序：敏感性分析
file_path = 'D:\\eventVision\\collecting\\3_3\\final\\event_stats_3_0.txt'
gt_sequence = [230, 320, 405, 640, 730, 950, 1035, 1265, 1355, 1570, 1645, 1873, 1954, 2200, 2285, 2525, 2610, 2750]
gt_segments = [(gt_sequence[i], gt_sequence[i + 1]) for i in range(0, len(gt_sequence), 2)]
num_gt_actions = 9

# 提取数据（只需一次）
frames, positive_events, negative_events, total_events = extract_frame_data(file_path)
pos_ratio, variance = compute_features(total_events, positive_events, window_size=10)

# 参数采样
pos_thresholds = np.linspace(0.48, 0.53, 5)  # A: 5 points
var_mults = np.linspace(0.42, 0.9, 20)  # B: 20 points

# 创建网格
pos_grid, var_grid = np.meshgrid(pos_thresholds, var_mults)
iou_grid = np.zeros_like(pos_grid)
center_error_grid = np.zeros_like(pos_grid)

# 循环计算
for i, pos_th in enumerate(pos_thresholds):
    for j, var_m in enumerate(var_mults):
        action_segments = segment_actions(pos_ratio, variance, total_events, pos_th, var_m, min_action_length=10)

        # 后处理
        min_duration_frames = 30
        filtered_action_segments = filter_short_actions(action_segments, min_duration_frames)

        max_gap_frames = 10
        merged_action_segments = merge_close_actions(filtered_action_segments, max_gap_frames)

        min_duration_frames_3s = 91
        final_action_segments = filter_less_than_3s_actions(merged_action_segments, min_duration_frames_3s)

        # 计算指标
        avg_iou, avg_center_error = compute_metrics(final_action_segments, gt_segments)
        iou_grid[j, i] = avg_iou  # 注意meshgrid的索引：行对应var_mults，列对应pos_thresholds
        center_error_grid[j, i] = avg_center_error


# 打印X,Y,Z格式的数据到控制台（X=pos_threshold, Y=var_mult, Z=iou）
print("X,Y,Z")
for i in range(iou_grid.shape[0]):  # 行：var_mults
    for j in range(iou_grid.shape[1]):  # 列：pos_thresholds
        print(f"{pos_grid[i, j]},{var_grid[i, j]},{iou_grid[i, j]}")


# 绘制3D柱状图 - Average IOU
fig_iou = plt.figure(figsize=(12, 8))
ax_iou = fig_iou.add_subplot(111, projection='3d')
xpos = pos_grid.ravel()
ypos = var_grid.ravel()
zpos = np.zeros_like(xpos)
dx = 0.01  # 根据A步长调整
dy = 0.02  # 根据B步长调整
dz = iou_grid.ravel()
ax_iou.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
ax_iou.set_xlabel('pos_ratio_threshold (A)')
ax_iou.set_ylabel('variance_multiplier (B)')
ax_iou.set_zlabel('Average IOU')
ax_iou.set_title('Sensitivity Analysis: Average IOU')
plt.savefig('sensitivity_iou.png')
plt.show()

# 绘制3D柱状图 - Average Center Error
# fig_error = plt.figure(figsize=(12, 8))
# ax_error = fig_error.add_subplot(111, projection='3d')
# xpos = pos_grid.ravel()
# ypos = var_grid.ravel()
# zpos = np.zeros_like(xpos)
# dx = 0.01
# dy = 0.02
# dz = center_error_grid.ravel()
# ax_error.bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average')
# ax_error.set_xlabel('pos_ratio_threshold (A)')
# ax_error.set_ylabel('variance_multiplier (B)')
# ax_error.set_zlabel('Average Center Error')
# ax_error.set_title('Sensitivity Analysis: Average Center Error')
# plt.savefig('sensitivity_center_error.png')
# plt.show()

# 输出最佳参数（例如最大IOU）
best_idx = np.unravel_index(np.argmax(iou_grid), iou_grid.shape)
best_pos_th = pos_thresholds[best_idx[1]]
best_var_m = var_mults[best_idx[0]]
best_iou = iou_grid[best_idx]
best_center_error = center_error_grid[best_idx]
print(f"Best parameters: pos_ratio_threshold={best_pos_th:.3f}, variance_multiplier={best_var_m:.3f}")
print(f"With Average IOU={best_iou:.4f}, Average Center Error={best_center_error:.4f}")


# D:\anacoZNB\python.exe D:\workspace\worksEventUtils\3_4\senstivity_analysis.py
# D:\anacoZNB\lib\site-packages\mpl_toolkits\mplot3d\axes3d.py:2471: RuntimeWarning: invalid value encountered in multiply
#   polys[..., i] = p + dp * cuboid[..., i]
# D:\anacoZNB\lib\site-packages\mpl_toolkits\mplot3d\art3d.py:1171: RuntimeWarning: invalid value encountered in subtract
#   v1 = polygons[..., i1, :] - polygons[..., i2, :]
# D:\anacoZNB\lib\site-packages\mpl_toolkits\mplot3d\art3d.py:1172: RuntimeWarning: invalid value encountered in subtract
#   v2 = polygons[..., i2, :] - polygons[..., i3, :]
# D:\anacoZNB\lib\site-packages\mpl_toolkits\mplot3d\proj3d.py:180: RuntimeWarning: invalid value encountered in divide
#   txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
# Best parameters: pos_ratio_threshold=0.518, variance_multiplier=0.900
# With Average IOU=0.8793, Average Center Error=9.0556