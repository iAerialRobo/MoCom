import numpy as np

# GT 结果（去掉第一个 1）
gt_sequence = [100,230, 285,540, 600,830, 900,1120, 1185,1410, 1470,1710, 1770,2025, 2084,2320, 2395,2540]
num_actions = 9

# 确保帧数匹配动作数量165,245,    340,585,   683,915,    1005,1240,   1330,1555,    1655,1880,  1975,2220,   2315,2566,  2660,2799
assert len(gt_sequence) == 2 * num_actions, "Number of GT frames does not match number of actions!"

# 分离 GT 的开始帧和结束帧
gt_start_frames = gt_sequence[0::2]  # [100, 285, 600, ..., 2395]
gt_end_frames = gt_sequence[1::2]    # [230, 540, 830, ..., 2540]

# 假设的预测结果
pred_sequenceMe = [71,217, 275,546, 590,821, 868,1391, 868,1391, 1461,1717, 1747,2030, 2051,2329, 2381,2562,]
pred_sequenceHMM = [79,183, 275,446, 610,821, 900,1391, 1391,1430, 1494,1677, 1747,2030, 2041,2350, 2413,2555, ]
pred_sequenceCPD =  [ 0, 34, 545,574, 610,821, 1125, 1159,  1420, 1449,  1494,1677, 1747,2030, 2041,2350, 2580,2742,]

pred_sequence = pred_sequenceCPD # pred_sequenceHMM # pred_sequenceMe
# 分离预测的开始帧和结束帧
pred_start_frames = pred_sequence[0::2]  # [110, 290, 610, ..., 2400]
pred_end_frames = pred_sequence[1::2]    # [240, 530, 820, ..., 2530]

# 计算 GT 和预测的中心点
gt_centers = [(start + end) / 2 for start, end in zip(gt_start_frames, gt_end_frames)]
pred_centers = [(start + end) / 2 for start, end in zip(pred_start_frames, pred_end_frames)]
print("GT Center Points:", gt_centers)
print("Predicted Center Points:", pred_centers)

# 计算每个动作的中心点误差
center_errors = [pred - gt for pred, gt in zip(pred_centers, gt_centers)]
print("Center Point Errors (Predicted - GT):", center_errors)

# 计算平均中心误差
mean_center_error = np.mean(center_errors)
print("Mean Center Error:", mean_center_error)

# 计算每个动作的总帧数（包括开始帧和结束帧）
gt_total_frames = [end - start + 1 for start, end in zip(gt_start_frames, gt_end_frames)]
pred_total_frames = [end - start + 1 for start, end in zip(pred_start_frames, pred_end_frames)]
print("GT Total Frames:", gt_total_frames)
print("Predicted Total Frames:", pred_total_frames)

# 计算每个动作的总帧数误差
total_frame_errors = [pred - gt for pred, gt in zip(pred_total_frames, gt_total_frames)]
print("Total Frame Errors (Predicted - GT):", total_frame_errors)

# 计算平均动作总帧数误差
mean_total_frame_error = np.mean(total_frame_errors)
print("Mean Total Frame Error:", mean_total_frame_error)

# 计算每个动作的 IoU
ious = []
for gt_start, gt_end, pred_start, pred_end in zip(gt_start_frames, gt_end_frames, pred_start_frames, pred_end_frames):
    # 计算交集
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start) + 1)
    # 计算并集
    union = (gt_end - gt_start + 1) + (pred_end - pred_start + 1) - intersection
    # 计算 IoU
    iou = intersection / union if union > 0 else 0
    ious.append(iou)

print("IoU for each action:", ious)
print("Mean IoU for each action:", np.mean(ious))

# 打印总结
print("\nSummary:")
for i in range(num_actions):
    print(f"Action {i+1}:")
    print(f"  GT: Start {gt_start_frames[i]}, End {gt_end_frames[i]}, Center {gt_centers[i]:.1f}, Total Frames {gt_total_frames[i]}")
    print(f"  Predicted: Start {pred_start_frames[i]}, End {pred_end_frames[i]}, Center {pred_centers[i]:.1f}, Total Frames {pred_total_frames[i]}")
    print(f"  Center Error: {center_errors[i]:.2f}")
    print(f"  Total Frame Error: {total_frame_errors[i]:.2f}")
    print(f"  IoU: {ious[i]:.4f}")