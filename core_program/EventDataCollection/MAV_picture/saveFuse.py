import numpy as np
import cv2
import os

# 读取 npz 文件
file_path = "D:\\eventVision\\data_frame\\up_down\\up_down10.npz"  # 修改为你自己的文件路径
data = np.load(file_path)

# 列出所有数据的 key
print("Keys in npz file:", list(data.keys()))

# 选择一个 key 进行可视化
key = list(data.keys())[0]  # 选第一个 key
array = data[key]

# 检查数据形状
print(f"Shape of '{key}': {array.shape}")

# 假设数据形状是 (16, 2, 128, 128)
frames = data['frames']  # 形状应为 (16, 2, 128, 128)

# 保存路径
output_dir = "D:\\workspace\\worksEventUtils\\npzShow\\output_fused_up_down\\"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # 创建输出目录

# 遍历 16 帧，将正事件和负事件叠加并保存
for frame in range(16):
    # 提取正事件和负事件
    img1 = frames[frame, 0, :, :]  # 正事件 (通道 0)
    img2 = frames[frame, 1, :, :]  # 负事件 (通道 1)

    # 将数据转换为 uint8 类型（0-255）
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)

    # 构造 RGB 图像
    img1_rgb = np.dstack((img1, np.zeros_like(img1), np.zeros_like(img1)))  # 红色通道 (R, 0, 0) 表示正事件
    img2_rgb = np.dstack((np.zeros_like(img2), img2, np.zeros_like(img2)))  # 绿色通道 (0, G, 0) 表示负事件

    # 将正事件和负事件叠加
    combined_img = img1_rgb + img2_rgb  # 形状仍为 (128, 128, 3)

    # 保存图片
    filename = f"{output_dir}frame_{frame:02d}.png"
    cv2.imwrite(filename, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))  # OpenCV 使用 BGR 格式
    print(f"Saved: {filename}")

print("All 16 overlay images have been saved successfully!")