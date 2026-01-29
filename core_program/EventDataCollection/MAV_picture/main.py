import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
# 生成 (16, 2, 128, 128) 的二值数据（仅 0 和 1）
# data = np.random.randint(0, 2, (16, 2, 128, 128), dtype=np.uint8)

# 创建画布
fig, ax = plt.subplots(figsize=(3, 6))
img_display = ax.imshow(np.zeros((960, 640, 3), dtype=np.uint8))  # 初始空图
ax.axis("off")  # 关闭坐标轴

# 读取 npz 文件
# file_path = "C:\\Users\\ZhuanZ.DESKTOP-LJJ6K3U\\Desktop\\0\\user24_fluorescent_0.npz"  # 修改为你的 .npz 文件路径
# file_path = "D:\\eventVision\\data_frame\\inv_vShape\\inv_vShape1.npz"
# file_path = "D:\\eventVision\\data_frame\\left_right\\left_right10.npz"
# file_path = "D:\\eventVision\\data_frame\\up_down\\up_down10.npz"
file_path = "D:\\eventVision\\data_frame\\vShape\\vShape10.npz"
file_path = "./vShape_217_shift+4.npz"
data = np.load(file_path)

# 列出所有数据的 key
print("Keys in npz file:", list(data.keys()))

# 选择一个 key 进行可视化
key = list(data.keys())[0]  # 选第一个 key
array = data[key]

# 检查数据形状
print(f"Shape of '{key}': {array.shape}")

import numpy as np
import matplotlib.pyplot as plt

# 假设你的数组是 (16, 2, 128, 128)
# data = np.random.rand(16, 2, 128, 128)  # 这里用随机数据模拟

# 遍历 16 张图
def update(frame):
    index = frame # % len(data)
    img1 = data['frames'][index, 0, :, :]  # 第一张 1x128x128
    img2 = data['frames'][index, 1, :, :]  # 第二张 1x128x128
    # img1_ = (img1 - img1.min()) / (img1.max() - img1.min())
   #  img2_ = (img2 - img2.min()) / (img2.max() - img2.min())
    img1 = (img1 * 255).astype(np.uint8)  # 转换为 uint8
    img2 = (img2 * 255).astype(np.uint8)  # 转换为 uint8
   # cv2.imshow("Normalized Image", img1)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
    max1 = np.max(img1)
    max2 = np.max(img1)
    min1 = np.min(img2)
    min2 = np.min(img2)
    # 构造 RGB 图像
    img1_rgb = np.dstack((img1, np.zeros_like(img1), np.zeros_like(img1)))  # 红色通道 (R,0,0)
    img2_rgb = np.dstack((np.zeros_like(img2), img2, np.zeros_like(img2)))  # 绿色通道 (0,G,0)

    # 纵向拼接成 256x128
    combined_img = np.vstack((img1_rgb, img2_rgb))

    img_display.set_array(combined_img)
    print(frame)
    return [img_display]

# 创建动画（16 帧，每帧间隔 500ms）
ani = animation.FuncAnimation(fig, update, frames=16, interval=100, blit=True)

ani.save("vShape.gif", writer="pillow", fps=2)  # fps=2 → 每秒 2 帧（500ms 间隔）

# 显示动画
# plt.show()
