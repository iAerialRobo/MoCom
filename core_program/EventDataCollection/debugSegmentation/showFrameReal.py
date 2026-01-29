import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
import os

# 主要目标是生成过滤后的事件帧。

# 设置参数
resolution = (640, 480)  # VGA分辨率
frame_duration = timedelta(milliseconds=33)  # 每帧33毫秒
# input_file = "D:\\eventVision\\collecting\\3_3\\1.aedat4"  # 输入文件路径（请修改为你的文件路径）
# output_dir_raw = "D:\\eventVision\\collecting\\3_3\\result3\\raw"  # 原始事件图像数据集保存目录
# output_dir_filtered = "D:\\eventVision\\collecting\\3_3\\result3\\filtered"  # 过滤后事件图像数据集保存目录


# input_file = "D:\\eventVision\\collecting\\4_23\\3.aedat4"  # 输入文件路径（请修改为你的文件路径）
# output_dir_raw = "D:\\eventVision\\collecting\\4_23\\result3\\raw"  # 原始事件图像数据集保存目录
# output_dir_filtered = "D:\\eventVision\\collecting\\4_23\\result3\\filtered"  # 过滤后事件图像数据集保存目录


input_file = "D:\\eventVision\\collecting\\11_17\\4_0s.aedat4"  # 输入文件路径（请修改为你的文件路径）
output_dir_raw = "D:\\eventVision\\collecting\\11_17\\1_process\\raw"  # 原始事件图像数据集保存目录
output_dir_filtered = "D:\\eventVision\\collecting\\11_17\\1_process\\filtered"  # 过滤后事件图像数据集保存目录


raw_stats_file = os.path.join(output_dir_raw, "event_stats.txt")
filtered_stats_file = os.path.join(output_dir_filtered, "event_stats.txt")

# 创建输出目录
os.makedirs(output_dir_raw, exist_ok=True)
os.makedirs(output_dir_filtered, exist_ok=True)

# 初始化读取器
reader = dv.io.MonoCameraRecording(input_file)

# 初始化噪声过滤器
filter = dv.noise.BackgroundActivityNoiseFilter(resolution,
                                                backgroundActivityDuration=timedelta(milliseconds=1))

# 初始化可视化器，并设置正负事件的颜色
visualizer = dv.visualization.EventVisualizer(resolution)
visualizer.setPositiveColor(dv.visualization.colors.red())  # 正事件为红色
visualizer.setNegativeColor(dv.visualization.colors.blue())  # 负事件为蓝色

# 初始化事件累加器和时间管理
accumulator = dv.EventStore()
frame_count = 0
current_frame_start_time = None  # 当前帧的起始时间戳


def count_polarity_events(events_store):
    """计算正事件和负事件的数量"""
    positive_count = sum(1 for event in events_store if event.polarity() == 1)
    negative_count = sum(1 for event in events_store if event.polarity() == 0)
    return positive_count, negative_count


def process_frame(events_store, frame_idx, frame_start_time, output_dir, stats_file, is_filtered=False):
    """处理并保存一帧数据，包括图像和统计信息"""
    if len(events_store) > 0:
        # 获取帧的结束时间
        # frame_end_time = events_store[-1].timestamp()
        size = events_store.size()
        frame_end_time = events_store[size - 1].timestamp() if size > 0 else frame_start_time
        # 计算正负事件数量
        positive_count, negative_count = count_polarity_events(events_store)

        # 生成事件图像
        image = visualizer.generateImage(events_store)

        # 保存图像
        filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        cv.imwrite(filename, image)

        # 写入统计信息到TXT文件
        with open(stats_file, 'a') as f:
            f.write(f"Frame {frame_idx:06d}: "
                    f"Start Time: {frame_start_time}, "
                    f"End Time: {frame_end_time}, "
                    f"Positive Events: {positive_count}, "
                    f"Negative Events: {negative_count}\n")

        # 如果是原始帧，显示预览
        if not is_filtered:
            return image
        return None


# 清空统计文件（如果已存在）
for stats_file in [raw_stats_file, filtered_stats_file]:
    if os.path.exists(stats_file):
        os.remove(stats_file)

while reader.isRunning():
    # 读取事件批次
    events = reader.getNextEventBatch()

    if events is not None and len(events) > 0:
        # 如果是第一个事件批次，初始化起始时间
        if current_frame_start_time is None:
            current_frame_start_time = events[0].timestamp()

        # 处理当前批次中的所有事件
        for event in events:
            event_time = event.timestamp()
            time_since_frame_start = timedelta(microseconds=event_time - current_frame_start_time)

            # 如果当前事件超出了当前帧的时间窗口
            if time_since_frame_start >= frame_duration:
                # 处理当前帧（原始和过滤后的）
                raw_image = process_frame(accumulator, frame_count, current_frame_start_time,
                                          output_dir_raw, raw_stats_file, is_filtered=False)
                filter.accept(accumulator)
                filtered_events = filter.generateEvents()
                process_frame(filtered_events, frame_count, current_frame_start_time,
                              output_dir_filtered, filtered_stats_file, is_filtered=True)

                # 显示预览
                # if raw_image is not None:
                #    filtered_image = visualizer.generateImage(filtered_events)
                #    preview = cv.hconcat([raw_image, filtered_image])
                #    cv.imshow("Preview (Left: Raw, Right: Filtered)", preview)
                #    cv.waitKey(1)

                # 计算当前事件属于哪个新的时间窗口
                frames_elapsed = int(time_since_frame_start / frame_duration)
                new_frame_start_time = (current_frame_start_time +
                                        frames_elapsed * frame_duration.total_seconds() * 1_000_000)

                # 如果跨越了多个帧，填充中间的空帧
                for i in range(frame_count + 1, frame_count + frames_elapsed):
                    empty_store = dv.EventStore()
                    process_frame(empty_store, i, current_frame_start_time,
                                  output_dir_raw, raw_stats_file, is_filtered=False)
                    process_frame(empty_store, i, current_frame_start_time,
                                  output_dir_filtered, filtered_stats_file, is_filtered=True)

                # 重置累加器并更新帧信息
                accumulator = dv.EventStore()
                current_frame_start_time = new_frame_start_time
                frame_count += frames_elapsed

            # 将单个事件添加到当前累加器
            accumulator.push_back(event)

# 处理最后的剩余事件
if len(accumulator) > 0:
    raw_image = process_frame(accumulator, frame_count, current_frame_start_time,
                              output_dir_raw, raw_stats_file, is_filtered=False)
    filter.accept(accumulator)
    filtered_events = filter.generateEvents()
    process_frame(filtered_events, frame_count, current_frame_start_time,
                  output_dir_filtered, filtered_stats_file, is_filtered=True)
    frame_count += 1

# 关闭窗口
cv.destroyAllWindows()

print(f"处理完成！总共生成了 {frame_count} 帧")
print(f"原始图像和统计信息保存在: {output_dir_raw}")
print(f"过滤后图像和统计信息保存在: {output_dir_filtered}")