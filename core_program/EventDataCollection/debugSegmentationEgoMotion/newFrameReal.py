import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
import os
from pathlib import Path


class EventFrameProcessor:
    """事件相机帧处理器"""

    def __init__(self, resolution=(640, 480), frame_duration_ms=33):
        """
        初始化处理器

        Args:
            resolution: 图像分辨率 (width, height)
            frame_duration_ms: 帧持续时间(毫秒)
        """
        self.resolution = resolution
        self.frame_duration = timedelta(milliseconds=frame_duration_ms)
        self.frame_duration_us = int(frame_duration_ms * 1000)

        # 初始化可视化器
        self.visualizer = dv.visualization.EventVisualizer(resolution)
        self.visualizer.setPositiveColor(dv.visualization.colors.red())
        self.visualizer.setNegativeColor(dv.visualization.colors.blue())

        # 初始化噪声过滤器
        self.noise_filter = dv.noise.BackgroundActivityNoiseFilter(
            resolution,
            backgroundActivityDuration=timedelta(milliseconds=1)
        )

    def count_events_by_polarity(self, events):
        """统计正负事件数量"""
        if len(events) == 0:
            return 0, 0

        # 使用numpy加速统计
        polarities = np.array([e.polarity() for e in events])
        positive_count = np.sum(polarities == 1)
        negative_count = np.sum(polarities == 0)
        return int(positive_count), int(negative_count)

    def save_frame(self, events, frame_idx, start_time, end_time, output_dir, stats_file):
        """
        保存单帧数据

        Args:
            events: 事件数据
            frame_idx: 帧索引
            start_time: 起始时间戳
            end_time: 结束时间戳
            output_dir: 输出目录
            stats_file: 统计文件路径

        Returns:
            生成的图像
        """
        if len(events) == 0:
            return None

        # 统计事件
        pos_count, neg_count = self.count_events_by_polarity(events)

        # 生成图像
        image = self.visualizer.generateImage(events)

        # 保存图像
        filename = output_dir / f"frame_{frame_idx:06d}.png"
        cv.imwrite(str(filename), image)

        # 写入统计信息
        with open(stats_file, 'a', encoding='utf-8') as f:
            f.write(
                f"Frame {frame_idx:06d}: "
                f"Start={start_time}, End={end_time}, "
                f"Positive={pos_count}, Negative={neg_count}, "
                f"Total={pos_count + neg_count}\n"
            )

        return image

    def process_recording(self, input_file, output_base_dir, show_preview=True):
        """
        处理录制文件

        Args:
            input_file: 输入aedat4文件路径
            output_base_dir: 输出基础目录
            show_preview: 是否显示预览窗口
        """
        # 创建输出目录
        output_base = Path(output_base_dir)
        raw_dir = output_base / "raw"
        filtered_dir = output_base / "filtered"
        raw_dir.mkdir(parents=True, exist_ok=True)
        filtered_dir.mkdir(parents=True, exist_ok=True)

        raw_stats = raw_dir / "event_stats.txt"
        filtered_stats = filtered_dir / "event_stats.txt"

        # 清空已存在的统计文件
        for stats_file in [raw_stats, filtered_stats]:
            if stats_file.exists():
                stats_file.unlink()

        # 初始化读取器
        reader = dv.io.MonoCameraRecording(str(input_file))

        # 使用切片器按时间切分事件
        slicer = dv.EventStreamSlicer()
        frame_count = 0

        def process_slice(events_slice):
            """处理单个时间切片"""
            nonlocal frame_count

            if len(events_slice) == 0:
                return

            # 获取时间范围
            start_time = events_slice[0].timestamp()
            end_time = events_slice[len(events_slice) - 1].timestamp()

            # 处理原始事件
            raw_image = self.save_frame(
                events_slice, frame_count, start_time, end_time,
                raw_dir, raw_stats
            )

            # 过滤噪声
            self.noise_filter.accept(events_slice)
            filtered_events = self.noise_filter.generateEvents()

            # 保存过滤后的事件
            filtered_image = self.save_frame(
                filtered_events, frame_count, start_time, end_time,
                filtered_dir, filtered_stats
            )

            # 显示预览
            if show_preview and raw_image is not None and filtered_image is not None:
                preview = cv.hconcat([raw_image, filtered_image])
                cv.imshow("Preview (Left: Raw, Right: Filtered)", preview)
                cv.waitKey(1)

            frame_count += 1

            # 打印进度
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count} 帧...")

        # 配置切片器：按固定时间间隔切分
        slicer.doEveryTimeInterval(self.frame_duration, process_slice)

        # 读取并处理所有事件
        print("开始处理事件数据...")
        while reader.isRunning():
            events = reader.getNextEventBatch()
            if events is not None:
                slicer.accept(events)

        # 关闭窗口
        if show_preview:
            cv.destroyAllWindows()

        print(f"\n处理完成！")
        print(f"总共生成了 {frame_count} 帧")
        print(f"原始数据保存在: {raw_dir}")
        print(f"过滤数据保存在: {filtered_dir}")

        return frame_count


def main():
    """主函数"""
    # 配置参数
    INPUT_FILE = "E:\\IEEE_tro_compareExperiment\\11_14\\2\\order.aedat4"
    OUTPUT_DIR = "E:\\IEEE_tro_compareExperiment\\11_14\\png"

    # 创建处理器
    processor = EventFrameProcessor(
        resolution=(640, 480),
        frame_duration_ms=33  # 约30fps
    )

    # 处理录制文件
    try:
        processor.process_recording(
            input_file=INPUT_FILE,
            output_base_dir=OUTPUT_DIR,
            show_preview=True
        )
    except Exception as e:
        print(f"处理出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()