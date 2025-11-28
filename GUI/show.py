import sys
import os
import time
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QSlider, QListWidget, QListWidgetItem, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage

import dv_processing as dv
from datetime import timedelta
import cv2 as cv
import numpy as np


class VideoThread(QThread):
    progress_signal = pyqtSignal(float)
    duration_signal = pyqtSignal(float)
    frame_signal = pyqtSignal(object)  # 发送帧数据

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._running = True
        self._paused = False
        self.speed = 1.0
        self.target_percent = None

    def set_speed(self, speed):
        self.speed = speed

    def jump_to_percent(self, percent):
        self.target_percent = percent

    def set_paused(self, paused):
        self._paused = paused

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        import dv_processing as dv
        from datetime import timedelta
        import time
        import cv2 as cv

        reader = dv.io.MonoCameraRecording(self.file_path)

        # 正确获取时间范围
        start_time, end_time = reader.getTimeRange()
        total_duration = (end_time - start_time) / 1_000_000.0  # ).total_seconds()
        self.duration_signal.emit(total_duration)

        visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
        visualizer.setBackgroundColor(dv.visualization.colors.white())
        visualizer.setPositiveColor(dv.visualization.colors.red())
        visualizer.setNegativeColor(dv.visualization.colors.green())

        slicer = dv.EventStreamSlicer()

        def slicing_callback(events: dv.EventStore):
            frame = visualizer.generateImage(events)
            if frame is not None:
                self.frame_signal.emit(frame)  # 发送帧数据
                percent = ((events.getLowestTime() - start_time) / 1_000_000.0) / total_duration
                self.progress_signal.emit(percent)

        slicer.doEveryTimeInterval(timedelta(milliseconds=int(33 / self.speed)), slicing_callback)

        while reader.isRunning() and self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            if self.target_percent is not None:
                target_time = self.target_percent * total_duration + start_time.total_seconds()
                reader.seekToTimestamp(timedelta(seconds=target_time))
                self.target_percent = None

            events = reader.getNextEventBatch()
            if events is not None:
                slicer.accept(events)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEDAT4 视频播放与识别结果展示")
        self.resize(1200, 700)

        self.video_thread = None
        self.is_playing = True

        main_layout = QHBoxLayout(self)

        # 左边视频控制区域
        video_layout = QVBoxLayout()

        # 创建视频显示标签
        self.video_display = QLabel("等待视频加载...")
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")

        self.open_btn = QPushButton("加载 AEDAT4 文件")
        self.play_pause_btn = QPushButton("暂停")
        self.speed_label = QLabel("播放速度：")
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 10)
        self.speed_spin.setValue(1)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.sliderReleased.connect(self.seek_video)

        self.open_btn.clicked.connect(self.load_file)
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.speed_spin.valueChanged.connect(self.change_speed)

        video_layout.addWidget(self.video_display)  # 添加视频显示区域
        video_layout.addWidget(self.progress_slider)
        video_layout.addWidget(self.open_btn)
        video_layout.addWidget(self.play_pause_btn)
        speed_box = QHBoxLayout()
        speed_box.addWidget(self.speed_label)
        speed_box.addWidget(self.speed_spin)
        video_layout.addLayout(speed_box)

        # 右边识别结果显示区
        result_layout = QVBoxLayout()
        self.result_label = QLabel("识别结果")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 创建识别结果图像显示
        self.result_display = QLabel("选择识别结果查看")
        self.result_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_display.setMinimumSize(400, 300)
        self.result_display.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")

        self.result_list = QListWidget()
        self.result_list.setIconSize(QSize(64, 64))
        self.result_list.currentItemChanged.connect(self.show_selected_result)
        self.load_sample_results()

        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_display)
        result_layout.addWidget(self.result_list)

        main_layout.addLayout(video_layout, 2)
        main_layout.addLayout(result_layout, 1)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 AEDAT4 文件", "", "AEDAT4 Files (*.aedat4)")
        if file_path:
            if self.video_thread:
                self.video_thread.stop()

            self.video_thread = VideoThread(file_path)
            self.video_thread.progress_signal.connect(self.update_progress)
            self.video_thread.frame_signal.connect(self.update_frame)
            self.video_thread.start()

    def update_frame(self, frame):
        # 将OpenCV图像转换为QImage并显示在QLabel上
        if frame is not None:
            # 假设frame是一个numpy数组，格式为BGR
            if isinstance(frame, np.ndarray):
                # 如果是灰度图
                if len(frame.shape) == 2:
                    height, width = frame.shape
                    qimg = QImage(frame.data, width, height, width, QImage.Format.Format_Grayscale8)
                # 如果是彩色图
                elif len(frame.shape) == 3:
                    height, width, channels = frame.shape
                    # OpenCV使用BGR，Qt使用RGB，需要转换
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    qimg = QImage(rgb_frame.data, width, height, width * channels, QImage.Format.Format_RGB888)
                else:
                    return

                # 调整图像大小以适应标签
                pixmap = QPixmap.fromImage(qimg)
                pixmap = pixmap.scaled(self.video_display.width(), self.video_display.height(),
                                       Qt.AspectRatioMode.KeepAspectRatio)
                self.video_display.setPixmap(pixmap)
            else:
                # 如果frame不是numpy数组，可能是DV库的特定类型
                # 这里需要根据dv_processing返回的frame类型来具体实现
                # 示例代码（需要根据实际情况调整）:
                try:
                    # 假设frame有一个转换为numpy数组的方法
                    np_frame = np.array(frame)
                    height, width = np_frame.shape[:2]
                    qimg = QImage(np_frame.data, width, height, width * 3, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    pixmap = pixmap.scaled(self.video_display.width(), self.video_display.height(),
                                           Qt.AspectRatioMode.KeepAspectRatio)
                    self.video_display.setPixmap(pixmap)
                except Exception as e:
                    print(f"转换帧失败: {e}")

    def toggle_playback(self):
        if self.video_thread:
            self.is_playing = not self.is_playing
            self.video_thread.set_paused(not self.is_playing)
            self.play_pause_btn.setText("播放" if not self.is_playing else "暂停")

    def update_progress(self, percent):
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(int(percent * 100))
        self.progress_slider.blockSignals(False)

    def seek_video(self):
        if self.video_thread:
            value = self.progress_slider.value()
            self.video_thread.jump_to_percent(value / 100)

    def change_speed(self, val):
        if self.video_thread:
            self.video_thread.set_speed(val)

    def load_sample_results(self):
        result_dir = "recognition_images"
        if os.path.isdir(result_dir):
            for file in os.listdir(result_dir):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    item = QListWidgetItem(file)
                    item.setIcon(QPixmap(os.path.join(result_dir, file)))
                    self.result_list.addItem(item)

    def show_selected_result(self, current, previous):
        if current is not None:
            result_dir = "recognition_images"
            file_path = os.path.join(result_dir, current.text())
            if os.path.isfile(file_path):
                pixmap = QPixmap(file_path)
                pixmap = pixmap.scaled(self.result_display.width(), self.result_display.height(),
                                       Qt.AspectRatioMode.KeepAspectRatio)
                self.result_display.setPixmap(pixmap)

    def resizeEvent(self, event):
        # 当窗口大小改变时重新调整图像大小
        super().resizeEvent(event)
        if hasattr(self.video_display, 'pixmap') and not self.video_display.pixmap().isNull():
            pixmap = self.video_display.pixmap()
            self.video_display.setPixmap(pixmap.scaled(
                self.video_display.width(), self.video_display.height(),
                Qt.AspectRatioMode.KeepAspectRatio))
        if hasattr(self.result_display, 'pixmap') and not self.result_display.pixmap().isNull():
            pixmap = self.result_display.pixmap()
            self.result_display.setPixmap(pixmap.scaled(
                self.result_display.width(), self.result_display.height(),
                Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())