from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QListWidget, QLabel, QFileDialog, QGridLayout, QFrame,
    QSplitter, QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage
import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import time
from datetime import timedelta
import cv2 as cv
import os
import glob
from recognizer import Recognizer


# Video Thread for playing AEDAT4 files
class VideoThread(QThread):
    progress_signal = pyqtSignal(float)
    duration_signal = pyqtSignal(float)
    frame_signal = pyqtSignal(object)  # Send frame data

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
        try:
            import dv_processing as dv

            reader = dv.io.MonoCameraRecording(self.file_path)

            # Get time range
            start_time, end_time = reader.getTimeRange()
            total_duration = (end_time - start_time) / 1_000_000.0
            self.duration_signal.emit(total_duration)

            visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
            visualizer.setBackgroundColor(dv.visualization.colors.white())
            visualizer.setPositiveColor(dv.visualization.colors.red())
            visualizer.setNegativeColor(dv.visualization.colors.green())

            slicer = dv.EventStreamSlicer()

            def slicing_callback(events: dv.EventStore):
                frame = visualizer.generateImage(events)
                if frame is not None:
                    self.frame_signal.emit(frame)  # Send frame data
                    percent = ((events.getLowestTime() - start_time) / 1_000_000.0) / total_duration
                    self.progress_signal.emit(percent)

            slicer.doEveryTimeInterval(timedelta(milliseconds=int(33 / self.speed)), slicing_callback)

            while reader.isRunning() and self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue

                if self.target_percent is not None:
                    target_time = start_time + timedelta(seconds=self.target_percent * total_duration)
                    reader.seekToTimestamp(target_time)
                    self.target_percent = None

                events = reader.getNextEventBatch()
                if events is not None:
                    slicer.accept(events)
        except Exception as e:
            print(f"Video thread error: {str(e)}")


# Custom thread class for running recognition or segmentation tasks
class WorkerThread(QThread):
    output_signal = pyqtSignal(str)
    result_signal = pyqtSignal(list)
    image_signal = pyqtSignal(dict)
    result_received_signal = pyqtSignal(list)

    def __init__(self, command, task_type, file_path=None, output_dir=None):
        super().__init__()
        self.command = command
        self.task_type = task_type
        self.file_path = file_path
        self.output_dir = output_dir

    def run(self):
        try:
            if self.task_type == "direct_run":
                # Extract frame data, compute features, and segment actions
                import numpy as np
                import os
                import sys

                # Add the directory containing the module to sys.path
                parent_dir = os.path.dirname(os.path.abspath(__file__))
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)

                # Import functions from your script
                try:
                    from paste import (
                        process_event_data, compute_features, segment_actions,
                        post_process_actions, frames_to_timestamps,
                        split_aedat4_by_actions, convert_aedat4_to_npz
                    )
                except ImportError as e:
                    self.output_signal.emit(f"[错误] 无法导入 paste 模块: {str(e)}")
                    self.image_signal.emit({
                        "segments": [],
                        "total_events": np.array([]),
                        "positive_events": np.array([]),
                        "negative_events": np.array([]),
                        "action_segments": []
                    })
                    self.result_signal.emit([])
                    return

                self.output_signal.emit(f"Processing event data from {self.file_path}...")
                frame_to_time, positive_events, negative_events, total_events = process_event_data(self.file_path)
                self.output_signal.emit(f"Computed events data: {len(total_events)} frames")

                pos_ratio, variance = compute_features(total_events, positive_events)
                self.output_signal.emit("Computed features for segmentation")

                action_segments = segment_actions(pos_ratio, variance, total_events)
                final_action_segments = post_process_actions(action_segments)

                # Print segmentation results
                self.output_signal.emit(f"\nTotal actions detected: {len(final_action_segments)}")
                for i, (start, end) in enumerate(final_action_segments):
                    duration_ms = (end - start + 1) * 33
                    self.output_signal.emit(f"Action {i + 1}: Frame {start} to {end}, Duration {duration_ms} ms")

                # Convert to timestamps
                action_timestamps = frames_to_timestamps(final_action_segments, frame_to_time)

                # Generate visualization data for the segmentation
                visualization_data = {
                    "segments": [],
                    "total_events": total_events,
                    "positive_events": positive_events,
                    "negative_events": negative_events,
                    "action_segments": final_action_segments
                }
                for i, (start, end) in enumerate(final_action_segments):
                    segment_data = {
                        "index": i + 1,
                        "start": start,
                        "end": end,
                        "events": total_events[start:end + 1] if start < len(total_events) and end < len(
                            total_events) else np.array([]),
                        "pos_events": positive_events[start:end + 1] if start < len(positive_events) and end < len(
                            positive_events) else np.array([]),
                        "neg_events": negative_events[start:end + 1] if start < len(negative_events) and end < len(
                            negative_events) else np.array([])
                    }
                    visualization_data["segments"].append(segment_data)

                print(f"Emitting visualization_data: type={type(visualization_data)}, keys={list(visualization_data.keys())}")
                # Send the visualization data
                self.output_signal.emit(f"Emitting visualization_data: type={type(visualization_data)}, keys={list(visualization_data.keys())}")
                self.image_signal.emit(visualization_data)

                # Create output directory if it doesn't exist
                os.makedirs(self.output_dir, exist_ok=True)
                self.output_signal.emit(f"Output directory: {self.output_dir}")
                # Split aedat4 file
                output_aedat4_files = split_aedat4_by_actions(
                    self.file_path, action_timestamps,
                    os.path.join(self.output_dir, "UAV_action")
                )

                self.output_signal.emit(f"Split {len(output_aedat4_files)} action segments to {self.output_dir}")

                # Convert each .aedat4 to .npz
                npz_file_paths = []
                for aedat4_file in output_aedat4_files:
                    self.output_signal.emit(f"Attempting to convert {aedat4_file} to .npz...")
                    npz_path = convert_aedat4_to_npz(
                        aedat4_file,
                        self.output_dir,
                        frames_num=16,
                        split_by="number",
                        H=480,
                        W=640
                    )
                    if npz_path:
                        npz_file_paths.append(npz_path)
                        self.output_signal.emit(f"Successfully converted {aedat4_file} to {npz_path}")
                    else:
                        self.output_signal.emit(f"Failed to convert {aedat4_file} to .npz")

                # Emit all output files (.aedat4 and .npz)
                all_output_files = output_aedat4_files + npz_file_paths
                self.output_signal.emit(f"Emitting {len(all_output_files)} output files: {all_output_files}")
                self.result_signal.emit(all_output_files)
                self.result_received_signal.emit(all_output_files)  # 发射新信号
                self.output_signal.emit(f"Generated {len(npz_file_paths)} .npz files")

            else:
                # Run as subprocess
                process = subprocess.Popen(
                    self.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    text=True,
                    bufsize=1
                )
                for line in process.stdout:
                    self.output_signal.emit(line.strip())

        except Exception as e:
            self.output_signal.emit(f"[错误] {str(e)}")
            import traceback
            self.output_signal.emit(traceback.format_exc())
            # Emit an empty dictionary to prevent crashes
            self.image_signal.emit({
                "segments": [],
                "total_events": np.array([]),
                "positive_events": np.array([]),
                "negative_events": np.array([]),
                "action_segments": []
            })
            self.result_signal.emit([])


class SegmentCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(SegmentCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

    def plot_segment(self, segment_data, action_segments=None):
        print('============')
        self.axes.clear()
        events = segment_data.get("events", [])
        if len(events) == 0:
            self.axes.set_title(f"No data for segment {segment_data['index']}")
            self.draw()
            return

        x = np.arange(segment_data["start"], segment_data["end"] + 1)
        self.axes.plot(x, segment_data["events"], 'b-', label='Total Events')
        self.axes.plot(x, segment_data["pos_events"], 'g-', label='Positive Events')
        self.axes.plot(x, segment_data["neg_events"], 'r-', label='Negative Events')

        if action_segments:
            for start, end in action_segments:
                if start >= segment_data["start"] and start <= segment_data["end"]:
                    self.axes.axvline(x=start, color='k', linestyle='--', label='Segment Start' if start == action_segments[0][0] else '')
                if end >= segment_data["start"] and end <= segment_data["end"]:
                    self.axes.axvline(x=end, color='k', linestyle='--', label='Segment End' if end == action_segments[0][1] else '')

        self.axes.set_title(f"Segment {segment_data['index']}: Frames {segment_data['start']}-{segment_data['end']}")
        self.axes.set_xlabel("Frame")
        self.axes.set_ylabel("Event Count")
        self.axes.legend()
        self.fig.tight_layout()
        self.draw()

    def plot_all_events(self, total_events, positive_events, negative_events, action_segments):
        self.axes.clear()
        if len(total_events) == 0:
            self.axes.set_title("No event data available")
            self.draw()
            return

        x = np.arange(len(total_events))
        self.axes.plot(x, total_events, 'b-', label='Total Events')
        # Uncomment to include positive and negative events
        # self.axes.plot(x, positive_events, 'g-', label='Positive Events')
        # self.axes.plot(x, negative_events, 'r-', label='Negative Events')

        # Plot segment start and end lines with different colors
        for i, (start, end) in enumerate(action_segments):
            # Start line: Green, dashed
            self.axes.axvline(x=start, color='green', linestyle='--',
                              label='Segment Start' if i == 0 else '')
            # End line: Red, dashed
            self.axes.axvline(x=end, color='red', linestyle='--',
                              label='Segment End' if i == 0 else '')
            # Add segment label (A1, A2, etc.) in the middle of the segment
            segment_mid = (start + end) / 2
            # Place label above the plot line, adjust y-position dynamically
            y_max = total_events[start:end + 1].max() if len(total_events[start:end + 1]) > 0 else 1
            self.axes.text(segment_mid, y_max * 1.1, f'A{i + 1}',
                           ha='center', va='bottom', fontsize=10, color='black',
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        self.axes.set_title("All Frames Event Data with Action Segments")
        self.axes.set_xlabel("Frame")
        self.axes.set_ylabel("Event Count")
        self.axes.legend()
        self.fig.tight_layout()
        self.draw()

    def plot_recognition_results(self, results):
        self.axes.clear()

        if not results:
            self.axes.set_title("No recognition results available")
            self.draw()
            return

        # Extract data for plotting
        labels = [result['label'] for result in results]
        files = [os.path.basename(result['file']) for result in results]
        indices = np.arange(len(results))

        # Plot bar chart
        bars = self.axes.bar(indices, [1] * len(results), tick_label=files)
        self.axes.set_title("Recognition Results")
        self.axes.set_xlabel("NPZ File")
        self.axes.set_ylabel("Prediction")

        # Add labels on top of bars
        for i, bar in enumerate(bars):
            self.axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                labels[i],
                ha='center',
                va='center',
                rotation=45,
                fontsize=8,
                color='black'
            )

        # Rotate x-axis labels for better readability
        self.axes.set_xticklabels(files, rotation=45, ha='right')
        self.fig.tight_layout()
        self.draw()

    def plot_single_recognition_result(self, result):
        self.axes.clear()

        if not result:
            self.axes.set_title("No recognition result selected")
            self.draw()
            return

        # Extract data from the result dictionary
        filename = os.path.basename(result['file'])
        label = result['label']
        comm_code = result['comm_code']

        # Display text information
        self.axes.text(
            0.5, 0.6,
            f"File: {filename}",
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )
        self.axes.text(
            0.5, 0.4,
            f"Label: {label} ({comm_code})",
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )

        self.axes.set_title("Recognition Result")
        self.axes.axis('off')  # Hide axes for a cleaner look
        self.fig.tight_layout()
        self.draw()


# Main window class
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEDAT4 分割、识别与播放系统")
        self.resize(1400, 900)

        # Initialize recognition_results
        self.recognition_results = []  # Add this line

        default_file_path = "D:/workspace/worksEventUtils/data/default.aedat4"
        self.file_path = default_file_path if os.path.exists(default_file_path) else ""
        self.output_dir = ""
        self.npz_files = ""

        # Main layout (vertical)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top section: Video Display and Segmentation Results
        top_frame = QFrame()
        top_frame.setFrameShape(QFrame.Shape.StyledPanel)
        top_layout = QHBoxLayout(top_frame)

        # Video display on the left
        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.Shape.StyledPanel)
        video_layout = QVBoxLayout(video_frame)

        self.video_label = QLabel("视频播放区域")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setStyleSheet("background-color: black; color: white;")

        video_layout.addWidget(self.video_label)

        # Segmentation results on the right
        segment_frame = QFrame()
        segment_frame.setFrameShape(QFrame.Shape.StyledPanel)
        segment_layout = QVBoxLayout(segment_frame)

        self.segment_display_label = QLabel("分割结果显示:")
        segment_layout.addWidget(self.segment_display_label)

        self.segment_canvas = SegmentCanvas(width=6, height=3)
        segment_layout.addWidget(self.segment_canvas)

        self.segment_list = QListWidget()
        self.segment_list.setMaximumHeight(100)
        self.segment_list.currentRowChanged.connect(self.update_segment_display)
        segment_layout.addWidget(self.segment_list)

        self.show_all_btn = QPushButton("显示所有帧事件数据")
        self.show_all_btn.clicked.connect(self.show_all_events)
        self.show_all_btn.setEnabled(False)
        segment_layout.addWidget(self.show_all_btn)

        # Add video and segment areas to top layout
        top_layout.addWidget(video_frame, 1)
        top_layout.addWidget(segment_frame, 1)

        # Bottom section: Controls and output
        bottom_frame = QFrame()
        bottom_frame.setFrameShape(QFrame.Shape.StyledPanel)
        bottom_layout = QHBoxLayout(bottom_frame)

        # Left control panel with video controls
        left_layout = QVBoxLayout()

        # File loading controls
        self.load_btn = QPushButton("加载 AEDAT4 文件")
        self.output_dir_btn = QPushButton("选择输出目录")
        self.checkpoint_btn = QPushButton("选择模型检查点")  # New button
        self.file_label = QLabel("当前文件：无")
        self.output_dir_label = QLabel("输出目录：无")
        self.checkpoint_label = QLabel("检查点：无")  # New label for checkpoint

        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.output_dir_btn)
        left_layout.addWidget(self.checkpoint_btn)  # Add to layout
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.output_dir_label)
        left_layout.addWidget(self.checkpoint_label)  # Add to layout

        # Video playback controls
        video_controls_layout = QVBoxLayout()

        play_controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.pause_btn = QPushButton("暂停")
        self.stop_btn = QPushButton("停止")
        play_controls_layout.addWidget(self.play_btn)
        play_controls_layout.addWidget(self.pause_btn)
        play_controls_layout.addWidget(self.stop_btn)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("播放速度:"))
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setMinimum(1)
        self.speed_spinbox.setMaximum(10)
        self.speed_spinbox.setValue(1)
        speed_layout.addWidget(self.speed_spinbox)

        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        video_controls_layout.addLayout(play_controls_layout)
        video_controls_layout.addWidget(self.time_slider)
        video_controls_layout.addLayout(speed_layout)
        video_controls_layout.addWidget(self.time_label)

        # Recognition-specific initialization
        self.recognizer = None
        self.checkpoint_path = "E:/Tro_model/checkpoint_finetuned_100.pth"
        self.device = "cpu"  # Change to "cuda" if GPU available
        self.channels = 128
        self.T = 16
        self.batch_size = 1



        # Processing buttons
        self.segment_btn = QPushButton("执行分割")
        self.recognize_btn = QPushButton("执行识别")

        left_layout.addLayout(video_controls_layout)
        left_layout.addWidget(self.segment_btn)
        left_layout.addWidget(self.recognize_btn)
        left_layout.addStretch()

        # Right results panel
        right_layout = QVBoxLayout()
        self.result_label = QLabel("识别结果：")
        self.result_list = QListWidget()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)

        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.result_list)
        right_layout.addWidget(QLabel("控制台输出："))
        right_layout.addWidget(self.console_output)

        # Combine left and right layouts
        bottom_layout.addLayout(left_layout, 1)
        bottom_layout.addLayout(right_layout, 3)

        # Add splitter between top and bottom
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_frame)
        splitter.addWidget(bottom_frame)
        splitter.setSizes([400, 500])

        main_layout.addWidget(splitter)

        # Connect buttons for file operations
        self.load_btn.clicked.connect(self.load_file)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        self.checkpoint_btn.clicked.connect(self.select_checkpoint)  # Connect new button
        self.segment_btn.clicked.connect(self.run_segmentation)
        self.recognize_btn.clicked.connect(self.run_recognition)

        # Connect video playback buttons
        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)
        self.stop_btn.clicked.connect(self.stop_video)
        self.time_slider.sliderMoved.connect(self.seek_video)
        self.speed_spinbox.valueChanged.connect(self.change_speed)

        # Connect result_list selection to update recognition display
        self.result_list.currentRowChanged.connect(self.update_recognition_display)



        self.video_thread = None
        self.total_duration = 0
        self.visualization_data = {}

        # Disable buttons initially
        self.segment_btn.setEnabled(False)
        self.recognize_btn.setEnabled(False)
        self.checkpoint_btn.setEnabled(True)  # Always enabled
        self.play_btn.setEnabled(bool(self.file_path))
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.time_slider.setEnabled(bool(self.file_path))
        self.speed_spinbox.setEnabled(bool(self.file_path))

    def load_file(self):
        default_dir = "D:\\eventVision\\collecting\\4_23\\"  # 设置默认目录
        if not os.path.exists(default_dir):
            default_dir = os.path.expanduser("~")
        file_dialog = QFileDialog()
        file_dialog.setDirectory(default_dir)
        path, _ = file_dialog.getOpenFileName(self, "选择 AEDAT4 文件", default_dir, "AEDAT4 Files (*.aedat4)")
        if path:
            self.file_path = path
            self.file_label.setText(f"当前文件：{os.path.basename(path)}")
            self.update_button_states()
            self.play_btn.setEnabled(True)
            self.time_slider.setEnabled(True)
            self.speed_spinbox.setEnabled(True)


    def update_segment_display(self, row):
        if row >= 0 and row < len(self.visualization_data["segments"]):
            self.segment_canvas.plot_segment(
                self.visualization_data["segments"][row],
                self.visualization_data["action_segments"]
            )

    def show_all_events(self):
        self.segment_canvas.plot_all_events(
            self.visualization_data["total_events"],
            self.visualization_data["positive_events"],
            self.visualization_data["negative_events"],
            self.visualization_data["action_segments"]
        )

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(f"输出目录：{dir_path}")
            self.console_output.append(f"已选择输出目录: {dir_path}")
            self.update_button_states()  # Add this line
            self.recognize_btn.setEnabled(bool(self.file_path and self.output_dir and self.checkpoint_path))
    def update_button_states(self):
        self.segment_btn.setEnabled(self.file_path != "" and self.output_dir != "")
        # Recognition depends on segmentation being done first
        self.recognize_btn.setEnabled(False)

    def play_video(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.set_paused(False)
            return

        if self.file_path:
            # Create and start video thread
            self.video_thread = VideoThread(self.file_path)
            self.video_thread.frame_signal.connect(self.update_video_frame)
            self.video_thread.progress_signal.connect(self.update_progress)
            self.video_thread.duration_signal.connect(self.set_duration)
            self.video_thread.start()

            # Update UI
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)

    def pause_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_paused(True)

    def stop_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.time_slider.setValue(0)
            self.time_label.setText("00:00:00 / " + self.format_time(self.total_duration))
            # Reset video display
            self.video_label.clear()
            self.video_label.setText("视频播放区域")
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def seek_video(self, position):
        if self.video_thread and self.video_thread.isRunning():
            percent = position / 1000.0
            self.video_thread.jump_to_percent(percent)

    def change_speed(self, speed):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_speed(float(speed))

    def update_video_frame(self, frame):
        # Convert frame to QImage
        if len(frame.shape) == 2:
            height, width = frame.shape
            bytes_per_line = width
            format = QImage.Format.Format_Grayscale8
        elif len(frame.shape) == 3:
            height, width, channel = frame.shape
            bytes_per_line = width * channel  # Adjust for multi-channel images
            format = QImage.Format.Format_RGB888 if channel == 3 else QImage.Format.Format_Grayscale8
        else:
            print(f"Unexpected frame shape: {frame.shape}")
            return
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def update_recognition_display(self, row):
        if not hasattr(self, 'recognition_results') or row < 0 or row >= len(self.recognition_results):
            self.segment_canvas.plot_single_recognition_result(None)
            self.console_output.append("No recognition results available")
            return
        result = self.recognition_results[row]
        self.segment_canvas.plot_single_recognition_result(result)
        self.console_output.append(f"Displaying recognition result for Action {row + 1}")

    def plot_recognition_results(self, results):
        self.segment_canvas.plot_recognition_results(results)
        self.console_output.append("Recognition results visualized on the segmentation canvas")

    def update_progress(self, percent):
        percent = max(0.0, min(1.0, percent))  # 限制在 [0, 1]
        slider_value = int(percent * 1000)
        # Update slider without triggering the valueChanged signal
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(slider_value))
        self.time_slider.blockSignals(False)

        # Update time label
        current_time = percent * self.total_duration
        self.time_label.setText(
            f"{self.format_time(current_time)} / {self.format_time(self.total_duration)}"
        )

    def set_duration(self, duration):
        self.total_duration = duration
        self.time_label.setText(f"00:00:00 / {self.format_time(duration)}")

    def format_time(self, seconds):
        """Format seconds into HH:MM:SS string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def run_segmentation(self):
        if not self.file_path or not self.output_dir:
            self.console_output.append("请先选择 AEDAT4 文件和输出目录")
            return
        if not os.path.isfile(self.file_path):
            self.console_output.append(f"[错误] 文件 {self.file_path} 不存在")
            return
        if not os.path.isdir(self.output_dir) or not os.access(self.output_dir, os.W_OK):
            self.console_output.append(f"[错误] 输出目录 {self.output_dir} 不可写")
            return

        self.console_output.clear()
        self.segment_list.clear()
        self.console_output.append(f"开始处理文件：{self.file_path}")
        self.console_output.append(f"输出目录：{self.output_dir}")

        # Run segmentation directly in a worker thread
       # self.worker_thread.file_path = self.file_path
       #  self.worker_thread.output_dir = self.output_dir
        self.worker_thread = WorkerThread("", "direct_run", self.file_path, self.output_dir)
        self.worker_thread.output_signal.connect(self.process_output)
        self.worker_thread.result_signal.connect(self.handle_segmentation_result)
        self.worker_thread.image_signal.connect(self.update_segment_images)
        self.worker_thread.result_received_signal.connect(self.on_result_received)  # 连接新信号
        self.worker_thread.start()

    def select_checkpoint(self):
        default_dir = "E:/Tro_model" if os.path.exists("E:/Tro_model") else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型检查点",
            default_dir,
            "PyTorch Checkpoint (*.pth)"
        )
        if path:
            self.checkpoint_path = path
            self.checkpoint_label.setText(f"检查点：{os.path.basename(path)}")
            self.console_output.append(f"已选择检查点: {path}")
            self.recognizer = None  # Reset recognizer to force reinitialization
            self.recognize_btn.setEnabled(bool(self.file_path and self.output_dir and self.checkpoint_path))
        else:
            self.console_output.append("未选择检查点文件")

    def on_result_received(self, output_files):
        self.console_output.append(f"Received {len(output_files)} output files")
        aedat4_files = [f for f in output_files if f.endswith(".aedat4")]
        npz_files = [f for f in output_files if f.endswith(".npz")]
        for file_path in output_files:
            if file_path.endswith(".aedat4"):
                self.console_output.append(f"Generated AEDAT4: {file_path}")
            elif file_path.endswith(".npz"):
                self.console_output.append(f"Generated NPZ: {file_path}")
        self.npz_files = npz_files
        self.recognize_btn.setEnabled(bool(self.npz_files and self.checkpoint_path))

    def run_recognition(self):
        if not self.output_dir:
            self.console_output.append("[错误] 请先选择输出目录")
            return

        if not self.checkpoint_path:
            self.console_output.append("[错误] 请先选择模型检查点")
            return

        if not hasattr(self, 'npz_files') or not self.npz_files:
            self.console_output.append("[错误] 没有可用的 .npz 文件")
            return

        self.console_output.append(f"找到 {len(self.npz_files)} 个 .npz 文件，开始分类...")
        # Find .npz files in output_dir
        npz_file_paths = glob.glob(os.path.join(self.output_dir, "*.npz"))
        if not npz_file_paths:
            self.console_output.append("[错误] 输出目录中没有找到 .npz 文件")
            return

        self.console_output.append(f"找到 {len(npz_file_paths)} 个 .npz 文件，开始分类...")

        try:
            # Initialize recognizer if not already done or if checkpoint changed
            if self.recognizer is None:
                self.recognizer = Recognizer(
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                    channels=self.channels,
                    T=self.T,
                    batch_size=self.batch_size
                )
                self.recognizer.initialize_model()

            # Run classification
            results, summary = self.recognizer.classify_npz_files(self.npz_files)

            # Update UI with results
            self.result_list.clear()
            for i, result in enumerate(results):
                item_text = f"Action {i + 1}: {os.path.basename(result['file'])} - {result['label']} ({result['comm_code']})"
                self.result_list.addItem(item_text)
                self.console_output.append(item_text)  # Print to console

            # Display summary in console
            self.console_output.append(summary)

            if results:
                self.recognition_results = results
                print("Stored recognition_results:", len(self.recognition_results))
                self.result_list.setCurrentRow(0)  # Trigger initial visualization
            # Visualize recognition results on the segmentation canvas
           #  self.plot_recognition_results(results)  # New method call

        except Exception as e:
            self.console_output.append(f"[错误] 分类过程中发生错误: {str(e)}")
            import traceback
            self.console_output.append(traceback.format_exc())



    def process_output(self, line):
        self.console_output.append(line)
        if line.startswith("[识别]"):
            result = line.replace("[识别]", "").strip()
            self.result_list.addItem(result)

    def update_segment_images(self, visualization_data):
        print(f"Received visualization_data type: {type(visualization_data)}")  # Debug
        print(f"visualization_data: {visualization_data[:100]}" if isinstance(visualization_data,
                                                                              list) else f"visualization_data keys: {list(visualization_data.keys())}")  # Debug

        self.segment_list.clear()
        self.visualization_data = {}

        if isinstance(visualization_data, list):
            # Handle old list format (backward compatibility)
            self.console_output.append("[警告] 收到旧格式的列表数据，将按段列表处理")
            segments = visualization_data
            self.visualization_data = {
                "segments": segments,
                "total_events": np.array([]),  # Placeholder, as old format lacks these
                "positive_events": np.array([]),
                "negative_events": np.array([]),
                "action_segments": [(data["start"], data["end"]) for data in segments if isinstance(data, dict)]
            }
        elif isinstance(visualization_data, dict):
            # Handle new dictionary format
            self.visualization_data = visualization_data
            segments = visualization_data.get("segments", [])
        else:
            self.console_output.append(
                f"[错误] 预期 visualization_data 是字典或列表，实际类型是 {type(visualization_data)}")
            return

        for data in segments:
            if not isinstance(data, dict):
                self.console_output.append(f"[错误] 预期 segment_data 是字典，实际类型是 {type(data)}")
                continue
            self.segment_list.addItem(f"动作 {data['index']}: 帧 {data['start']} - {data['end']}")

        if segments:
            self.segment_list.setCurrentRow(0)
        self.show_all_btn.setEnabled(bool(segments))

    def update_segment_display(self, row):
        if row >= 0 and row < len(self.visualization_data.get("segments", [])):
            self.segment_canvas.plot_segment(
                self.visualization_data["segments"][row],
                self.visualization_data["action_segments"]
            )
    def handle_segmentation_result(self, output_files):
        if output_files:
            self.console_output.append(f"分割完成，共生成 {len(output_files)} 个文件")
            self.recognize_btn.setEnabled(True)
        else:
            self.console_output.append("分割过程未生成文件")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())