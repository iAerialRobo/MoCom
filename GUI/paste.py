import numpy as np
import dv_processing as dv
from datetime import timedelta
import os
import aedat
# Process event data from an AEDAT4 file
def process_event_data(input_file, resolution=(640, 480), frame_duration_ms=33):
    """
    Process event data from an AEDAT4 file and return frame-based statistics.

    Args:
        input_file (str): Path to the input AEDAT4 file
        resolution (tuple): Camera resolution (width, height)
        frame_duration_ms (int): Duration of each frame in milliseconds

    Returns:
        tuple: (frame_to_time, positive_events, negative_events, total_events)
            - frame_to_time (dict): Maps frame number to start timestamp
            - positive_events (np.array): Array of positive event counts per frame
            - negative_events (np.array): Array of negative event counts per frame
            - total_events (np.array): Array of total event counts per frame
    """
    frame_duration = timedelta(milliseconds=frame_duration_ms)
    frame_to_time = {}
    frame_to_time_real = {}
    positive_events = []
    negative_events = []
    global_index = 0

    reader = dv.io.MonoCameraRecording(input_file)
    filter = dv.noise.BackgroundActivityNoiseFilter(
        resolution,
        backgroundActivityDuration=timedelta(milliseconds=1)
    )

    accumulator = dv.EventStore()
    frame_count = 0
    current_frame_start_time = None

    def count_polarity_events(events_store):
        """Count positive and negative events."""
        positive_count = sum(1 for event in events_store if event.polarity() == 1)
        negative_count = sum(1 for event in events_store if event.polarity() == 0)
        return positive_count, negative_count

    def process_frame_stats(events_store, frame_idx, frame_start_time):
        """Process statistics for a frame and store in lists if counts are non-zero."""
        nonlocal global_index
        positive_count, negative_count = count_polarity_events(events_store)
        if positive_count > 0 or negative_count > 0:
            frame_to_time[frame_idx] = frame_start_time
            frame_to_time_real[global_index] = frame_start_time
            positive_events.append(positive_count)
            negative_events.append(negative_count)
            global_index += 1

    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is not None and len(events) > 0:
            if current_frame_start_time is None:
                current_frame_start_time = events[0].timestamp()

            for event in events:
                event_time = event.timestamp()
                time_since_frame_start = timedelta(microseconds=event_time - current_frame_start_time)

                if time_since_frame_start >= frame_duration:
                    filter.accept(accumulator)
                    filtered_events = filter.generateEvents()
                    process_frame_stats(filtered_events, frame_count, current_frame_start_time)

                    frames_elapsed = int(time_since_frame_start / frame_duration)
                    new_frame_start_time = (
                        current_frame_start_time +
                        frames_elapsed * frame_duration.total_seconds() * 1_000_000
                    )

                    for i in range(frame_count + 1, frame_count + frames_elapsed):
                        process_frame_stats(dv.EventStore(), i, current_frame_start_time)

                    accumulator = dv.EventStore()
                    current_frame_start_time = new_frame_start_time
                    frame_count += frames_elapsed

                accumulator.push_back(event)

    if len(accumulator) > 0:
        filter.accept(accumulator)
        filtered_events = filter.generateEvents()
        process_frame_stats(filtered_events, frame_count, current_frame_start_time)
        frame_count += 1

    positive_events = np.array(positive_events)
    negative_events = np.array(negative_events)
    total_events = positive_events + negative_events

    return frame_to_time_real, positive_events, negative_events, total_events

# Compute features
def compute_features(total_events, positive_events, window_size=10):
    """
    Compute features from event counts.

    Args:
        total_events (np.array): Array of total event counts per frame
        positive_events (np.array): Array of positive event counts per frame
        window_size (int): Window size for variance calculation

    Returns:
        tuple: (pos_ratio, variance)
            - pos_ratio (np.array): Ratio of positive events to total events
            - variance (np.array): Variance of total events within window
    """
    pos_ratio = positive_events / (total_events + 1e-6)
    variance = np.zeros(len(total_events))
    half_window = window_size // 2
    for i in range(len(total_events)):
        start = max(0, i - half_window)
        end = min(len(total_events), i + half_window + 1)
        variance[i] = np.var(total_events[start:end])
    return pos_ratio, variance

# Segment actions
def segment_actions(pos_ratio, variance, total_events, min_action_length=10):
    """
    Segment actions based on smoothed features.

    Args:
        pos_ratio (np.array): Ratio of positive events to total events
        variance (np.array): Variance of total events
        total_events (np.array): Array of total event counts per frame
        min_action_length (int): Minimum length of an action segment

    Returns:
        list: List of tuples (start, end) representing action segments
    """
    pos_ratio_smooth = np.convolve(pos_ratio, np.ones(5) / 5, mode='same')
    variance_smooth = np.convolve(variance, np.ones(5) / 5, mode='same')

    pos_ratio_threshold = 0.5
    variance_threshold = np.median(variance) * 0.5

    static_frames = (pos_ratio_smooth < pos_ratio_threshold) & (variance_smooth < variance_threshold)
    boundaries = np.where(np.diff(static_frames.astype(int)))[0]
    action_segments = []
    start = 0
    for boundary in boundaries:
        if not static_frames[start]:
            if boundary - start >= min_action_length:
                action_segments.append((start, boundary))
        start = boundary + 1
    if start < len(total_events) and not static_frames[start]:
        if len(total_events) - start >= min_action_length:
            action_segments.append((start, len(total_events) - 1))
    return action_segments

# Post-process actions
def post_process_actions(action_segments, min_duration_frames=30, max_gap_frames=10, min_duration_frames_3s=91):
    """
    Filter and merge action segments.

    Args:
        action_segments (list): List of tuples (start, end) for action segments
        min_duration_frames (int): Minimum duration for a segment to be kept
        max_gap_frames (int): Maximum gap between segments to merge
        min_duration_frames_3s (int): Minimum duration for final segments (3s equivalent)

    Returns:
        list: List of tuples (start, end) for filtered and merged action segments
    """
    filtered_segments = [(s, e) for s, e in action_segments if e - s + 1 >= min_duration_frames]
    if not filtered_segments:
        return []
    merged_segments = []
    current_start, current_end = filtered_segments[0]
    for next_start, next_end in filtered_segments[1:]:
        if next_start - current_end - 1 <= max_gap_frames:
            current_end = next_end
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_segments.append((current_start, current_end))
    final_segments = [(s, e) for s, e in merged_segments if e - s + 1 >= min_duration_frames_3s]
    return final_segments

# Convert frames to timestamps
def frames_to_timestamps(action_segments, frame_to_time, max_attempts=100):
    """
    Convert frame-based action segments to timestamp-based segments using frame_to_time.
    If a timestamp is None, try the next frame index up to max_attempts times.

    Args:
        action_segments (list): List of tuples (start_frame, end_frame) for actions
        frame_to_time (dict): Maps frame number to timestamp
        max_attempts (int): Maximum number of frame increments to try

    Returns:
        list: List of tuples (start_timestamp, end_timestamp) for valid action segments
    """
    if not isinstance(frame_to_time, dict):
        raise TypeError("frame_to_time must be a dictionary")

    action_timestamps = []
    for start, end in action_segments:
        start_time = None
        end_time = None
        start_idx = start
        end_idx = end
        attempts = 0

        while start_time is None and attempts < max_attempts:
            start_time = frame_to_time.get(start_idx)
            if start_time is None:
                start_idx += 1
                attempts += 1
        if start_time is None:
            print(f"Warning: Skipping action (Frame {start} to {end}) due to no valid start timestamp after {max_attempts} attempts")
            continue

        attempts = 0
        while end_time is None and attempts < max_attempts:
            end_time = frame_to_time.get(end_idx)
            if end_time is None:
                end_idx += 1
                attempts += 1
        if end_time is None:
            print(f"Warning: Skipping action (Frame {start} to {end}) due to no valid end timestamp after {max_attempts} attempts")
            continue

        action_timestamps.append((start_time, end_time))
        if start_idx != start or end_idx != end:
            print(f"Info: Used adjusted frames (Frame {start_idx} to {end_idx}) for action (Frame {start} to {end})")

    return action_timestamps

# Split aedat4 file
def split_aedat4_by_actions(input_aedat4, action_timestamps, output_prefix="action", resolution=(640, 480)):
    reader = dv.io.MonoCameraRecording(input_aedat4)
    print(f"Opened AEDAT4 file from [{reader.getCameraName()}] camera")
    output_files = []

    for i, (start_time, end_time) in enumerate(action_timestamps):
        events = reader.getEventsTimeRange(int(start_time), int(end_time))
        event_count = len(events)
        print(f"Action {i + 1}: {event_count} events between {start_time} and {end_time}")
        if events.isEmpty():
            print(f"Action {i + 1}: No events between {start_time} and {end_time}, skipping")
            continue

        output_file = f"{output_prefix}_{i + 1}.aedat4"
        config = dv.io.MonoCameraWriter.EventOnlyConfig(cameraName="DVXplorer_DXUS0002", resolution=resolution)
        writer = dv.io.MonoCameraWriter(output_file, config)
        writer.writeEvents(events)
        print(f"Action {i + 1}: Saved to {output_file} ({start_time} to {end_time})")
        output_files.append(output_file)

    return output_files

def load_aedat4(file_path: str) -> dict:
    decoder = aedat.Decoder(file_path)
    txyp = {'t': [], 'x': [], 'y': [], 'p': []}
    for packet in decoder:
        if packet["stream_id"] == 0:
            for event in packet["events"]:
                txyp['t'].append(event[0])
                txyp['x'].append(event[1])
                txyp['y'].append(event[2])
                txyp['p'].append(event[3])
    for key in txyp:
        txyp[key] = np.asarray(txyp[key], dtype=np.int64)
    return txyp

def integrate_events_segment_to_frame(x: np.ndarray, y: np.ndarray, p: np.ndarray,
                                     H: int, W: int, j_l: int = 0, j_r: int = -1) -> np.ndarray:
    frame = np.zeros((2, H * W), dtype=np.float32)
    x = x[j_l:j_r].astype(int)
    y = y[j_l:j_r].astype(int)
    p = p[j_l:j_r]
    mask = [p == 0, p == 1]
    for c in range(2):
        position = y[mask[c]] * W + x[mask[c]]
        events_per_pos = np.bincount(position, minlength=H * W)
        frame[c] += events_per_pos
    return frame.reshape((2, H, W))

def cal_fixed_frames_segment_index(t: np.ndarray, split_by: str, frames_num: int) -> tuple:
    j_l = np.zeros(frames_num, dtype=int)
    j_r = np.zeros(frames_num, dtype=int)
    N = t.size

    if split_by == 'number':
        di = N // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di
        j_r[-1] = N
    elif split_by == 'time':
        dt = (t[-1] - t[0]) // frames_num
        for i in range(frames_num):
            t_l = t[0] + dt * i
            t_r = t_l + dt
            mask = (t >= t_l) & (t < t_r)
            indices = np.where(mask)[0]
            j_l[i] = indices[0] if indices.size > 0 else N
            j_r[i] = indices[-1] + 1 if indices.size > 0 else N
        j_r[-1] = N
    else:
        raise ValueError("split_by must be 'number' or 'time'.")
    return j_l, j_r

def integrate_events_to_frames(events: dict, split_by: str, frames_num: int, H: int, W: int) -> np.ndarray:
    t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
    j_l, j_r = cal_fixed_frames_segment_index(t, split_by, frames_num)
    frames = np.zeros((frames_num, 2, H, W), dtype=np.float32)
    for i in range(frames_num):
        frames[i] = integrate_events_segment_to_frame(x, y, p, H, W, j_l[i], j_r[i])
    return frames

def convert_aedat4_to_npz(input_file: str, output_dir: str, frames_num: int,
                          split_by: str = 'number', H: int = 480, W: int = 640) -> str:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {input_file}...")
    events = load_aedat4(input_file)
    if not events['t'].size:
        print(f"No events found in {input_file}, skipping.")
        return None

    frames = integrate_events_to_frames(events, split_by, frames_num, H, W)
    fname = os.path.splitext(os.path.basename(input_file))[0] + '.npz'
    output_path = os.path.join(output_dir, fname)
    np.savez(output_path, frames=frames)
    print(f"Saved frames to {output_path}")
    return output_path