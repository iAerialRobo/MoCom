import torch
import os
import numpy as np
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate, neuron, layer
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from copy import deepcopy
import time
import gc

class NPZDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def load_npz_frames(file_name: str) -> np.ndarray:
        return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)

    def resize_frames(self, frames):
        frames = torch.tensor(frames, dtype=torch.float32)
        frames_resized = F.interpolate(frames, size=(128, 128), mode='bilinear', align_corners=False)
        return frames_resized

    def __getitem__(self, index):
        path = self.file_paths[index]
        frames = self.load_npz_frames(path)
        frames = self.resize_frames(frames)
        if self.transform:
            frames = self.transform(frames)
        return frames, path

class EventUAVNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        conv = []
        for i in range(2):
            in_channels = 2 if i == 0 else channels
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(4, 4))
        self.extractFeature = nn.Sequential(*conv)
        self.flatten = layer.Flatten()
        self.dropout1 = layer.Dropout(0.5)
        self.conv_fc = nn.Sequential(
            layer.Linear(channels * 64, 128),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            layer.Linear(128, 50),
            spiking_neuron(**deepcopy(kwargs)),
            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        x = self.extractFeature(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        result = self.conv_fc(x)
        return result

def get_npz_files(npz_dir):
    file_paths = []
    for file_name in os.listdir(npz_dir):
        if file_name.endswith('.npz'):
            file_path = os.path.join(npz_dir, file_name)
            if os.path.exists(file_path):
                file_paths.append(file_path)
            else:
                print(f"File {file_path} not found, skipping.")
    if not file_paths:
        raise ValueError(f"No valid .npz files found in {npz_dir}")
    print(f"Found {len(file_paths)} valid .npz files in {npz_dir}")
    return file_paths

def main():
    parser = argparse.ArgumentParser(description='Classify NPZ files using trained UAV action recognition model')
    parser.add_argument('-device', default='cuda:1', help='device to run the model on')
    parser.add_argument('-checkpoint', type=str, required=False, help='path to the trained model checkpoint')
    parser.add_argument('-npz-dir', type=str, required=False, help='directory containing .npz files')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-batch-size', default=1, type=int, help='batch size for inference')
    args = parser.parse_args()

    args.checkpoint = '/home/zhangnb/Tro/model/checkpoint_finetuned_100.pth'
    args.npz_dir = '/home/zhangnb/Tro/data/test/3'

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    file_paths = get_npz_files(args.npz_dir)

    print(f"Using device: {args.device}")

    net = EventUAVNet(
        channels=args.channels,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )
    functional.set_step_mode(net, 'm')
    net.to(args.device)

    if not os.path.exists(args.checkpoint):
        print(f"Fine-tuned checkpoint {args.checkpoint} not found")
        return

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    net.load_state_dict(checkpoint['net'])
    print(f"Loaded fine-tuned model from {args.checkpoint}")
    print(f"Model loaded to device: {next(net.parameters()).device}")
    net.eval()

    test_dataset = NPZDataset(file_paths=file_paths)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_labels = ['Negative', 'inv_vShape', 'left_right', 'up_down', 'vShape']
    comm_codes = {
        'Negative': '0 (background and noise)',
        'inv_vShape': '0 (信号 0)',
        'left_right': 'start (信号 start)',
        'up_down': 'end (信号 end)',
        'vShape': '1 (信号 1)'
    }
    signal_parts = {
        'Negative': None,
        'inv_vShape': '0',
        'left_right': 'start',
        'up_down': 'end',
        'vShape': '1'
    }

    print(f"\nProcessing {len(file_paths)} .npz files for classification...")

    gc.collect()

    with torch.no_grad():
        for frames, paths in test_dataloader:
            frames = frames.to(args.device).float()
            frames = frames.transpose(0, 1)
            out_fr = net(frames).mean(0)
            probs = F.softmax(out_fr, dim=1)
            predicted_classes = probs.argmax(dim=1)
            functional.reset_net(net)
            break

    num_runs = 100
    times = []
    class_counts = []
    last_signal_sequence = []
    last_results = []
    total_samples = len(test_dataset)

    for run in range(num_runs):
        signal_sequence = []

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            for frames, paths in test_dataloader:
                frames = frames.to(args.device).float()
                frames = frames.transpose(0, 1)
                out_fr = net(frames).mean(0)
                probs = F.softmax(out_fr, dim=1)
                predicted_classes = probs.argmax(dim=1)
                probs = probs.cpu().numpy()
                predicted_classes = predicted_classes.cpu().numpy()

                for i in range(len(paths)):
                    predicted_class = predicted_classes[i]
                    predicted_label = class_labels[predicted_class]
                    signal_part = signal_parts[predicted_label]
                    if signal_part is not None:
                        signal_sequence.append(signal_part)
                    if run == num_runs - 1:
                        last_results.append((paths[i], predicted_label, predicted_class, probs[i]))
                functional.reset_net(net)

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        times.append(end_time - start_time)
        class_counts.append(len(signal_sequence))
        if run == num_runs - 1:
            last_signal_sequence = signal_sequence

    avg_time_per_sample = np.mean(times) / total_samples
    std_time_per_sample = np.std(times) / total_samples
    min_classes = min(class_counts)
    max_classes = max(class_counts)
    avg_classes = np.mean(class_counts)

    print(f"\nAverage Inference Time Per Sample: {avg_time_per_sample:.6f} ± {std_time_per_sample:.6f} seconds (average over {num_runs} runs)")
    print(f"Signal counts: min={min_classes}, max={max_classes}, avg={avg_classes:.2f}")

    print("\nDetailed Results for Last Run:")
    for path, predicted_label, predicted_class, prob in last_results:
        print(f"\nFile: {path}")
        print(f"Predicted Class: {predicted_label} (Class {predicted_class})")
        print(f"Communication Code: {comm_codes[predicted_label]}")
        print("Class Probabilities:")
        for j, (label, p) in enumerate(zip(class_labels, prob)):
            print(f"  {label} ({comm_codes[label]}): {p:.4f}")

    print("\n" + "="*50)
    print("Concatenated Signal Sequence (Last Run)")
    print("="*50)
    if last_signal_sequence:
        concatenated_signal = "".join(last_signal_sequence)
        print(f"Signal: {concatenated_signal}")
    else:
        print("No valid signals detected (all predictions were Negative).")

if __name__ == '__main__':
    main()
