import torch
import os
import numpy as np
import torch.nn.functional as F
# from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based import functional, surrogate, neuron, layer
import argparse
from torch.utils.data import Dataset, DataLoader
import time
import os
import argparse
import datetime
from torchvision.datasets import DatasetFolder
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

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
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))  # Added BatchNorm2d
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

def main():
    parser = argparse.ArgumentParser(description='Classify NPZ files using trained UAV action recognition model')
    parser.add_argument('-device', default='cuda:0', help='device to run the model on')
    parser.add_argument('-checkpoint', type=str, required=False, help='path to the trained model checkpoint')
    parser.add_argument('-npz-dir', type=str, required=False, help='directory containing .npz files')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    args = parser.parse_args()
    args.checkpoint = '/home/zhangnb/Tro/source/log_uav/T16_b32_sgd_lr0.1_c128/checkpoint_max.pth'
    args.npz_dir = '/home/zhangnb/Tro/data/recResult/'
    # Initialize the model
    net = EventUAVNet(
        channels=args.channels,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )
    functional.set_step_mode(net, 'm')
    net.to(args.device)

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    # Get list of .npz files
    npz_files = [os.path.join(args.npz_dir, f) for f in os.listdir(args.npz_dir) if f.endswith('.npz')]
    if not npz_files:
        print(f"No .npz files found in {args.npz_dir}")
        return

    # Create dataset and dataloader
    dataset = NPZDataset(file_paths=npz_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Class labels (adjust according to your dataset)
    # class_labels = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']  # Update with your actual class names
    class_labels = ['Negative','inv_vShape','left_right','up_down','vShape']
    print(f"Processing {len(npz_files)} .npz files...")
    with torch.no_grad():
        for frames, path in dataloader:
            frames = frames.to(args.device)
            frames = frames.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            out_fr = net(frames).mean(0)  # Average over time steps
            probs = F.softmax(out_fr, dim=1)  # Convert to probabilities
            predicted_class = probs.argmax(dim=1).item()
            predicted_label = class_labels[predicted_class]
            probs = probs.cpu().numpy()[0]

            # Output results
            print(f"\nFile: {path[0]}")
            print(f"Predicted Class: {predicted_label} (Class {predicted_class})")
            print("Class Probabilities:")
            for i, (label, prob) in enumerate(zip(class_labels, probs)):
                print(f"  {label}: {prob:.4f}")

            functional.reset_net(net)

if __name__ == '__main__':
    main()