import torch
import os
import numpy as np
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate, neuron, layer
import argparse
from torch.utils.data import Dataset, DataLoader
import time
import datetime
from torchvision.datasets import DatasetFolder
import torch.nn as nn
from copy import deepcopy

class NPZDataset(Dataset):
    def __init__(self, file_paths, transform=None, labels=None):
        self.file_paths = file_paths
        self.transform = transform
        self.labels = labels

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
        if self.labels is not None:
            label = self.labels[index]
            return frames, label, path
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

def read_finetune_file(label_file_path, npz_dir):
    """
    Read finetune.txt to get filenames and labels for fine-tuning.
    Returns a list of full file paths and corresponding labels.
    """
    file_paths = []
    labels = []
    with open(label_file_path, 'r') as f:
        for line in f:
            # Skip empty lines or lines starting with #
            if line.strip() and not line.strip().startswith('#'):
                # Split by whitespace and ignore comments after #
                parts = line.split('#')[0].strip().split()
                if len(parts) >= 2:
                    filename, label = parts[0], parts[1]
                    file_path = os.path.join(npz_dir, filename)
                    if os.path.exists(file_path):
                        file_paths.append(file_path)
                        labels.append(int(label))
                    else:
                        print(f"Warning: File {file_path} not found, skipping.")
    if not file_paths:
        raise ValueError(f"No valid .npz files found in {label_file_path}")
    return file_paths, labels

def main():
    parser = argparse.ArgumentParser(description='Fine-tune and classify NPZ files using trained UAV action recognition model')
    parser.add_argument('-device', default='cuda:0', help='device to run the model on')
    parser.add_argument('-checkpoint', type=str, required=False, help='path to the trained model checkpoint')
    parser.add_argument('-npz-dir', type=str, required=False, help='directory containing .npz files')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-finetune-epochs', default=50, type=int, help='number of epochs for fine-tuning')
    parser.add_argument('-finetune-lr', default=0.005, type=float, help='learning rate for fine-tuning')
    parser.add_argument('-batch-size', default=16, type=int, help='batch size for fine-tuning')
    args = parser.parse_args()
    args.checkpoint = '/home/zhangnb/Tro/source/log_uav/T16_b32_sgd_lr0.1_c128/checkpoint_max.pth'
    args.npz_dir = '/home/zhangnb/Tro/data/recResult/'
    args.npz_dir = '/home/zhangnb/Tro/data/recResultSUM/'

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

    # Read finetune.txt for fine-tuning files and labels
    label_file = os.path.join(args.npz_dir, 'finetune.txt')
    if not os.path.exists(label_file):
        print(f"Label file {label_file} not found")
        return
    finetune_files, labels = read_finetune_file(label_file, args.npz_dir)

    # Create dataset and dataloader for fine-tuning
    dataset = NPZDataset(file_paths=finetune_files, labels=labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Fine-tuning
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.finetune_lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting fine-tuning with {len(finetune_files)} files for {args.finetune_epochs} epochs...")
    for epoch in range(args.finetune_epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        for frames, labels, _ in dataloader:
            frames, labels = frames.to(args.device), labels.to(args.device)
            frames = frames.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            optimizer.zero_grad()
            outputs = net(frames).mean(0)  # Average over time steps
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            functional.reset_net(net)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.finetune_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%, Time: {time.time()-start_time:.2f}s")

    # Save fine-tuned model
    fine_tuned_checkpoint = os.path.join(args.npz_dir, 'checkpoint_finetuned.pth')
    torch.save({'net': net.state_dict()}, fine_tuned_checkpoint)
    print(f"Fine-tuned model saved to {fine_tuned_checkpoint}")

    # Testing phase: Process all .npz files in args.npz_dir
    npz_files = [os.path.join(args.npz_dir, f) for f in os.listdir(args.npz_dir) if f.endswith('.npz')]
    if not npz_files:
        print(f"No .npz files found in {args.npz_dir} for testing")
        return

    net.eval()
    test_dataset = NPZDataset(file_paths=npz_files)  # No labels for testing
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    class_labels = ['Negative', 'inv_vShape', 'left_right', 'up_down', 'vShape']
    print(f"\nProcessing {len(npz_files)} .npz files for testing...")
    with torch.no_grad():
        for frames, path in test_dataloader:
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