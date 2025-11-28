import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from spikingjelly.activation_based import functional, surrogate, neuron, layer
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

class Recognizer:
    def __init__(self, checkpoint_path, device='cpu', channels=128, T=16, batch_size=1):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.channels = channels
        self.T = T
        self.batch_size = batch_size
        self.model = None
        self.class_labels = ['Negative', 'inv_vShape', 'left_right', 'up_down', 'vShape']
        self.comm_codes = {
            'Negative': '0 (background and noise)',
            'inv_vShape': '0 (signal 0)',
            'left_right': 'start (signal start)',
            'up_down': 'end (signal end)',
            'vShape': '1 (signal 1)'
        }
        self.signal_parts = {
            'Negative': None,
            'inv_vShape': '0',
            'left_right': 'start',
            'up_down': 'end',
            'vShape': '1'
        }

    def initialize_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {self.checkpoint_path} not found")

        self.model = EventUAVNet(
            channels=self.channels,
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )
        functional.set_step_mode(self.model, 'm')
        self.model.to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        print(f"Loaded model from {self.checkpoint_path}")

    def classify_npz_files(self, npz_file_paths):
        if not npz_file_paths:
            return [], "No valid .npz files to classify"

        # Initialize dataset and dataloader
        test_dataset = NPZDataset(file_paths=npz_file_paths)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        signal_sequence = []
        results = []

        with torch.no_grad():
            for frames, paths in test_dataloader:
                frames = frames.to(self.device)
                frames = frames.transpose(0, 1)  # Adjust dimensions for model
                out_fr = self.model(frames).mean(0)
                probs = F.softmax(out_fr, dim=1)
                predicted_classes = probs.argmax(dim=1)
                probs = probs.cpu().numpy()
                predicted_classes = predicted_classes.cpu().numpy()

                for i in range(len(paths)):
                    predicted_class = predicted_classes[i]
                    predicted_label = self.class_labels[predicted_class]
                    prob = probs[i]
                    signal_part = self.signal_parts[predicted_label]

                    # Collect signal part if not Negative
                    if signal_part is not None:
                        signal_sequence.append(signal_part)

                    # Format result for display
                    result = {
                        "file": paths[i],
                        "label": predicted_label,
                        "class_id": predicted_class,
                        "comm_code": self.comm_codes[predicted_label],
                        "probs": {label: p for label, p in zip(self.class_labels, prob)}
                    }
                    results.append(result)

                functional.reset_net(self.model)

        # Generate summary
        summary = f"Processed {len(npz_file_paths)} .npz files for classification\n"
        for result in results:
            summary += f"\nFile: {result['file']}\n"
            summary += f"Predicted Class: {result['label']} (Class {result['class_id']})\n"
            summary += f"Communication Code: {result['comm_code']}\n"
            summary += "Class Probabilities:\n"
            for label, prob in result['probs'].items():
                summary += f"  {label} ({self.comm_codes[label]}): {prob:.4f}\n"

        summary += "\n" + "="*50 + "\n"
        summary += "Concatenated Signal Sequence\n"
        summary += "="*50 + "\n"
        if signal_sequence:
            truncated_sequence = []
            for signal in signal_sequence:
                truncated_sequence.append(signal)
                if signal == 'end':
                    break
            concatenated_signal = "".join(truncated_sequence)
            summary += f"Signal: {concatenated_signal}\n"
           #  summary += f"Signal: {concatenated_signal}\n"
            # Extract components
            start = signal_sequence[0]  # First bit is Start
            direction = signal_sequence[1]  # Second bit is Direction (0 for Forward, 1 for Backward)
            heading = signal_sequence[2:5]  # Next 3 bits for Heading (angle)
            distance = signal_sequence[5:7]  # Next 2 bits for Distance
            end = signal_sequence[7]  # Last bit is End

            direction_binary = "".join(direction)
            heading_binary = "".join(heading)
            distance_binary = "".join(distance)
            # Decode and append to summary
            direction_str = "Forward" if direction == "0" else "Backward"
            heading_multiplier = int(heading_binary, 2)  # Convert binary to integer (multiplier for alpha)
            heading_angle = f"α°" if heading_multiplier == 1 else f"{heading_multiplier}α°"
            distance_value = int(distance_binary, 2) * 0.1  # Convert binary to decimal (0.1m per unit)

            summary += f"Start: {start}\n"
            summary += f"Direction: {direction_str}\n"
            summary += f"Heading: {heading_angle}\n"
            summary += f"Distance: {distance_value} m\n"
            summary += f"End: {end}\n"
        else:
            summary += "No valid signals detected (all predictions were Negative).\n"

        return results, summary
