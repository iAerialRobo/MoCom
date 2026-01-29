import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime
from torchvision.datasets import DatasetFolder
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from syops import get_model_complexity_info


# Modify your NPZDataset class to support train/test splitting
class NPZDataset(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, loader=NPZDataset.load_npz_frames, extensions=('npz',),
                         target_transform=target_transform)
        self.transform = transform

        # Store sample paths by class for easier splitting
        self.samples_by_class = {}
        for path, label in self.samples:
            if label not in self.samples_by_class:
                self.samples_by_class[label] = []
            self.samples_by_class[label].append((path, label))

    @staticmethod
    def load_npz_frames(file_name: str) -> np.ndarray:
        '''
        :param file_name: path of the npz file that saves the frames
        :type file_name: str
        :return: frames
        :rtype: np.ndarray
        '''
        return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)

    def resize_frames(self, frames):
        # frames 的形状是 (16, 2, 480, 640)，需要变成 (16, 2, 128, 128)
        frames_resized = F.interpolate(frames, size=(128, 128), mode='bilinear', align_corners=False)
        return frames_resized

    def __getitem__(self, index):
        path, label = self.samples[index]
        frames = self.load_npz_frames(path)
        frames = torch.tensor(frames, dtype=torch.float32)
        frames = self.resize_frames(frames)
        
        #frames =  frames[:4, : ,:,:]
	# 下采样
      #  indices = torch.linspace(0, frames.shape[0] - 1, steps=4).long()  # [0, 5, 10, 15]
     #   frames = frames[indices, :,:,:]
        if self.transform:
            frames = self.transform(frames)

        return frames, label

    def create_train_test_split(self, test_size=0.7, random_state=42):
        """
        Create train and test subsets with stratification (equal class distribution)

        Args:
            test_size: Proportion of the dataset to include in the test split (0.25 = 25%)
            random_state: Random seed for reproducibility

        Returns:
            train_indices, test_indices: Lists of indices for training and testing
        """
        train_indices = []
        test_indices = []

        # For each class, split samples in 3:1 ratio
        for label in sorted(self.samples_by_class.keys()):
            class_samples = self.samples_by_class[label]
            class_indices = [i for i, (path, sample_label) in enumerate(self.samples)
                             if sample_label == label]

            # Split indices for this class
            class_train_indices, class_test_indices = train_test_split(
                class_indices, test_size=test_size, random_state=random_state
            )

            train_indices.extend(class_train_indices)
            test_indices.extend(class_test_indices)

            print(f"Class {label}: {len(class_train_indices)} train, {len(class_test_indices)} test samples")

        return train_indices, test_indices


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
           # *conv,
            layer.Linear(channels * 64, 128),  #  layer.Linear(channels * 4, 128),
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

class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 50),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        result = self.conv_fc(x)
        return result


def main():
    # python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8

    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')

    args = parser.parse_args()
    print(args)
    print("new activation function: ")
    # neuron.LIFNode   neuron.IzhikevichNode
    # net = DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    net = EventUAVNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
  #   ops, params = get_model_complexity_info(net, ())

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    print(net)


    net.to(args.device)
    # args.resume = '/media/mav_ficker/129e41ac-2a58-4c8c-8014-3432c588108c/UAVVGG_logs/T16_b2_sgd_lr0.1_c128/checkpoint_latest.pth'
    # args.data_dir = '/media/mav_ficker/129e41ac-2a58-4c8c-8014-3432c588108c/mav/data' # '/home/dwqdx3/my_drive_1/znb/data/UAV_event/data_frame'
    args.data_dir = '/home/zhangnb/Tro/data/data_remote_frame'
    full_dataset = NPZDataset(root=args.data_dir)

    # Create train/test split with 3:1 ratio (test_size=0.25)
    train_indices, test_indices = full_dataset.create_train_test_split(test_size=0.3)

    # Create train and test subsets using the indices
    train_set = Subset(full_dataset, train_indices)
    test_set = Subset(full_dataset, test_indices)

    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )


    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    args.opt = 'sgd'
    args.epochs = 100   # epoch represent what
    args.out_dir = '/home/zhangnb/Tro/source/log'
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    net.to(args.device)
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')
    print(out_dir)
    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 5).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 5).float()
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
