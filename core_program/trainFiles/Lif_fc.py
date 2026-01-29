import os
import time
import argparse
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# NPZDataset class (unchanged)
class NPZDataset(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, loader=NPZDataset.load_npz_frames, extensions=('npz',),
                         target_transform=target_transform)
        self.transform = transform
        self.samples_by_class = {}
        for path, label in self.samples:
            if label not in self.samples_by_class:
                self.samples_by_class[label] = []
            self.samples_by_class[label].append((path, label))

    @staticmethod
    def load_npz_frames(file_name: str) -> np.ndarray:
        return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)

    def resize_frames(self, frames):
        frames_resized = F.interpolate(frames, size=(128, 128), mode='bilinear', align_corners=False)
        return frames_resized

    def __getitem__(self, index):
        path, label = self.samples[index]
        frames = self.load_npz_frames(path)
        frames = torch.tensor(frames, dtype=torch.float32)
        frames = self.resize_frames(frames)
    #    indices = torch.linspace(0, frames.shape[0] - 1, steps=4).long()
    #    frames = frames[indices, :, :, :]
        if self.transform:
            frames = self.transform(frames)
        return frames, label

    def create_train_test_split(self, test_size=0.7, random_state=42):
        train_indices = []
        test_indices = []
        for label in sorted(self.samples_by_class.keys()):
            class_samples = self.samples_by_class[label]
            class_indices = [i for i, (path, sample_label) in enumerate(self.samples)
                             if sample_label == label]
            class_train_indices, class_test_indices = train_test_split(
                class_indices, test_size=test_size, random_state=random_state
            )
            train_indices.extend(class_train_indices)
            test_indices.extend(class_test_indices)
            print(f"Class {label}: {len(class_train_indices)} train, {len(class_test_indices)} test samples")
        return train_indices, test_indices

# Modified EventUAVNet to match LIF fully connected architecture
class EventUAVNet(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(2 * 128 * 128, 5, bias=False),  # Input: 2 channels * 128x128, Output: 5 classes
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        return self.layer(x)

def main():
    parser = argparse.ArgumentParser(description='LIF UAV Action Recognition Training')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of UAV dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

    args = parser.parse_args()
    print(args)

    net = EventUAVNet(tau=args.tau)
    print(net)
    net.to(args.device)

    args.data_dir = '/home/zhangnb/Tro/data/data_small_frame'
    full_dataset = NPZDataset(root=args.data_dir)
    train_indices, test_indices = full_dataset.create_train_test_split(test_size=0.7)
    train_set = Subset(full_dataset, train_indices)
    test_set = Subset(full_dataset, test_indices)
    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_tau{args.tau}')
    if args.amp:
        out_dir += '_amp'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)  # [N, T, C, H, W]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 5).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        # Select frame at time t and process
                        frame_t = frame[:, t, :, :, :]  # [N, C, H, W]
                        encoded_frame = encoder(frame_t)
                        out_fr += net(encoded_frame)
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    frame_t = frame[:, t, :, :, :]  # [N, C, H, W]
                    encoded_frame = encoder(frame_t)
                    out_fr += net(encoded_frame)
                out_fr = out_fr / args.T
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

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 5).float()
                out_fr = 0.
                for t in range(args.T):
                    frame_t = frame[:, t, :, :, :]
                    encoded_frame = encoder(frame_t)
                    out_fr += net(encoded_frame)
                out_fr = out_fr / args.T
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
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # Save neuron voltage and spike data for analysis
    net.eval()
    output_layer = net.layer[-1]  # Output layer (LIFNode)
    output_layer.v_seq = []
    output_layer.s_seq = []

    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)

    with torch.no_grad():
        frame, label = test_set[0]
        frame = frame.to(args.device)  # [T, C, H, W]
        out_fr = 0.
        for t in range(args.T):
            frame_t = frame[t, :, :, :]  # [C, H, W]
            encoded_frame = encoder(frame_t)
            out_fr += net(encoded_frame)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()
        np.save(os.path.join(out_dir, "v_t_array.npy"), v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()
        np.save(os.path.join(out_dir, "s_t_array.npy"), s_t_array)

if __name__ == '__main__':
    main()