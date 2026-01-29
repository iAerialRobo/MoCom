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
from copy import deepcopy

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# NPZDataset class
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
      #  indices = torch.linspace(0, frames.shape[0] - 1, steps=4).long()
  #      frames = frames[indices, :, :, :]
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


# SpikingResNet implementation
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.sn2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.sn3(out)
        return out


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(SpikingResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)  # Changed input channels to 2
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def spiking_resnet18(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):
    return _spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, spiking_neuron, **kwargs)


def spiking_resnet34(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):
    return _spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, spiking_neuron, **kwargs)


def spiking_resnet50(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):
    return _spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, spiking_neuron, **kwargs)


def _spiking_resnet(arch, block, layers, pretrained, progress, spiking_neuron, **kwargs):
    model = SpikingResNet(block, layers, spiking_neuron=spiking_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url({
                                                  'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
                                                  'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
                                                  'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                                              }[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


class EventUAVNet(nn.Module):
    def __init__(self, T: int, resnet_variant: str = 'resnet18', use_cupy=False):
        super().__init__()
        self.T = T
        resnet_models = {
            'resnet18': spiking_resnet18,
            'resnet34': spiking_resnet34,
            'resnet50': spiking_resnet50,
        }
        if resnet_variant not in resnet_models:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}. Choose from {list(resnet_models.keys())}")

        self.resnet = resnet_models[resnet_variant](
            pretrained=False,
            progress=True,
            spiking_neuron=neuron.LIFNode,
            num_classes=5,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )

        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy', instance=neuron.LIFNode)

    def forward(self, x: torch.Tensor):
        x_seq = self.resnet(x)
        fr = x_seq.mean(0)
        return fr


def main(args, output_file):
    original_stdout = sys.stdout
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print(args)

        net = EventUAVNet(T=args.T, resnet_variant=args.resnet_variant, use_cupy=args.cupy)
        print(net)
        net.to(args.device)

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

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            max_test_acc = checkpoint['max_test_acc']

        out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_{args.resnet_variant}')
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
                frame = frame.transpose(0, 1)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 5).float()

                if scaler is not None:
                    with amp.autocast():
                        out_fr = net(frame)
                        loss = F.mse_loss(out_fr, label_onehot)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out_fr = net(frame)
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
                    frame = frame.transpose(0, 1)
                    label = label.to(args.device)
                    label_onehot = F.one_hot(label, 5).float()
                    out_fr = net(frame)
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
            print(
                f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
            print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
            print(
                f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

        net.eval()
        output_layer = net.resnet.fc
        output_layer.v_seq = []
        output_layer.s_seq = []

        def save_hook(m, x, y):
            output_layer.v_seq.append(x[0].unsqueeze(0))
            output_layer.s_seq.append(y.unsqueeze(0))

        output_layer.register_forward_hook(save_hook)

        with torch.no_grad():
            frame, label = test_set[0]
            frame = frame.to(args.device)
            frame = frame.unsqueeze(1)
            out_fr = net(frame)
            out_spikes_counter_frequency = out_fr.cpu().numpy()
            print(f'Firing rate: {out_spikes_counter_frequency}')

            output_layer.v_seq = torch.cat(output_layer.v_seq)
            output_layer.s_seq = torch.cat(output_layer.s_seq)
            v_t_array = output_layer.v_seq.cpu().numpy().squeeze()
            np.save(os.path.join(out_dir, "v_t_array.npy"), v_t_array)
            s_t_array = output_layer.s_seq.cpu().numpy().squeeze()
            np.save(os.path.join(out_dir, "s_t_array.npy"), s_t_array)

        sys.stdout = original_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spiking ResNet UAV Action Recognition Training')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=40, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('-j', default=10, type=int, help='number of data loading workers')
    parser.add_argument('-out-dir', type=str, default='/home/zhangnb/Tro/source/log',
                        help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', default='sgd', type=str, help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-resnet-variant', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Spiking ResNet variant to use')

    args = parser.parse_args()

    datasets = [
        '/home/zhangnb/Tro/data/data_small_frame/',
        '/home/zhangnb/Tro/data/data_frame/',
        '/home/zhangnb/Tro/data/data_remote_frame/'
    ]
    output_files = [
        '/home/zhangnb/Tro/source/log/resnet_output_small_frame.txt',
        '/home/zhangnb/Tro/source/log/resnet_output_frame.txt',
        '/home/zhangnb/Tro/source/log/resnet_output_remote_frame.txt'
    ]

    for data_dir, output_file in zip(datasets, output_files):
        print(f"Running experiment with dataset: {data_dir}")
        args.data_dir = data_dir
        main(args, output_file)
        print(f"Finished experiment with dataset: {data_dir}. Output saved to: {output_file}")