# -*- coding: utf-8 -*-
import torch
import time
import numpy as np
import os
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch import nn
from copy import deepcopy
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Model 1: EventUAVNet from main1.py
class EventUAVNetMain1(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        conv = []
        for i in range(2):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
         #   conv.append(layer.BatchNorm2d(channels))
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


# Model 2: DVSGestureNet from VGG_dvsGesture.py
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


# Model 3: EventUAVNet from CNN.py
class EventUAVNetCNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T
        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Flatten(),
            layer.Linear(channels * 32 * 32, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(channels * 4 * 4, 5, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        x_seq = self.conv_fc(x)
        fr = x_seq.mean(0)
        return fr


# Model 4: SpikingResNet from spiking_resnetUAV_experiments.py
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


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5, zero_init_residual=False,
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
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
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
                if isinstance(m, BasicBlock):
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
    return SpikingResNet(BasicBlock, [2, 2, 2, 2], spiking_neuron=spiking_neuron, **kwargs)


class EventUAVNetResNet(nn.Module):
    def __init__(self, T: int, resnet_variant: str = 'resnet18', use_cupy=False):
        super().__init__()
        self.T = T
        self.resnet = spiking_resnet18(
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


def measure_inference_speed(model, input_tensor, num_batches=10, num_repeats=100, device='cuda:1'):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    times = []

    with torch.no_grad():
        for _ in range(num_repeats):
            start_time = time.time()
            for _ in range(num_batches):
                _ = model(input_tensor)
                functional.reset_net(model)
            end_time = time.time()
            times.append((end_time - start_time) / num_batches)
            torch.cuda.synchronize() if device.startswith('cuda') else None

    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000  # Convert to milliseconds
    return avg_time, std_time


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: cuda:1 not available, using cpu")
    batch_size = 10
    T = 16
    channels = 128
    height, width = 128, 128
    input_shape = (T, batch_size, 2, height, width)
    input_tensor = torch.randn(input_shape).to(device)

    # Reshaped input for models that don't handle time dimension
    input_tensor_4d = input_tensor.view(T * batch_size, 2, height, width)  # [T*batch_size, channels, height, width]

    # Initialize models
    models = {
        'DVSGestureNet (VGG_dvsGesture.py)': DVSGestureNet(
            channels=channels,
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        ),
        'EventUAVNet (main1.py)': EventUAVNetMain1(
            channels=channels,
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        ),
        'EventUAVNet (CNN.py)': EventUAVNetCNN(
            T=T,
            channels=channels,
            use_cupy=False
        ),
        'SpikingResNet18 (spiking_resnetUAV_experiments.py)': EventUAVNetResNet(
            T=T,
            resnet_variant='resnet18',
            use_cupy=False
        )
    }

    results = {}
    for model_name, model in models.items():
        print(f"Measuring inference speed for {model_name} on cuda:1...")
        # Use 4D input for EventUAVNetMain1 and DVSGestureNet, 5D for others
        input_to_use = input_tensor_4d if model_name in ['EventUAVNet (main1.py)',
                                                         'DVSGestureNet (VGG_dvsGesture.py)'] else input_tensor
        avg_time, std_time = measure_inference_speed(model, input_to_use, device=device)
        results[model_name] = (avg_time, std_time)
        print(f"{model_name}: {avg_time:.2f} ms ± {std_time:.2f} ms per batch")

    # Save results to file
    output_dir = '/home/zhangnb/Tro/source/log'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'inference_speed_results.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Inference Speed Results (10 batches, batch_size=10, 100 repeats)\n")
        f.write("============================================================\n\n")
        for model_name, (avg_time, std_time) in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Average Time per Batch: {avg_time:.2f} ms\n")
            f.write(f"  Standard Deviation: {std_time:.2f} ms\n\n")
        f.write("Device: cuda:1\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()