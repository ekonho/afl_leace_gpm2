"""ResNet-18/34 for CIFAR-sized inputs (32x32).

Modified from torchvision ResNet:
    - Replaced 7x7 conv + maxpool with 3x3 conv (standard for CIFAR)
    - Added feature extraction hook support
    - Separated backbone (feature extractor) from classification head
"""
from __future__ import annotations

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetBackbone(nn.Module):
    """ResNet backbone (feature extractor) for CIFAR-32x32.

    Outputs a feature vector of shape [B, feat_dim] after global average pooling.
    The feat_dim is nf * 8 * BasicBlock.expansion = nf * 8 for BasicBlock.
    """

    def __init__(self, block: Type[BasicBlock], num_blocks: List[int], nf: int = 64):
        super().__init__()
        self.in_planes = nf
        self.nf = nf

        self.conv1 = conv3x3(3, nf)
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(block, nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.feat_dim = nf * 8 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(1)  # [B, feat_dim]


class LinearHead(nn.Module):
    """Simple linear classification head."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes, bias=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc(features)


class ResNetCIFAR(nn.Module):
    """Full ResNet model for CIFAR, split into backbone + head.

    This separation is essential for DualOrtho:
        - backbone: feature extractor (subject to dual projection)
        - head: classification layer (drift detection target)
    """

    def __init__(self, num_classes: int, nf: int = 64,
                 block: Type[BasicBlock] = BasicBlock,
                 num_blocks: List[int] = [2, 2, 2, 2]):
        super().__init__()
        self.backbone = ResNetBackbone(block, num_blocks, nf=nf)
        self.head = LinearHead(self.backbone.feat_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_head_params(self):
        return self.head.parameters()


def resnet18_cifar(num_classes: int, nf: int = 64) -> ResNetCIFAR:
    return ResNetCIFAR(num_classes, nf=nf, block=BasicBlock, num_blocks=[2, 2, 2, 2])


def resnet34_cifar(num_classes: int, nf: int = 64) -> ResNetCIFAR:
    return ResNetCIFAR(num_classes, nf=nf, block=BasicBlock, num_blocks=[3, 4, 6, 3])
