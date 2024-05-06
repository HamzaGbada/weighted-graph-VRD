from torch import Tensor, is_tensor, from_numpy
import torch.nn as nn


class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channel, out_dim) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 2, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(True),
            nn.Linear(128, out_dim)
        )

    def forward(self, images: Tensor):
        if not is_tensor(images):
            images = from_numpy(images)

        images = self.cnn(images)
        images = self.flatten(images)
        images = self.linear(images)
        return images



class VGG16Encoder(nn.Module):
    def __init__(self, in_channel, out_dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(True),
            nn.Linear(128, out_dim)
        )

    def forward(self, images: Tensor):
        images = self.conv1(images)
        images = self.max_pool(images)
        images = self.conv2(images)
        images = self.max_pool(images)
        images = self.conv3(images)
        images = self.max_pool(images)
        images = self.conv4(images)
        # images = self.max_pool(images)
        images = self.conv5(images)
        # images = self.max_pool(images)
        images = self.flatten(images)
        images = self.linear(images)
        return images


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, residual=False) -> None:
        super().__init__()
        if residual:
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x += shortcut
        return nn.ReLU()(x)


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channel, residual_block, out_dim) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv1 = nn.Sequential(
            residual_block(64, 64),
            residual_block(64, 64)
        )

        self.conv2 = nn.Sequential(
            residual_block(64, 128, residual=True),
            residual_block(128, 128)
        )

        self.conv3 = nn.Sequential(
            residual_block(128, 256, residual=True),
            residual_block(256, 256)
        )

        self.conv4 = nn.Sequential(
            residual_block(256, 512, residual=True),
            residual_block(512, 512)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(True),
            nn.Linear(128, out_dim)
        )

    def forward(self, images):
        images = self.conv0(images)
        images = self.conv1(images)
        images = self.conv2(images)
        images = self.conv3(images)
        images = self.conv4(images)
        images = self.gap(images)
        images = self.flatten(images)
        images = self.linear(images)
        return images
