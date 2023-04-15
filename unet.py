import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # Downsample layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # Upsample layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Output layer
        self.conv_out = nn.Conv2d(64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Downsample path
        x1 = F.relu(self.conv1(x))
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x2 = F.relu(self.conv2(x2))
        x3 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x3 = F.relu(self.conv3(x3))
        x4 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x4 = F.relu(self.conv4(x4))
        # Upsample path
        x = F.relu(self.upconv1(x4))
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.upconv2(x))
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.upconv3(x))
        x = torch.cat([x, x1], dim=1)
        # Output
        x = self.conv_out(x)
        return x