import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock3D(nn.Module):
    """
    Symmetric 3D Residual Block as used in Occupancy Networks.
    """
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut
        return self.relu(out)

class OccNetEncoder_Vector(nn.Module):
    def __init__(self, latent_dim=512):
        super(OccNetEncoder_Vector, self).__init__()
        
        # Initial Stem: No aggressive downsampling
        self.conv_in = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)

        # Symmetric Downsampling Layers
        # 64 -> 32
        self.layer1 = nn.Sequential(
            ResnetBlock3D(32, 64),
            nn.MaxPool3d(2)
        )
        # 32 -> 16
        self.layer2 = nn.Sequential(
            ResnetBlock3D(64, 128),
            nn.MaxPool3d(2)
        )
        # 16 -> 8
        self.layer3 = nn.Sequential(
            ResnetBlock3D(128, 256),
            nn.MaxPool3d(2)
        )
        # 8 -> 4
        self.layer4 = nn.Sequential(
            ResnetBlock3D(256, 512),
            nn.MaxPool3d(2)
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Regression Head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        raw_v = self.fc(x)
        
        # CRITICAL: L2 Normalization to ensure output is a unit vector
        unit_v = F.normalize(raw_v, p=2, dim=1)
        return unit_v