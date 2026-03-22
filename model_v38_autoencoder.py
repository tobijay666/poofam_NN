import torch
import torch.nn as nn

class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class VoxelAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1), # 64
            ResnetBlock3D(32, 64),
            nn.MaxPool3d(2), # 32
            ResnetBlock3D(64, 128),
            nn.MaxPool3d(2), # 16
            ResnetBlock3D(128, 256),
            nn.MaxPool3d(2), # 8
            ResnetBlock3D(256, 512),
            nn.MaxPool3d(2), # 4
            nn.AdaptiveAvgPool3d(1) # 1x1x1
        )
        
        self.flatten = nn.Flatten()
        self.latent_fc = nn.Linear(512, latent_dim)
        
        # --- DECODER ---
        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1), # 8
            nn.BatchNorm3d(256), nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), # 16
            nn.BatchNorm3d(128), nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # 32
            nn.BatchNorm3d(64), nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),   # 64
            nn.BatchNorm3d(32), nn.ReLU(True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Output probability of occupancy [0, 1]
        )

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        z = self.flatten(z)
        latent = self.latent_fc(z)
        
        # Decode
        d = self.decoder_fc(latent)
        d = d.view(-1, 512, 4, 4, 4)
        reconstruction = self.decoder(d)
        
        return reconstruction, latent