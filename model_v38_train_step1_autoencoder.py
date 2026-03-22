import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import pandas as pd
import time
from tqdm import tqdm



# --- ARCHITECTURAL COMPONENTS (Ref: Mescheder et al., 2019) ---

class ResnetBlock3D(nn.Module):
    """Symmetric 3D Residual Block for Geometric Feature Extraction."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class VoxelAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        # ENCODER: Compresses 64^3 -> 1^3 (512 channels)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            ResnetBlock3D(32, 64),
            nn.MaxPool3d(2), # 32
            ResnetBlock3D(64, 128),
            nn.MaxPool3d(2), # 16
            ResnetBlock3D(128, 256),
            nn.MaxPool3d(2), # 8
            ResnetBlock3D(256, 512),
            nn.MaxPool3d(2), # 4
            nn.AdaptiveAvgPool3d(1) 
        )
        self.flatten = nn.Flatten()
        self.latent_fc = nn.Linear(512, latent_dim)
        
        # DECODER: Expands 512 -> 64^3
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
            nn.Conv3d(32, 1, kernel_size=3, padding=1)
            # Sigmoid omitted here; handled by BCEWithLogitsLoss for stability
        )

    def forward(self, x):
        z = self.encoder(x)
        latent = self.latent_fc(self.flatten(z))
        d = self.decoder_fc(latent).view(-1, 512, 4, 4, 4)
        reconstruction = self.decoder(d)
        return reconstruction, latent

# --- DATASET WITH RAM CACHING ---

class VoxelDatasetAE(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob(os.path.join(folder_path, "*.npy"))
        self.cache = []
        print(f"Researcher Mode: Caching {len(self.files)} models into RAM...")
        for f in tqdm(self.files):
            voxel = np.load(f).astype(np.uint8)
            if voxel.shape != (64, 64, 64):
                padded = np.zeros((64, 64, 64), dtype=np.uint8)
                padded[:voxel.shape[0], :voxel.shape[1], :voxel.shape[2]] = voxel
                voxel = padded
            self.cache.append(voxel)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.cache[idx]).float().unsqueeze(0)
        return tensor

# --- TRAINING ENGINE ---

def train():
    # Config
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 1e-4 # Standard for 3D ResNets
    LATENT_DIM = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False  # Disable heavy 3D auto-tuning
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    # Load Dataset
    dataset = VoxelDatasetAE("./MN40_surface_voxels")
    
    # num_workers MUST be 0 when using large RAM caches on Windows
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    model = VoxelAutoencoder(latent_dim=LATENT_DIM).to(device)
    
    # AdamW is superior for weight decay (Loshchilov & Hutter, 2017)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    # BCEWithLogits is more numerically stable than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()

    # Logging
    history = []
    os.makedirs("AE_Logs", exist_ok=True)
    os.makedirs("AE_Checkpoints", exist_ok=True)

    print(f"Starting Training on {device}...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(loader)
        duration = time.time() - start_time
        
        # Log results
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'time_sec': duration
        })
        
        # Save CSV every epoch for safety
        pd.DataFrame(history).to_csv("AE_Logs/training_history.csv", index=False)
        
        print(f"Epoch {epoch+1} Summary: Loss {avg_loss:.6f} | Time {duration:.2f}s")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"AE_Checkpoints/ae_v38_ep{epoch+1}.pth")

    torch.save(model.state_dict(), "ae_pretrained_final.pth")
    print("Pre-training Complete. Log saved to AE_Logs/training_history.csv")

if __name__ == "__main__":
    train()