import os
# Memory management for RTX 3090
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# --- ARCHITECTURE (Must match Step 01 Encoder exactly) ---

class ResnetBlock3D(nn.Module):
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

class OrientationRegressor(nn.Module):
    def __init__(self, pretrained_path, latent_dim=512):
        super().__init__()
        
        # 1. Encoder Backbone
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            ResnetBlock3D(32, 64),
            nn.MaxPool3d(2), 
            ResnetBlock3D(64, 128),
            nn.MaxPool3d(2), 
            ResnetBlock3D(128, 256),
            nn.MaxPool3d(2), 
            ResnetBlock3D(256, 512),
            nn.MaxPool3d(2), 
            nn.AdaptiveAvgPool3d(1) 
        )
        self.flatten = nn.Flatten()
        self.latent_fc = nn.Linear(512, latent_dim)
        
        # 2. Load Pre-trained Weights from Epoch 30
        if os.path.exists(pretrained_path):
            print(f"Researcher Mode: Loading pre-trained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path)
            # Filter to keep only encoder and latent_fc
            filtered_dict = {k: v for k, v in state_dict.items() if "decoder" not in k}
            self.load_state_dict(filtered_dict, strict=False)
        else:
            raise FileNotFoundError(f"Could not find checkpoint at {pretrained_path}")

        # 3. Task-Specific Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2) # Outputs: [Angle_X, Angle_Y]
        )

    def forward(self, x):
        z = self.encoder(x)
        feat = self.latent_fc(self.flatten(z))
        angles = self.regressor(feat)
        return angles

# --- DATASET WITH RAM CACHING ---

class VoxelDatasetSupervised(Dataset):
    def __init__(self, csv_path, voxel_dir):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.voxel_dir = voxel_dir
        self.samples = []
        self.labels = []
        
        print(f"Researcher Mode: Scanning {voxel_dir} for {len(self.df)} models...")
        
        found_count = 0
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            v_id = str(row['voxel_id'])
            v_id = v_id + '_aug0'
            # Try a few common path variations
            path_options = [
                os.path.join(self.voxel_dir, f"{v_id}.npy"),
                os.path.join(self.voxel_dir, v_id) if v_id.endswith('.npy') else ""
            ]
            
            actual_path = None
            for p in path_options:
                if p and os.path.exists(p):
                    actual_path = p
                    break
            
            if actual_path:
                voxel = np.load(actual_path).astype(np.uint8)
                if voxel.shape != (64, 64, 64):
                    padded = np.zeros((64, 64, 64), dtype=np.uint8)
                    # Handle different possible input shapes safely
                    dx, dy, dz = voxel.shape
                    padded[:min(dx,64), :min(dy,64), :min(dz,64)] = voxel[:min(dx,64), :min(dy,64), :min(dz,64)]
                    voxel = padded
                self.samples.append(voxel)
                self.labels.append([row['angle_x'] / 180.0, row['angle_y'] / 180.0])
                found_count += 1

        if found_count == 0:
            print(f"\nCRITICAL ERROR: 0 files were loaded from {voxel_dir}!")
            print(f"Check if your CSV 'voxel_id' column matches the filenames in the folder.")
            print(f"Example path tried: {os.path.join(self.voxel_dir, str(self.df.iloc[0]['voxel_id']) + '.npy')}")
            exit()
        else:
            print(f"Successfully cached {found_count} models.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        voxel = torch.from_numpy(self.samples[idx]).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return voxel, label

# --- UPDATED TRAINING CONFIG ---

Update : "Fine tuning donedef train():
    # Config
    EPOCHS = 100
    BATCH_SIZE = 16 
    
    # Verify these paths manually before running!
    PRETRAINED_FILE = "AE_Checkpoints/ae_v38_ep30.pth"
    CSV_PATH = "MN40_Best_Orientations.csv"
    VOXEL_DIR = "MN40_Surface_Voxels" # Ensure this matches your actual folder name
    
    if not os.path.exists(VOXEL_DIR):
        print(f"ERROR: Voxel directory '{VOXEL_DIR}' does not exist.")
        return

    device = torch.device("cuda")
    
    # Load Data
    dataset = VoxelDatasetSupervised(CSV_PATH, VOXEL_DIR)
    
    # Safety check for split
    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    
    print(f"Splitting dataset: {train_size} train, {val_size} val")
    
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Use num_workers=0 for Windows RAM cache
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    
    model = OrientationRegressor(pretrained_path=PRETRAINED_FILE).to(device)
    
    # Differential Learning Rates
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.latent_fc.parameters(), 'lr': 1e-5},
        {'params': model.regressor.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.SmoothL1Loss()
    
    history = []
    os.makedirs("FineTune_Logs", exist_ok=True)

    print(f"Starting Fine-tuning on {torch.cuda.get_device_name(0)}...")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                # MAE in degrees
                val_mae += torch.mean(torch.abs(preds * 180.0 - batch_y * 180.0)).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_mae = val_mae / len(val_loader)
        duration = time.time() - start_time
        
        scheduler.step(avg_val_mae)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_mae_deg': avg_val_mae,
            'lr': optimizer.param_groups[2]['lr']
        })
        
        pd.DataFrame(history).to_csv("FineTune_Logs/finetune_history.csv", index=False)
        print(f"Epoch {epoch+1}: Loss {avg_train_loss:.6f} | Val MAE {avg_val_mae:.2f}°")

    torch.save(model.state_dict(), "orientation_model_final.pth")

if __name__ == "__main__":
    train()