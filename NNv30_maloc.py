import os
# --- FIX 1: HANDLE FRAGMENTATION ---
# This must be set BEFORE torch is imported (or as early as possible)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.video as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split
import math

# --- CONFIGURATION ---
MODEL_VERSION = "V33_Gold_Static"
OUTPUT_PLOT_DIR = f"{MODEL_VERSION}_Results"
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

NORM_FACTOR = 180.0 

# Paths
DATA_FOLDER = "MN_voxels_dataset_360_gold"
CSV_PATH = "MN_metadata_360_gold/MN_voxel_data_360.csv"

# --- 1. LOSS FUNCTION ---
class PeriodicMAELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        p_deg = pred * 180.0
        t_deg = target * 180.0
        diff = torch.abs(p_deg - t_deg)
        diff = diff % 360.0
        error = torch.min(diff, 360.0 - diff)
        return torch.mean(error)

# --- 2. MODEL ARCHITECTURE ---
class PretrainedResNet18_Regression(nn.Module):
    def __init__(self, num_outputs=2):
        super(PretrainedResNet18_Regression, self).__init__()
        
        print("Loading Pre-trained Kinetics-400 weights...")
        weights = models.R3D_18_Weights.KINETICS400_V1
        self.backbone = models.r3d_18(weights=weights)
        
        original_conv1 = self.backbone.stem[0]
        new_conv1 = nn.Conv3d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        self.backbone.stem[0] = new_conv1
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        return self.backbone(x)

# --- 3. DATASET CLASS ---
class VoxelDataset_Gold(Dataset):
    def __init__(self, folder_path, file_paths, labels):
        self.folder_path = folder_path
        self.file_paths = file_paths
        self.labels = labels 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        if not filename.endswith('.npy'):
            filename += '.npy'
            
        file_path = os.path.join(self.folder_path, filename)
        voxel = np.load(file_path, allow_pickle=True)
        
        fix_x = self.labels[idx, 0]
        fix_y = self.labels[idx, 1]
        
        # Padding
        target_size = (64, 64, 64)
        pad_x = (target_size[0] - voxel.shape[0]) // 2
        pad_y = (target_size[1] - voxel.shape[1]) // 2
        pad_z = (target_size[2] - voxel.shape[2]) // 2
        
        padded_voxel = np.zeros(target_size, dtype=np.uint8)
        ex = min(pad_x + voxel.shape[0], 64)
        ey = min(pad_y + voxel.shape[1], 64)
        ez = min(pad_z + voxel.shape[2], 64)
        padded_voxel[pad_x:ex, pad_y:ey, pad_z:ez] = voxel[:ex-pad_x, :ey-pad_y, :ez-pad_z]

        padded_voxel = np.expand_dims(padded_voxel, axis=0)
        voxel_tensor = torch.from_numpy(padded_voxel).float()
        
        label_tensor = torch.tensor([fix_x / NORM_FACTOR, fix_y / NORM_FACTOR], dtype=torch.float32)
        
        return voxel_tensor, label_tensor

if __name__ == '__main__':
    # --- 4. SETUP ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Training on NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        # Clear any leftover memory
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    if os.path.exists(DATA_FOLDER) and os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        file_ids = df['voxel_id'].values
        fix_data = df[["fix_x", "fix_y"]].values

        train_files, val_files, train_labels, val_labels = train_test_split(
            file_ids, fix_data, test_size=0.2, random_state=42
        )

        train_dataset = VoxelDataset_Gold(DATA_FOLDER, train_files, train_labels)
        val_dataset = VoxelDataset_Gold(DATA_FOLDER, val_files, val_labels)

        # --- FIX 2: REDUCE BATCH SIZE & WORKERS ---
        # Batch 64 -> 32 (Safe for 24GB GPU with 3D ResNet)
        BATCH_SIZE = 10
        # Num_workers 4 -> 2 (Reduces system overhead)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    else:
        print(f"Error: Dataset not found at {DATA_FOLDER}")
        exit()

    # --- 5. INITIALIZE ---
    model = PretrainedResNet18_Regression(num_outputs=2)
    model.to(device)

    criterion = PeriodicMAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 6. TRAINING ---
    num_epochs = 50
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

    print(f"\nSTARTING TRAINING V33 (Gold Dataset - Static)")
    print(f"Batch Size: {BATCH_SIZE}")
    print("-" * 60)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            total_train += inputs.size(0)
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / total_train

        model.eval()
        val_loss = 0.0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                total_val += inputs.size(0)

        avg_val_loss = val_loss / total_val
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Results: Train MAE: {avg_train_loss:.2f}° | Val MAE: {avg_val_loss:.2f}°")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{MODEL_VERSION}_checkpoint.pth")

    torch.save(model.state_dict(), f"{MODEL_VERSION}_final.pth")

    # --- PLOTTING ---
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_loss'], label='Train MAE')
    plt.plot(epochs_range, history['val_loss'], label='Val MAE')
    plt.title('Mean Absolute Error (Degrees)')
    plt.xlabel('Epochs')
    plt.ylabel('Error (Degrees)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, f"{MODEL_VERSION}_metrics.png"))
    plt.show()