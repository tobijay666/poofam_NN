import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from model_v37_occnet import OccNetEncoder_Vector

# --- CONFIGURATION ---
MODEL_VERSION = "V37_OccNet_Symmetric"
OUTPUT_DIR = f"{MODEL_VERSION}_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")

# Use the Physics-Based Dataset (The one we verified with red arrows)
DATA_FOLDER = "MN40_Physics_Voxels"
CSV_PATH = "MN40_Physics_Vectors.csv"

BATCH_SIZE = 64  # Optimized for RTX 3090
LEARNING_RATE = 1e-4 # Slower start for training from scratch
WEIGHT_DECAY = 1e-5

# --- 1. DATASET ---
class VectorVoxelDataset(Dataset):
    def __init__(self, folder_path, file_paths, labels):
        self.folder_path = folder_path
        self.file_paths = file_paths
        self.labels = labels 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        if not filename.endswith('.npy'): filename += '.npy'
        
        path = os.path.join(self.folder_path, filename)
        voxel = np.load(path).astype(np.float32)
        
        # Ensure 64x64x64
        target_size = (64, 64, 64)
        padded = np.zeros(target_size, dtype=np.float32)
        ex, ey, ez = [min(s, 64) for s in voxel.shape]
        padded[:ex, :ey, :ez] = voxel[:ex, :ey, :ez]
        
        tensor = torch.from_numpy(padded).unsqueeze(0) 
        target_vector = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return tensor, target_vector

# --- 2. LOSS & METRICS ---
def cosine_distance_loss(pred, target):
    sim = F.cosine_similarity(pred, target, dim=1)
    return torch.mean(1.0 - sim)

def calculate_angular_error(pred, target):
    sim = F.cosine_similarity(pred, target, dim=1)
    sim = torch.clamp(sim, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(sim))

# --- 3. MAIN ---
def main():
    # Hardware Optimization
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    print(f"Researcher Mode: Training on {torch.cuda.get_device_name(0)}")

    # Data Loading
    df = pd.read_csv(CSV_PATH)
    file_ids = df['voxel_id'].values
    vector_data = df[['v_x', 'v_y', 'v_z']].values

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_ids, vector_data, test_size=0.15, random_state=42
    )

    train_loader = DataLoader(VectorVoxelDataset(DATA_FOLDER, train_files, train_labels), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(VectorVoxelDataset(DATA_FOLDER, val_files, val_labels), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = OccNetEncoder_Vector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    log_data = []

    print("\n--- STARTING V37 SYMMETRIC ENCODER TRAINING ---")

    for epoch in range(40):
        # Train
        model.train()
        train_loss, train_err = 0.0, 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/40")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = cosine_distance_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            with torch.no_grad():
                train_err += torch.sum(calculate_angular_error(preds, targets)).item()
            
            loop.set_postfix(loss=loss.item())

        # Val
        model.eval()
        val_loss, val_err = 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                val_loss += cosine_distance_loss(preds, targets).item() * inputs.size(0)
                val_err += torch.sum(calculate_angular_error(preds, targets)).item()

        metrics = {
            'Epoch': epoch + 1,
            'Train_Loss': train_loss / len(train_files),
            'Train_Err': train_err / len(train_files),
            'Val_Loss': val_loss / len(val_files),
            'Val_Err': val_err / len(val_files),
            'LR': optimizer.param_groups[0]['lr']
        }
        log_data.append(metrics)
        pd.DataFrame(log_data).to_csv(LOG_CSV_PATH, index=False)
        
        print(f"Results: Train Err: {metrics['Train_Err']:.2f}° | Val Err: {metrics['Val_Err']:.2f}°")
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    print("Training Complete.")

if __name__ == "__main__":
    main()