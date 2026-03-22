import os
# Set memory allocation configuration before torch imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from model_v37_occnet import OccNetEncoder_Vector

# --- CONFIGURATION ---
MODEL_VERSION = "V37_OccNet_Final"
OUTPUT_DIR = f"{MODEL_VERSION}_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")

DATA_FOLDER = "MN40_Physics_Voxels"
CSV_PATH = "MN40_Physics_Vectors.csv"

# RESEARCHER NOTE: 3D ResNets have massive activation volumes. 
# BATCH_SIZE 16 is the stable limit for 24GB VRAM to avoid fragmentation.
BATCH_SIZE = 16  
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NORM_FACTOR = 180.0

# --- 1. DATASET WITH RAM CACHING ---
class VectorVoxelDataset(Dataset):
    def __init__(self, folder_path, file_paths, labels):
        self.samples = []
        self.labels = labels
        
        print(f"Researcher Mode: Caching {len(file_paths)} models into System RAM...")
        for filename in tqdm(file_paths):
            if not filename.endswith('.npy'): filename += '.npy'
            path = os.path.join(folder_path, filename)
            
            # Load as uint8 to save RAM (256KB per file vs 1MB for float32)
            voxel = np.load(path).astype(np.uint8)
            
            # Pre-process padding to 64x64x64 once
            target_size = (64, 64, 64)
            padded = np.zeros(target_size, dtype=np.uint8)
            ex, ey, ez = [min(s, 64) for s in voxel.shape]
            padded[:ex, :ey, :ez] = voxel[:ex, :ey, :ez]
            
            self.samples.append(padded)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Convert to float32 and add channel dim during batch creation
        voxel = torch.from_numpy(self.samples[idx]).float().unsqueeze(0)
        target = torch.tensor(self.labels[idx], dtype=torch.float32)
        return voxel, target

# --- 2. LOSS & METRICS ---
def cosine_distance_loss(pred, target):
    sim = F.cosine_similarity(pred, target, dim=1)
    return torch.mean(1.0 - sim)

def calculate_angular_error(pred, target):
    sim = F.cosine_similarity(pred, target, dim=1)
    sim = torch.clamp(sim, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(sim))

# --- 3. MAIN PIPELINE ---
def main():
    device = torch.device("cuda")
    # Disable benchmark to prevent VRAM spike during initial algorithm search
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    
    print(f"Hardware: {torch.cuda.get_device_name(0)} | VRAM: 24GB")

    # Data Preparation
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    train_files, val_files, train_labels, val_labels = train_test_split(
        df['voxel_id'].values, df[['v_x', 'v_y', 'v_z']].values, test_size=0.15, random_state=42
    )

    train_dataset = VectorVoxelDataset(DATA_FOLDER, train_files, train_labels)
    val_dataset = VectorVoxelDataset(DATA_FOLDER, val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = OccNetEncoder_Vector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    # Logging Setup
    log_history = []

    print("\n--- STARTING V37 RESEARCH TRAINING ---")

    for epoch in range(40):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_err = 0.0, 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/40 [Train]")
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

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_err = 0.0, 0.0
        acc_15, acc_30 = 0, 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                
                v_loss = cosine_distance_loss(preds, targets)
                val_loss += v_loss.item() * inputs.size(0)
                
                errors = calculate_angular_error(preds, targets)
                val_err += torch.sum(errors).item()
                
                # Accuracy thresholds
                acc_15 += torch.sum(errors < 15.0).item()
                acc_30 += torch.sum(errors < 30.0).item()

        # --- METRIC CALCULATION ---
        epoch_time = time.time() - start_time
        metrics = {
            'Epoch': epoch + 1,
            'Train_Loss': train_loss / len(train_dataset),
            'Train_Err_Deg': train_err / len(train_dataset),
            'Val_Loss': val_loss / len(val_dataset),
            'Val_Err_Deg': val_err / len(val_dataset),
            'Val_Acc_15': (acc_15 / len(val_dataset)) * 100,
            'Val_Acc_30': (acc_30 / len(val_dataset)) * 100,
            'LR': optimizer.param_groups[0]['lr'],
            'Time': epoch_time
        }
        
        log_history.append(metrics)
        pd.DataFrame(log_history).to_csv(LOG_CSV_PATH, index=False)
        
        print(f"Results: Train Err: {metrics['Train_Err_Deg']:.2f}° | Val Err: {metrics['Val_Err_Deg']:.2f}°")
        print(f"         Acc <15°: {metrics['Val_Acc_15']:.1f}% | Acc <30°: {metrics['Val_Acc_30']:.1f}%")
        
        scheduler.step()

        # Checkpoint saving
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"checkpoint_ep{epoch+1}.pth"))

    # Final Save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model_v37.pth"))
    
    # Final Plotting
    df = pd.read_csv(LOG_CSV_PATH)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Train_Err_Deg'], label='Train Error')
    plt.plot(df['Epoch'], df['Val_Err_Deg'], label='Val Error')
    plt.title('Angular Error (Degrees)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Val_Acc_15'], label='Acc < 15°')
    plt.plot(df['Epoch'], df['Val_Acc_30'], label='Acc < 30°')
    plt.title('Accuracy Rate (%)')
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "training_metrics.png"))
    print(f"Research artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()