import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split
import time

# Import YOUR Custom Architecture
from model_v27 import ThreeDResNet_V27

# --- CONFIGURATION ---
MODEL_VERSION = "V34_Custom_Gold_Logged"
OUTPUT_DIR = f"{MODEL_VERSION}_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")

NORM_FACTOR = 180.0 
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.05

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

# --- 2. DATASET CLASS ---
class VoxelDataset_Gold(Dataset):
    def __init__(self, folder_path, file_paths, labels):
        self.folder_path = folder_path
        self.file_paths = file_paths
        self.labels = labels 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        if not filename.endswith('.npy'): filename += '.npy'
        
        file_path = os.path.join(self.folder_path, filename)
        voxel = np.load(file_path, allow_pickle=True)
        
        fix_x = self.labels[idx, 0]
        fix_y = self.labels[idx, 1]
        
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
    # --- 3. SETUP ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Training on NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    if os.path.exists(DATA_FOLDER) and os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        file_ids = df['voxel_id'].values
        fix_data = df[["fix_x", "fix_y"]].values

        train_files, val_files, train_labels, val_labels = train_test_split(
            file_ids, fix_data, test_size=0.2, random_state=42
        )

        train_dataset = VoxelDataset_Gold(DATA_FOLDER, train_files, train_labels)
        val_dataset = VoxelDataset_Gold(DATA_FOLDER, val_files, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    else:
        print("Error: Dataset not found.")
        exit()

    # --- 4. INITIALIZE MODEL ---
    model = ThreeDResNet_V27(input_channels=1, num_outputs=2, dropout_prob=0.5)
    model.to(device)

    criterion = PeriodicMAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 5. LOGGING SETUP ---
    # Initialize the DataFrame or load existing if resuming (optional logic)
    log_columns = [
        'Epoch', 'Train_Loss', 'Train_MAE', 
        'Val_Loss', 'Val_MAE', 
        'Val_Acc_10deg', 'Val_Acc_20deg', # Accuracy within tolerance
        'Learning_Rate', 'Time_Sec'
    ]
    training_log = pd.DataFrame(columns=log_columns)

    # --- 6. TRAINING LOOP ---
    num_epochs = 50

    print(f"\nSTARTING TRAINING V34 (Logging to {LOG_CSV_PATH})")
    print("-" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()
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
        # For Periodic Loss, Loss value IS the MAE in degrees
        avg_train_mae = avg_train_loss 

        # Validation
        model.eval()
        val_loss = 0.0
        total_val = 0
        
        # Accuracy Counters
        count_within_10 = 0
        count_within_20 = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Calculate Loss
                batch_loss = criterion(outputs, labels).item()
                val_loss += batch_loss * inputs.size(0)
                
                # Calculate Accuracy (Per Sample)
                # We need raw errors for accuracy calculation
                p_deg = outputs * 180.0
                t_deg = labels * 180.0
                diff = torch.abs(p_deg - t_deg)
                diff = diff % 360.0
                errors = torch.min(diff, 360.0 - diff) # Shape: (Batch, 2)
                
                # Check if BOTH X and Y are within tolerance
                # max(error_x, error_y) must be < tolerance
                max_errors, _ = torch.max(errors, dim=1)
                
                count_within_10 += (max_errors < 10.0).sum().item()
                count_within_20 += (max_errors < 20.0).sum().item()
                
                total_val += inputs.size(0)

        avg_val_loss = val_loss / total_val
        avg_val_mae = avg_val_loss
        
        # Calculate Percentages
        acc_10 = (count_within_10 / total_val) * 100.0
        acc_20 = (count_within_20 / total_val) * 100.0
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - start_time
        
        print(f"Results: Train MAE: {avg_train_mae:.2f} | Val MAE: {avg_val_mae:.2f}")
        print(f"         Acc <10°: {acc_10:.1f}% | Acc <20°: {acc_20:.1f}%")

        # --- SAVE LOGS ---
        new_row = pd.DataFrame([{
            'Epoch': epoch + 1,
            'Train_Loss': avg_train_loss,
            'Train_MAE': avg_train_mae,
            'Val_Loss': avg_val_loss,
            'Val_MAE': avg_val_mae,
            'Val_Acc_10deg': acc_10,
            'Val_Acc_20deg': acc_20,
            'Learning_Rate': current_lr,
            'Time_Sec': epoch_duration
        }])
        
        training_log = pd.concat([training_log, new_row], ignore_index=True)
        training_log.to_csv(LOG_CSV_PATH, index=False)

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{MODEL_VERSION}_checkpoint.pth")

    # Final Save
    torch.save(model.state_dict(), f"{MODEL_VERSION}_final.pth")
    print(f"Training Complete. Log saved to {LOG_CSV_PATH}")

    # --- PLOTTING FROM CSV ---
    # Reload from CSV to ensure plot matches saved data
    df = pd.read_csv(LOG_CSV_PATH)

    plt.figure(figsize=(15, 5))

    # Plot 1: MAE
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Train_MAE'], label='Train MAE')
    plt.plot(df['Epoch'], df['Val_MAE'], label='Val MAE')
    plt.title('Mean Absolute Error (Degrees)')
    plt.xlabel('Epoch')
    plt.ylabel('Degrees')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Val_Acc_10deg'], label='Accuracy < 10°')
    plt.plot(df['Epoch'], df['Val_Acc_20deg'], label='Accuracy < 20°')
    plt.title('Validation Accuracy Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_VERSION}_metrics.png"))
    plt.show()