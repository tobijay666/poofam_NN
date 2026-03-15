import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.video as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
MODEL_VERSION = "V36_Physics_Vector_Cosine"
OUTPUT_DIR = f"{MODEL_VERSION}_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")

DATA_FOLDER = "MN40_Physics_Voxels"
CSV_PATH = "MN40_Physics_Vectors.csv"

BATCH_SIZE = 32
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.05

# --- 1. MODEL ARCHITECTURE (VECTOR OUTPUT) ---
class VectorResNet3D(nn.Module):
    def __init__(self):
        super(VectorResNet3D, self).__init__()
        
        print("Loading Pre-trained Kinetics-400 weights...")
        weights = models.R3D_18_Weights.KINETICS400_V1
        self.backbone = models.r3d_18(weights=weights)
        
        # Modify Input (3 channels -> 1 channel)
        original_conv1 = self.backbone.stem[0]
        new_conv1 = nn.Conv3d(1, original_conv1.out_channels, 
                              original_conv1.kernel_size, original_conv1.stride, 
                              original_conv1.padding, bias=False)
        with torch.no_grad():
            new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        self.backbone.stem[0] = new_conv1
        
        # Modify Output to predict 3 values (V_x, V_y, V_z)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3) 
        )

    def forward(self, x):
        raw_vector = self.backbone(x)
        # Force output to be a directional Unit Vector (Length = 1.0)
        unit_vector = F.normalize(raw_vector, p=2, dim=1)
        return unit_vector

# --- 2. DATASET ---
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
        
        target_size = (64, 64, 64)
        padded = np.zeros(target_size, dtype=np.float32)
        ex = min(voxel.shape[0], 64)
        ey = min(voxel.shape[1], 64)
        ez = min(voxel.shape[2], 64)
        padded[:ex, :ey, :ez] = voxel[:ex, :ey, :ez]
        
        tensor = torch.from_numpy(padded).unsqueeze(0) 
        target_vector = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return tensor, target_vector

# --- 3. LOSS & METRICS ---
def cosine_distance_loss(pred, target):
    # 1.0 for perfect alignment, -1.0 for opposite.
    # Loss = 1 - sim ensures 0.0 is perfect.
    sim = F.cosine_similarity(pred, target, dim=1)
    return torch.mean(1.0 - sim)

def calculate_angular_error(pred, target):
    # Returns the error in actual degrees
    sim = F.cosine_similarity(pred, target, dim=1)
    # Clamp to prevent floating point instability in acos
    sim = torch.clamp(sim, -1.0 + 1e-7, 1.0 - 1e-7)
    angles_rad = torch.acos(sim)
    return torch.rad2deg(angles_rad)

# --- 4. MAIN TRAINING LOOP ---
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from dataset.")
    
    file_ids = df['voxel_id'].values
    vector_data = df[['v_x', 'v_y', 'v_z']].values

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_ids, vector_data, test_size=0.2, random_state=42
    )

    train_dataset = VectorVoxelDataset(DATA_FOLDER, train_files, train_labels)
    val_dataset = VectorVoxelDataset(DATA_FOLDER, val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = VectorResNet3D().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    training_log = pd.DataFrame(columns=[
        'Epoch', 'Train_Loss', 'Train_Angle_Err', 'Val_Loss', 'Val_Angle_Err', 'Acc_15deg', 'LR'
    ])

    num_epochs = 40
    print("\n--- STARTING V36 VECTOR TRAINING ---")

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_angle_err = 0.0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            
            loss = cosine_distance_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            with torch.no_grad():
                err_deg = calculate_angular_error(preds, targets)
                train_angle_err += torch.sum(err_deg).item()
                
            total_train += inputs.size(0)
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / total_train
        avg_train_angle = train_angle_err / total_train

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_angle_err = 0.0
        total_val = 0
        correct_15deg = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                
                loss = cosine_distance_loss(preds, targets)
                val_loss += loss.item() * inputs.size(0)
                
                err_deg = calculate_angular_error(preds, targets)
                val_angle_err += torch.sum(err_deg).item()
                
                correct_15deg += torch.sum(err_deg < 15.0).item()
                total_val += inputs.size(0)

        avg_val_loss = val_loss / total_val
        avg_val_angle = val_angle_err / total_val
        acc_15 = (correct_15deg / total_val) * 100.0
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Results: Train Angle Err: {avg_train_angle:.2f}° | Val Angle Err: {avg_val_angle:.2f}° | Acc <15°: {acc_15:.1f}%")

        # --- LOGGING ---
        new_row = pd.DataFrame([{
            'Epoch': epoch + 1,
            'Train_Loss': avg_train_loss,
            'Train_Angle_Err': avg_train_angle,
            'Val_Loss': avg_val_loss,
            'Val_Angle_Err': avg_val_angle,
            'Acc_15deg': acc_15,
            'LR': current_lr
        }])
        training_log = pd.concat([training_log, new_row], ignore_index=True)
        training_log.to_csv(LOG_CSV_PATH, index=False)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{MODEL_VERSION}_ep{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{MODEL_VERSION}_final.pth"))

    # --- PLOTTING ---
    df = pd.read_csv(LOG_CSV_PATH)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Train_Angle_Err'], label='Train Error')
    plt.plot(df['Epoch'], df['Val_Angle_Err'], label='Val Error')
    plt.title('Angular Error (Degrees)')
    plt.xlabel('Epoch')
    plt.ylabel('Degrees Off-Target')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Acc_15deg'], label='Accuracy < 15°', color='green')
    plt.title('Validation Accuracy Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_VERSION}_metrics.png"))
    print("Training complete. Artifacts saved.")

if __name__ == "__main__":
    main()