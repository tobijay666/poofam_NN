import torch
import torch.nn as nn
import torchvision.models.video as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# --- CONFIGURATION ---
MODEL_VERSION = "V35_Hybrid_RF"
OUTPUT_DIR = f"{MODEL_VERSION}_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input Data
DATA_FOLDER = "MN_voxels_dataset_360_gold"
CSV_PATH = "MN_metadata_360_gold/MN_voxel_data_360.csv"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. FEATURE EXTRACTOR (Pre-trained CNN) ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading Pre-trained ResNet3D-18...")
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
        
        # Remove Classification Head
        # We want the 512-dimensional embedding
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

# --- 2. DATASET ---
class VoxelDataset(Dataset):
    def __init__(self, folder_path, df):
        self.folder_path = folder_path
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['voxel_id']
        if not filename.endswith('.npy'): filename += '.npy'
        
        path = os.path.join(self.folder_path, filename)
        voxel = np.load(path).astype(np.float32)
        
        # Padding
        target_size = (64, 64, 64)
        pad_x = (target_size[0] - voxel.shape[0]) // 2
        pad_y = (target_size[1] - voxel.shape[1]) // 2
        pad_z = (target_size[2] - voxel.shape[2]) // 2
        padded = np.zeros(target_size, dtype=np.float32)
        ex = min(pad_x + voxel.shape[0], 64)
        ey = min(pad_y + voxel.shape[1], 64)
        ez = min(pad_z + voxel.shape[2], 64)
        padded[pad_x:ex, pad_y:ey, pad_z:ez] = voxel[:ex-pad_x, :ey-pad_y, :ez-pad_z]
        
        tensor = torch.from_numpy(padded).unsqueeze(0)
        
        # Return ID as well for tracking
        return tensor, row['fix_x'], row['fix_y'], row['voxel_id']

# --- HELPER: CIRCULAR ERROR ---
def calculate_circular_error(true_vals, pred_vals):
    diff = np.abs(true_vals - pred_vals)
    diff = np.minimum(diff, 360.0 - diff)
    return diff

# --- 3. MAIN PIPELINE ---
def run_hybrid_pipeline():
    # A. Setup Data
    if not os.path.exists(CSV_PATH):
        print("Error: Dataset not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples.")
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(VoxelDataset(DATA_FOLDER, train_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    val_loader = DataLoader(VoxelDataset(DATA_FOLDER, val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # B. Extract Features
    print("\n--- Phase 1: Extracting Features (CNN) ---")
    extractor = FeatureExtractor().to(DEVICE)
    extractor.eval()
    
    def extract_and_save(loader, split_name):
        print(f"Extracting {split_name} set...")
        features_list = []
        meta_list = []
        
        with torch.no_grad():
            for inputs, y_x, y_y, ids in tqdm(loader):
                inputs = inputs.to(DEVICE)
                feats = extractor(inputs).cpu().numpy() # (Batch, 512)
                
                # Store batch data
                for i in range(len(ids)):
                    row = {
                        'voxel_id': ids[i],
                        'true_x': y_x[i].item(),
                        'true_y': y_y[i].item()
                    }
                    # Add features f_0 to f_511
                    for f_idx, val in enumerate(feats[i]):
                        row[f'f_{f_idx}'] = val
                    
                    meta_list.append(row)
        
        # Convert to DataFrame
        df_features = pd.DataFrame(meta_list)
        
        # Save CSV
        save_path = os.path.join(OUTPUT_DIR, f"features_{split_name}.csv")
        df_features.to_csv(save_path, index=False)
        print(f"Saved {split_name} features to {save_path}")
        return df_features

    # Extract and Save
    df_train = extract_and_save(train_loader, "train")
    df_val = extract_and_save(val_loader, "val")

    # Prepare Data for Random Forest
    # Columns starting with 'f_' are features
    feat_cols = [c for c in df_train.columns if c.startswith('f_')]
    
    X_train = df_train[feat_cols].values
    y_train_x = df_train['true_x'].values
    y_train_y = df_train['true_y'].values
    
    X_val = df_val[feat_cols].values
    y_val_x = df_val['true_x'].values
    y_val_y = df_val['true_y'].values

    # C. Train Random Forest
    print("\n--- Phase 2: Training Random Forest ---")
    
    rf_x = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, verbose=1)
    print("Fitting X-Axis Model...")
    rf_x.fit(X_train, y_train_x)
    
    rf_y = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, verbose=1)
    print("Fitting Y-Axis Model...")
    rf_y.fit(X_train, y_train_y)

    # D. Predict and Analyze
    print("\n--- Phase 3: Evaluation & Logging ---")
    
    pred_val_x = rf_x.predict(X_val)
    pred_val_y = rf_y.predict(X_val)
    
    # Calculate errors
    err_x = calculate_circular_error(y_val_x, pred_val_x)
    err_y = calculate_circular_error(y_val_y, pred_val_y)
    
    # Create Detailed Prediction Log
    pred_log = pd.DataFrame({
        'voxel_id': df_val['voxel_id'],
        'true_x': y_val_x,
        'pred_x': pred_val_x,
        'error_x': err_x,
        'true_y': y_val_y,
        'pred_y': pred_val_y,
        'error_y': err_y
    })
    
    pred_save_path = os.path.join(OUTPUT_DIR, "predictions_val.csv")
    pred_log.to_csv(pred_save_path, index=False)
    print(f"Detailed predictions saved to {pred_save_path}")

    # E. Summary Metrics
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)
    
    # Calculate Accuracy within tolerance
    acc_10_x = np.mean(err_x < 10) * 100
    acc_10_y = np.mean(err_y < 10) * 100
    
    print(f"\nFINAL RESULTS:")
    print(f"X-Axis MAE: {mae_x:.2f}° (Acc <10°: {acc_10_x:.1f}%)")
    print(f"Y-Axis MAE: {mae_y:.2f}° (Acc <10°: {acc_10_y:.1f}%)")
    
    # Save Summary
    summary_df = pd.DataFrame([{
        'MAE_X': mae_x,
        'MAE_Y': mae_y,
        'Acc_10_X': acc_10_x,
        'Acc_10_Y': acc_10_y
    }])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)

    # F. Save Models
    joblib.dump(rf_x, os.path.join(OUTPUT_DIR, 'rf_model_x.pkl'))
    joblib.dump(rf_y, os.path.join(OUTPUT_DIR, 'rf_model_y.pkl'))
    print("Models saved.")

if __name__ == "__main__":
    run_hybrid_pipeline()