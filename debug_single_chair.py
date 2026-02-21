import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.ndimage

# --- CONFIGURATION ---
# Target Model ID (No .npy extension here)
TARGET_ID = "chair_0298_x5_y100_aug_44_85"

# Paths
DATA_FOLDER = "../MN_voxels_dataset_360_gold"
CSV_PATH = "../MN_metadata_360_gold/MN_voxel_data_360.csv"

def visualize_slice(voxel, title, ax):
    # Project along axis 0 (Front/Side view)
    ax.imshow(np.max(voxel, axis=0), cmap='gray')
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def inspect_model():
    # 1. Load Metadata
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 2. Find the specific row
    row = df[df['voxel_id'] == TARGET_ID]
    
    if len(row) == 0:
        print(f"Error: ID '{TARGET_ID}' not found in CSV.")
        return
    
    # Extract data
    fix_x = row.iloc[0]['fix_x']
    fix_y = row.iloc[0]['fix_y']
    
    print(f"Found Model: {TARGET_ID}")
    print(f"Required Fix from CSV -> X: {fix_x:.2f}, Y: {fix_y:.2f}")

    # 3. Load Voxel File
    file_path = os.path.join(DATA_FOLDER, TARGET_ID + '.npy')
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
        
    voxel = np.load(file_path)

    # 4. Apply Fix (Using Verified Axis)
    # We apply X rotation on axes (0, 2) based on your previous visual confirmation
    # We apply Y rotation on axes (0, 1) assuming orthogonality
    
    # Apply Fix X
    fixed_voxel = scipy.ndimage.rotate(voxel, fix_x, axes=(0, 2), reshape=False, order=0)
    
    # Apply Fix Y
    fixed_voxel = scipy.ndimage.rotate(fixed_voxel, fix_y, axes=(0, 1), reshape=False, order=0)

    # 5. Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot Original
    visualize_slice(voxel, f"RAW INPUT\n{TARGET_ID}", axes[0])
    
    # Plot Fixed
    visualize_slice(fixed_voxel, f"AFTER FIX\nApplied X:{fix_x:.1f}, Y:{fix_y:.1f}", axes[1])
    
    plt.tight_layout()
    save_path = 'debug_single_chair.png'
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    inspect_model()