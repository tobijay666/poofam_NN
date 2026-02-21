import pandas as pd
import numpy as np
import os
import scipy.ndimage
from tqdm import tqdm

# --- CONFIGURATION ---
# Source (Original Balanced Data)
SOURCE_VOXELS_DIR = 'MN_voxels_dataset_fully_balanced'
SOURCE_METADATA_CSV = 'MN_metadata_with_fix/MN_voxel_data_fix_simple.csv'

# New Destination (Overwriting the broken Gold dataset)
OUTPUT_VOXELS_DIR = 'MN_voxels_dataset_360_gold_v2'
OUTPUT_METADATA_CSV = 'MN_metadata_360_gold_v2/MN_voxel_data_360.csv'

os.makedirs(OUTPUT_VOXELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_METADATA_CSV), exist_ok=True)

# Parameters
TARGET_COUNT_PER_BIN = 50 
BIN_SIZE = 5

def apply_rotation_correct_axes(voxel, rot_x, rot_y):
    # --- CRITICAL FIX ---
    # X-Rotation: You verified this is Plane (0, 2)
    voxel = scipy.ndimage.rotate(voxel, rot_x, axes=(0, 2), reshape=False, order=0)
    
    # Y-Rotation: Orthogonal to X, Plane (0, 1)
    voxel = scipy.ndimage.rotate(voxel, rot_y, axes=(0, 1), reshape=False, order=0)
    
    return voxel

def generate_dataset():
    print(f"Loading source: {SOURCE_METADATA_CSV}")
    df = pd.read_csv(SOURCE_METADATA_CSV)
    
    new_records = []
    bins = list(range(0, 360, BIN_SIZE))
    
    print(f"Generating {TARGET_COUNT_PER_BIN} samples per bin (Total ~{len(bins)*TARGET_COUNT_PER_BIN})...")

    for target_angle_x in tqdm(bins):
        count = 0
        while count < TARGET_COUNT_PER_BIN:
            row = df.sample(1).iloc[0]
            
            # Calculate required rotation
            # New_Fix = Old_Fix + Rot  =>  Rot = New_Fix - Old_Fix
            jitter = np.random.randint(0, BIN_SIZE)
            desired_fix_x = target_angle_x + jitter
            rot_x = desired_fix_x - row['fix_x']
            
            # Randomize Y
            desired_fix_y = np.random.randint(0, 360)
            rot_y = desired_fix_y - row['fix_y']
            
            # Load
            src_path = os.path.join(SOURCE_VOXELS_DIR, row['voxel_id'] + '.npy')
            if not os.path.exists(src_path): continue
            voxel = np.load(src_path)
            
            # Apply Rotation (CORRECTED)
            new_voxel = apply_rotation_correct_axes(voxel, rot_x, rot_y)
            
            # Save
            new_id = f"{row['voxel_id']}_v2_{count}_{target_angle_x}"
            np.save(os.path.join(OUTPUT_VOXELS_DIR, new_id + '.npy'), new_voxel)
            
            # Metadata
            final_fix_x = (row['fix_x'] + rot_x) % 360
            final_fix_y = (row['fix_y'] + rot_y) % 360
            
            new_records.append({
                'voxel_id': new_id,
                'fix_x': final_fix_x,
                'fix_y': final_fix_y
            })
            count += 1

    pd.DataFrame(new_records).to_csv(OUTPUT_METADATA_CSV, index=False)
    print("✅ Dataset V2 Generated Successfully.")

if __name__ == "__main__":
    generate_dataset()