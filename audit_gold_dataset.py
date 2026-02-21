import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.ndimage

# --- CONFIGURATION ---
DATA_FOLDER = "MN_voxels_dataset_360_gold"
CSV_PATH = "MN_metadata_360_gold/MN_voxel_data_360.csv"

def visualize_slice(voxel, title, ax):
    ax.imshow(np.max(voxel, axis=0), cmap='gray')
    ax.set_title(title, fontsize=9)
    ax.axis('off')

def audit_dataset():
    if not os.path.exists(CSV_PATH):
        print("Error: CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Find a sample with a large X fix (approx 90 degrees)
    # This makes it easy to see if it's tilted forward (correct) or sideways (wrong)
    samples = df[ (df['fix_x'] > 80) & (df['fix_x'] < 100) & (df['fix_y'].abs() < 10) ]
    
    if len(samples) == 0:
        print("No specific samples found, picking random.")
        samples = df.sample(3)
    else:
        samples = samples.sample(3)
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        file_path = os.path.join(DATA_FOLDER, row['voxel_id'] + '.npy')
        if not os.path.exists(file_path): continue

        voxel = np.load(file_path)
        
        # 1. Show the Voxel from the dataset
        visualize_slice(voxel, f"Dataset File\nLabel says Fix X={row['fix_x']:.0f}", axes[i, 0])
        
        # 2. Apply the Fix using the VERIFIED AXIS (0, 2)
        # If the dataset is correct, this should make the chair upright.
        # If the dataset was made with the wrong axis, this will twist it weirdly.
        fixed_voxel = scipy.ndimage.rotate(voxel, row['fix_x'], axes=(0, 2), reshape=False, order=0)
        
        visualize_slice(fixed_voxel, "Attempt to Fix\n(Using Axis 0,2)", axes[i, 1])

    plt.tight_layout()
    plt.savefig('audit_gold.png')
    print("Check 'audit_gold.png'.")
    print("If the 'Attempt to Fix' column looks UPRIGHT, the dataset is fine.")
    print("If the 'Attempt to Fix' column looks WRONG/SIDEWAYS, the dataset is CORRUPTED.")

if __name__ == "__main__":
    audit_dataset()