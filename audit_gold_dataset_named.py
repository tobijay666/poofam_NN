import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.ndimage

# --- CONFIGURATION ---
# Ensure these point to your CURRENT Gold dataset folder
DATA_FOLDER = "MN_voxels_dataset_360_gold"
CSV_PATH = "MN_metadata_360_gold/MN_voxel_data_360.csv"

def visualize_slice(voxel, title, ax):
    # Show projection (sum) for clearer view of the orientation
    ax.imshow(np.max(voxel, axis=0), cmap='gray')
    ax.set_title(title, fontsize=8)
    ax.axis('off')

def audit_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Filter for samples with a distinct rotation (approx 90 degrees)
    # This makes it obvious if the axis is wrong.
    samples = df[ (df['fix_x'] > 80) & (df['fix_x'] < 100) & (df['fix_y'].abs() < 10) ]
    
    if len(samples) == 0:
        print("No samples found in the 80-100 degree range. Picking random samples.")
        samples = df.sample(3)
    else:
        samples = samples.sample(3)
    
    print("\n" + "="*40)
    print("      AUDITING SAMPLES      ")
    print("="*40)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        voxel_id = row['voxel_id']
        filename = voxel_id + '.npy'
        file_path = os.path.join(DATA_FOLDER, filename)
        
        print(f"Sample {i+1}:")
        print(f"  ID:       {voxel_id}")
        print(f"  Fix X:    {row['fix_x']:.2f}")
        print(f"  Fix Y:    {row['fix_y']:.2f}")
        print(f"  Path:     {file_path}")
        print("-" * 40)

        if not os.path.exists(file_path): 
            print(f"File not found!")
            continue

        voxel = np.load(file_path)
        
        # 1. Show the Voxel from the dataset (AS IS)
        title_orig = f"ID: {voxel_id}\nDataset File (As Is)\nLabel Fix X={row['fix_x']:.0f}"
        visualize_slice(voxel, title_orig, axes[i, 0])
        
        # 2. Apply the Fix using the VERIFIED AXIS (0, 2)
        # If the dataset was generated correctly, this rotation should fix it.
        # If the dataset was generated with Axis (0,1), this will look weird.
        fixed_voxel = scipy.ndimage.rotate(voxel, row['fix_x'], axes=(0, 2), reshape=False, order=0)
        
        visualize_slice(fixed_voxel, "Attempt to Fix\n(Applying Rotation on Axis 0,2)", axes[i, 1])

    save_path = 'audit_gold_named.png'
    plt.savefig(save_path)
    print(f"\nVisual Audit saved to: {save_path}")
    print("Check the image. If the right column is NOT upright, the dataset is wrong.")

if __name__ == "__main__":
    audit_dataset()