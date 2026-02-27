import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DATA_DIR = "MN40_Surface_Voxels"
CSV_PATH = "MN40_Vector_Labels.csv"
OUTPUT_PLOT = "verify_chair_0965.png"
GRID_SIZE = 64
TARGET_MODEL = "chair_0965"

def plot_3d_voxel_and_vector(ax, voxel_grid, vector, title):
    coords = np.argwhere(voxel_grid == 1)
    
    if len(coords) == 0:
        ax.set_title("Empty Voxel Grid")
        return

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Plot the 3D surface points
    ax.scatter(x, y, z, c='steelblue', s=1, alpha=0.3)

    # Calculate center of the grid for the vector origin
    center = GRID_SIZE / 2.0

    # Scale the vector for visualization (make it 30 units long)
    v_x = vector[0] * 30.0
    v_y = vector[1] * 30.0
    v_z = vector[2] * 30.0

    # Draw the Gravity Vector (Red Arrow)
    ax.quiver(center, center, center, 
              v_x, v_y, v_z, 
              color='red', linewidth=3, arrow_length_ratio=0.2)
    
    # Draw a small black dot at the center origin
    ax.scatter([center], [center], [center], color='black', s=50)

    # Format the plot
    ax.set_xlim([0, GRID_SIZE])
    ax.set_ylim([0, GRID_SIZE])
    ax.set_zlim([0, GRID_SIZE])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)
    
    # Set a good default viewing angle
    ax.view_init(elev=20, azim=45)

def verify_specific_target():
    print(f"Loading metadata from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print("Error: CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Isolate only the augmentations for the target model
    target_df = df[df['voxel_id'].str.contains(TARGET_MODEL)]
    
    if len(target_df) == 0:
        print(f"Error: Could not find any entries for {TARGET_MODEL} in the CSV.")
        return
        
    print(f"Found {len(target_df)} augmented variations of {TARGET_MODEL}.")
    
    # Limit to maximum 3 plots for visual clarity
    plot_count = min(3, len(target_df))
    samples = target_df.head(plot_count)
    
    fig = plt.figure(figsize=(18, 6))
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        voxel_id = row['voxel_id']
        file_path = os.path.join(DATA_DIR, voxel_id + '.npy')
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        voxel_grid = np.load(file_path)
        
        # Extract the target vector
        vector = np.array([row['v_x'], row['v_y'], row['v_z']])
        
        # Create 3D subplot
        ax = fig.add_subplot(1, plot_count, i + 1, projection='3d')
        
        title = f"ID: {voxel_id}\nTarget Gravity Vector: [{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]"
        plot_3d_voxel_and_vector(ax, voxel_grid, vector, title)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"Visual verification saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    verify_specific_target()