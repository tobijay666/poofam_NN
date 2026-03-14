import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DATA_DIR = "MN40_Physics_Voxels"
CSV_PATH = "MN40_Physics_Vectors.csv"

def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2"""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s == 0:
        if c > 0: return np.eye(3)
        else: return -np.eye(3)
        
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def verify_printability():
    if not os.path.exists(CSV_PATH):
        print("Error: CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    samples = df.sample(3)
    
    fig = plt.figure(figsize=(15, 6))
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        voxel_path = os.path.join(DATA_DIR, row['voxel_id'] + '.npy')
        if not os.path.exists(voxel_path): continue
        
        voxel = np.load(voxel_path)
        vector = np.array([row['v_x'], row['v_y'], row['v_z']])
        
        # Extract surface points for plotting
        coords = np.argwhere(voxel == 1).astype(float)
        if len(coords) == 0: continue
        
        # Center the coordinates around 0
        coords -= 32.0
        
        # The vector points "Down". We want to align it with [0, 0, -1]
        target_down = np.array([0, 0, -1])
        R = rotation_matrix_from_vectors(vector, target_down)
        
        # Apply the rotation to the coordinates
        aligned_coords = np.dot(coords, R.T)
        
        # Shift back
        aligned_coords += 32.0
        
        # Plotting
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        x, y, z = aligned_coords[:, 0], aligned_coords[:, 1], aligned_coords[:, 2]
        
        ax.scatter(x, y, z, c='seagreen', s=1, alpha=0.3)
        
        # Draw the Print Bed (Floor)
        min_z = np.min(z)
        xx, yy = np.meshgrid(np.linspace(0, 64, 5), np.linspace(0, 64, 5))
        ax.plot_wireframe(xx, yy, np.full_like(xx, min_z), color='gray', alpha=0.4)
        
        ax.set_xlim([0, 64]); ax.set_ylim([0, 64]); ax.set_zlim([0, 64])
        ax.set_title(f"ID: {row['voxel_id']}\nSimulated Print Bed Alignment", fontsize=9)
        
        # View from slightly above the print bed
        ax.view_init(elev=15, azim=45)

    plt.tight_layout()
    plt.savefig('verify_physics_printability.png', dpi=150)
    print("Saved 'verify_physics_printability.png'. Please review the orientations.")

if __name__ == "__main__":
    verify_printability()