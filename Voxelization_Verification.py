import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# --- CONFIGURATION ---
# Point this to a raw .off file in your ModelNet10 folder
# Pick one that previously failed the watertight check if possible!
# TEST_OFF_FILE = "chair_0002.off" 
# Provide the full path if it's not in the same directory, e.g.:
TEST_OFF_FILE = "../modelnet40_manually_aligned/chair/train/chair_0002.off"

GRID_SIZE = 64
PADDING = 2 # Keep the object 2 voxels away from the edges
NUM_SAMPLES = 150000 # Number of points to sample from the surface

def create_surface_voxel_grid(mesh_path, rot_x_deg, rot_y_deg):
    print(f"Loading mesh: {mesh_path}")
    
    # 1. Load the raw mesh (force='mesh' bypasses scene loading)
    mesh = trimesh.load(mesh_path, force='mesh')
    print(f"Is watertight? {mesh.is_watertight} (Does not matter anymore!)")
    
    # 2. Mathematical Rotation (BEFORE Voxelization)
    # Convert degrees to radians
    rx = math.radians(rot_x_deg)
    ry = math.radians(rot_y_deg)
    
    # Create transformation matrices
    rot_matrix_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
    rot_matrix_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
    
    # Combine rotations (Apply X then Y)
    transform = trimesh.transformations.concatenate_matrices(rot_matrix_y, rot_matrix_x)
    mesh.apply_transform(transform)
    
    # 3. Normalization (Center and Scale)
    # Center the mesh at (0,0,0)
    bounding_box_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-bounding_box_center)
    
    # Scale to fit inside our grid minus padding
    max_extent = mesh.extents.max()
    scale_factor = (GRID_SIZE - (PADDING * 2)) / max_extent
    mesh.apply_scale(scale_factor)
    
    # 4. Surface Sampling
    # Randomly sample points uniformly across all faces
    points, _ = trimesh.sample.sample_surface(mesh, NUM_SAMPLES)
    
    # 5. Map to Voxel Grid
    # Shift points from centered (-30 to +30) to array indices (0 to 64)
    points_shifted = points + (GRID_SIZE / 2)
    
    # Round to nearest integer to get grid coordinates
    coords = np.round(points_shifted).astype(int)
    
    # Clip coordinates just in case floating point math pushes a point to 64
    coords = np.clip(coords, 0, GRID_SIZE - 1)
    
    # Create empty grid
    voxel_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    
    # Set voxels to 1
    # coords[:, 0] is X, coords[:, 1] is Y, coords[:, 2] is Z
    voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    
    return voxel_grid

def visualize_results(voxel_grid, rot_x, rot_y):
    # Visualize the 3 primary orthographic projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Surface Voxelization | Rotated X:{rot_x}°, Y:{rot_y}°", fontsize=16)
    
    # View 1: Projection along Z axis (Top/Bottom view)
    axes[0].imshow(np.max(voxel_grid, axis=2), cmap='gray')
    axes[0].set_title("Projection (Z-Axis)")
    axes[0].axis('off')
    
    # View 2: Projection along Y axis (Side view)
    axes[1].imshow(np.max(voxel_grid, axis=1), cmap='gray')
    axes[1].set_title("Projection (Y-Axis)")
    axes[1].axis('off')
    
    # View 3: Projection along X axis (Front/Back view)
    axes[2].imshow(np.max(voxel_grid, axis=0), cmap='gray')
    axes[2].set_title("Projection (X-Axis)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('surface_voxelization_test.png')
    print("Result saved as 'surface_voxelization_test.png'. Please inspect it.")

if __name__ == "__main__":
    if not os.path.exists(TEST_OFF_FILE):
        print(f"Error: Could not find {TEST_OFF_FILE}. Update the path in the script.")
    else:
        # Test an extreme rotation to prove it doesn't degrade
        test_rot_x = 45.0
        test_rot_y = 135.0
        
        grid = create_surface_voxel_grid(TEST_OFF_FILE, test_rot_x, test_rot_y)
        visualize_results(grid, test_rot_x, test_rot_y)