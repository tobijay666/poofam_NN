import trimesh
import numpy as np
import pandas as pd
import os
import glob
import math
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "../modelnet40"
OUTPUT_VOXELS_DIR = "MN40_Surface_Voxels"
OUTPUT_CSV = "MN40_Vector_Labels.csv"

# Hyperparameters
SEARCH_RESOLUTION = 15     # Degrees step for finding optimal angle (15 is a good balance of speed/accuracy)
AUGMENTATIONS_PER_MESH = 2 # How many random 3D tumbles to create per object
GRID_SIZE = 64
PADDING = 2
NUM_SAMPLES = 100000       # Number of points to sample on surface

os.makedirs(OUTPUT_VOXELS_DIR, exist_ok=True)

# --- MATHEMATICAL MODEL ---
def calculate_print_cost(mesh, rot_matrix, overhang_angle_threshold=45):
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rot_matrix)
    
    z_height = rotated_mesh.extents[2] 
    
    normals = rotated_mesh.face_normals
    face_areas = rotated_mesh.area_faces
    z_normals = normals[:, 2]
    
    threshold_rad = np.radians(overhang_angle_threshold)
    limit_z = -np.cos(threshold_rad) 
    
    overhang_mask = z_normals < limit_z
    overhang_area = np.sum(face_areas[overhang_mask])
    
    return z_height, overhang_area

def find_optimal_gravity_vector(mesh):
    """
    Finds the optimal rotation, and returns the 'Down' vector 
    relative to the object's original unrotated state.
    """
    mesh_normalized = mesh.copy()
    mesh_normalized.apply_scale(1.0 / mesh_normalized.extents.max())
    
    angles_x = np.arange(0, 360, SEARCH_RESOLUTION)
    angles_y = np.arange(0, 360, SEARCH_RESOLUTION)
    
    total_area = mesh_normalized.area
    max_possible_z = mesh_normalized.extents.max() 
    
    best_cost = float('inf')
    best_rot_matrix = np.eye(4)
    
    for rx in angles_x:
        for ry in angles_y:
            rx_rad, ry_rad = math.radians(rx), math.radians(ry)
            mat_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
            mat_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
            # Apply X then Y
            rot_matrix = trimesh.transformations.concatenate_matrices(mat_y, mat_x)
            
            z_h, ov_a = calculate_print_cost(mesh_normalized, rot_matrix)
            
            # Cost function
            cost = 0.5 * (z_h / max_possible_z) + 0.5 * (ov_a / total_area)
            
            if cost < best_cost:
                best_cost = cost
                best_rot_matrix = rot_matrix
                
    # The optimal rotation R aligns the object to the print bed.
    # The print bed's "Down" is [0, 0, -1].
    # To find what local vector corresponds to "Down", we apply the INVERSE of R.
    R_inv = np.linalg.inv(best_rot_matrix)
    down_vector_global = np.array([0, 0, -1, 0]) # 0 for direction vector
    local_down_vector = np.dot(R_inv, down_vector_global)[:3]
    
    # Ensure it's a unit vector
    local_down_vector = local_down_vector / np.linalg.norm(local_down_vector)
    
    return local_down_vector

# --- VOXELIZATION ---
def surface_voxelize(mesh):
    # Center and scale to fit in grid (with padding)
    mesh = mesh.copy()
    bounding_box_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-bounding_box_center)
    
    max_extent = mesh.extents.max()
    scale_factor = (GRID_SIZE - (PADDING * 2)) / max_extent
    mesh.apply_scale(scale_factor)
    
    # Sample points
    points, _ = trimesh.sample.sample_surface(mesh, NUM_SAMPLES)
    points_shifted = points + (GRID_SIZE / 2)
    coords = np.round(points_shifted).astype(int)
    coords = np.clip(coords, 0, GRID_SIZE - 1)
    
    voxel_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel_grid

# --- PIPELINE ---
def generate_dataset():
    print("--- Phase 2: Generating Voxel & Vector Dataset ---")
    
    # Load existing CSV to support pausing/resuming
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        processed_originals = set(df_existing['original_file'].unique())
        records = df_existing.to_dict('records')
        print(f"Resuming... Found {len(processed_originals)} already processed meshes.")
    else:
        processed_originals = set()
        records =[]
        
    search_pattern = os.path.join(DATA_DIR, "**", "*.off")
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} total .off files in ModelNet40.")
    
    for f in tqdm(files, desc="Processing Meshes"):
        file_base = os.path.basename(f)
        
        # Skip if already done
        if file_base in processed_originals:
            continue
            
        try:
            mesh = trimesh.load(f, force='mesh')
            
            # Sanity Checks (Skip 2D planes and corrupted files)
            if len(mesh.faces) < 20 or mesh.area == 0: continue
            if mesh.extents.max() / (mesh.extents.min() + 1e-6) > 50: continue
            
            # Find the true internal "Down" vector
            optimal_local_v = find_optimal_gravity_vector(mesh)
            
            # Generate augmentations
            for aug_idx in range(AUGMENTATIONS_PER_MESH):
                # Random, uniform spherical 3D rotation (Solves all distribution biases)
                random_rot = trimesh.transformations.random_rotation_matrix()
                
                # Apply rotation to mesh
                aug_mesh = mesh.copy()
                aug_mesh.apply_transform(random_rot)
                
                # Generate Voxel
                voxel_grid = surface_voxelize(aug_mesh)
                
                # Transform the Target Vector
                # If the mesh rotates, the physical "Down" feature rotates with it
                target_v = np.dot(random_rot, np.append(optimal_local_v, 0))[:3]
                target_v = target_v / np.linalg.norm(target_v) # Keep as unit vector
                
                # Save
                voxel_id = f"{file_base.replace('.off', '')}_aug{aug_idx}"
                np.save(os.path.join(OUTPUT_VOXELS_DIR, voxel_id + '.npy'), voxel_grid)
                
                records.append({
                    'voxel_id': voxel_id,
                    'original_file': file_base,
                    'v_x': target_v[0],
                    'v_y': target_v[1],
                    'v_z': target_v[2]
                })
                
            # Save CSV frequently so you don't lose data if you stop it
            if len(records) % 100 == 0:
                pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
                
        except Exception as e:
            # Silently skip totally broken meshes
            pass

    # Final Save
    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
    print(f"Generation Complete. Total Voxel Grids: {len(records)}")

if __name__ == "__main__":
    generate_dataset()