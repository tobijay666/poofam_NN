import trimesh
import numpy as np
import pandas as pd
import os
import glob
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- CONFIGURATION ---
DATA_DIR = "../modelnet40"
OUTPUT_VOXELS_DIR = "MN40_Physics_Voxels"
OUTPUT_CSV = "MN40_Physics_Vectors.csv"

# Hyperparameters
SEARCH_RESOLUTION = 15     # Check every 15 degrees
AUGMENTATIONS_PER_MESH = 2 
GRID_SIZE = 64
PADDING = 2
NUM_SAMPLES = 100000       

# --- PHYSICS ENGINE (Cost Function) ---
def calculate_print_cost(mesh, rot_matrix, overhang_angle=45):
    # Apply rotation virtually
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rot_matrix)
    
    # Metric 1: Projected Support Area (Shadow of overhangs)
    # Normals pointing down (-Z)
    normals = rotated_mesh.face_normals
    # Face areas
    areas = rotated_mesh.area_faces
    
    # Threshold: Faces angled more than 45 deg from vertical require support
    # Dot product with Down Vector (0,0,-1)
    # If dot > cos(45), it needs support
    # standard Z normal: -1 is down. 
    # We want faces where normal.z < -cos(45) (-0.707)
    
    limit = -math.cos(math.radians(overhang_angle))
    z_component = normals[:, 2]
    
    overhang_mask = z_component < limit
    support_area = np.sum(areas[overhang_mask])
    
    # Metric 2: Z-Height (Print Time)
    z_height = rotated_mesh.extents[2]
    
    return support_area, z_height

def find_optimal_physics_vector(mesh):
    """
    Brute force searches for the rotation that minimizes support material.
    Returns the Vector representing "Down" in the object's local space.
    """
    # Normalize scale for consistent cost calculation
    mesh_norm = mesh.copy()
    mesh_norm.apply_scale(1.0 / mesh_norm.extents.max())
    
    best_cost = float('inf')
    best_rot_matrix = np.eye(4)
    
    # Grid Search
    angles = np.arange(0, 360, SEARCH_RESOLUTION)
    
    # We scan X and Y rotations. Z rotation doesn't change support area.
    for rx in angles:
        for ry in angles:
            # Create Matrix
            rx_rad, ry_rad = math.radians(rx), math.radians(ry)
            mat_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
            mat_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
            rot_matrix = trimesh.transformations.concatenate_matrices(mat_y, mat_x)
            
            sup_area, z_h = calculate_print_cost(mesh_norm, rot_matrix)
            
            # COST FUNCTION
            # We prioritize Support Area (Weight 0.7) over Height (Weight 0.3)
            # This is key for "Printability"
            cost = (sup_area * 0.7) + (z_h * 0.3)
            
            if cost < best_cost:
                best_cost = cost
                best_rot_matrix = rot_matrix
    
    # Calculate the Local Down Vector
    # The optimal rotation R takes the object to a state where [0,0,-1] is down.
    # So the Local Down is R_inverse * [0,0,-1].
    R_inv = np.linalg.inv(best_rot_matrix)
    global_down = np.array([0, 0, -1, 0])
    local_down = np.dot(R_inv, global_down)[:3]
    
    return local_down / np.linalg.norm(local_down)

def surface_voxelize(mesh):
    # Standard surface voxelization
    mesh = mesh.copy()
    bounding_box_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-bounding_box_center)
    max_extent = mesh.extents.max()
    if max_extent == 0: max_extent = 1.0
    scale_factor = (GRID_SIZE - (PADDING * 2)) / max_extent
    mesh.apply_scale(scale_factor)
    
    points, _ = trimesh.sample.sample_surface(mesh, NUM_SAMPLES)
    points_shifted = points + (GRID_SIZE / 2)
    coords = np.round(points_shifted).astype(int)
    coords = np.clip(coords, 0, GRID_SIZE - 1)
    
    voxel_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel_grid

# --- WORKER ---
def process_single_mesh(file_path):
    file_base = os.path.basename(file_path)
    records = []
    try:
        mesh = trimesh.load(file_path, force='mesh')
        # Skip garbage files
        if len(mesh.faces) < 100: return records 
        
        # 1. FIND GROUND TRUTH (Physics Based)
        optimal_local_v = find_optimal_physics_vector(mesh)
        
        # 2. AUGMENT
        for aug_idx in range(AUGMENTATIONS_PER_MESH):
            random_rot = trimesh.transformations.random_rotation_matrix()
            
            aug_mesh = mesh.copy()
            aug_mesh.apply_transform(random_rot)
            
            voxel_grid = surface_voxelize(aug_mesh)
            
            # Rotate the Target Vector to match the mesh's new orientation
            target_v = np.dot(random_rot, np.append(optimal_local_v, 0))[:3]
            target_v = target_v / np.linalg.norm(target_v)
            
            voxel_id = f"{file_base.replace('.off', '')}_phys_{aug_idx}"
            np.save(os.path.join(OUTPUT_VOXELS_DIR, voxel_id + '.npy'), voxel_grid)
            
            records.append({
                'voxel_id': voxel_id,
                'v_x': target_v[0],
                'v_y': target_v[1],
                'v_z': target_v[2]
            })
    except:
        pass
    return records

# --- MAIN ---
def generate_dataset_multiprocess():
    print("--- Generating Physics-Based Dataset ---")
    os.makedirs(OUTPUT_VOXELS_DIR, exist_ok=True)
    
    # (Same multiprocessing boilerplate as before...)
    header_needed = True
    processed_originals = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            processed_originals = set([x.split('_phys_')[0] + '.off' for x in df['voxel_id']])
            header_needed = False
        except: pass

    search_pattern = os.path.join(DATA_DIR, "**", "*.off")
    all_files = glob.glob(search_pattern, recursive=True)
    files_to_process = [f for f in all_files if os.path.basename(f) not in processed_originals]
    
    print(f"Processing {len(files_to_process)} files...")
    
    cpu_count = multiprocessing.cpu_count()
    max_workers = max(1, cpu_count - 2) if cpu_count else 4
    
    record_buffer = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_mesh, f): f for f in files_to_process}
        for future in tqdm(as_completed(futures), total=len(files_to_process)):
            res = future.result()
            if res:
                record_buffer.extend(res)
                if len(record_buffer) >= 500:
                    pd.DataFrame(record_buffer).to_csv(OUTPUT_CSV, mode='a', header=header_needed, index=False)
                    header_needed = False; record_buffer = []
                    
    if record_buffer:
        pd.DataFrame(record_buffer).to_csv(OUTPUT_CSV, mode='a', header=header_needed, index=False)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    generate_dataset_multiprocess()