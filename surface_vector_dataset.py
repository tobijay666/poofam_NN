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
OUTPUT_VOXELS_DIR = "MN40_Surface_Voxels"
OUTPUT_CSV = "MN40_Vector_Labels.csv"

# Hyperparameters
SEARCH_RESOLUTION = 15     
AUGMENTATIONS_PER_MESH = 2 
GRID_SIZE = 64
PADDING = 2
NUM_SAMPLES = 100000       

# --- MATHEMATICAL MODEL (TOP LEVEL FOR PICKLING) ---
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
    mesh_normalized = mesh.copy()
    max_extent = mesh_normalized.extents.max()
    if max_extent == 0:
        max_extent = 1.0
    mesh_normalized.apply_scale(1.0 / max_extent)
    
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
            rot_matrix = trimesh.transformations.concatenate_matrices(mat_y, mat_x)
            
            z_h, ov_a = calculate_print_cost(mesh_normalized, rot_matrix)
            
            cost = 0.5 * (z_h / max_possible_z) + 0.5 * (ov_a / (total_area + 1e-6))
            
            if cost < best_cost:
                best_cost = cost
                best_rot_matrix = rot_matrix
                
    R_inv = np.linalg.inv(best_rot_matrix)
    down_vector_global = np.array([0, 0, -1, 0])
    local_down_vector = np.dot(R_inv, down_vector_global)[:3]
    
    norm = np.linalg.norm(local_down_vector)
    if norm > 0:
        local_down_vector = local_down_vector / norm
    
    return local_down_vector

def surface_voxelize(mesh):
    mesh = mesh.copy()
    bounding_box_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-bounding_box_center)
    
    max_extent = mesh.extents.max()
    if max_extent == 0:
        max_extent = 1.0
    scale_factor = (GRID_SIZE - (PADDING * 2)) / max_extent
    mesh.apply_scale(scale_factor)
    
    points, _ = trimesh.sample.sample_surface(mesh, NUM_SAMPLES)
    points_shifted = points + (GRID_SIZE / 2)
    coords = np.round(points_shifted).astype(int)
    coords = np.clip(coords, 0, GRID_SIZE - 1)
    
    voxel_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel_grid

# --- WORKER FUNCTION ---
def process_single_mesh(file_path):
    """
    This function runs in a separate process.
    Returns a list of dictionary records to be saved by the main process.
    """
    file_base = os.path.basename(file_path)
    records =[]
    
    try:
        mesh = trimesh.load(file_path, force='mesh')
        
        if len(mesh.faces) < 20 or mesh.area == 0: 
            return records
        if mesh.extents.max() / (mesh.extents.min() + 1e-6) > 50: 
            return records
        
        optimal_local_v = find_optimal_gravity_vector(mesh)
        
        for aug_idx in range(AUGMENTATIONS_PER_MESH):
            random_rot = trimesh.transformations.random_rotation_matrix()
            
            aug_mesh = mesh.copy()
            aug_mesh.apply_transform(random_rot)
            
            voxel_grid = surface_voxelize(aug_mesh)
            
            target_v = np.dot(random_rot, np.append(optimal_local_v, 0))[:3]
            norm = np.linalg.norm(target_v)
            if norm > 0:
                target_v = target_v / norm
            
            voxel_id = f"{file_base.replace('.off', '')}_aug{aug_idx}"
            save_path = os.path.join(OUTPUT_VOXELS_DIR, voxel_id + '.npy')
            np.save(save_path, voxel_grid)
            
            records.append({
                'voxel_id': voxel_id,
                'original_file': file_base,
                'v_x': target_v[0],
                'v_y': target_v[1],
                'v_z': target_v[2]
            })
            
    except Exception as e:
        # We silently catch exceptions here to prevent one corrupt file 
        # from crashing the entire multiprocessing pool
        pass
        
    return records

# --- MAIN PIPELINE ---
def generate_dataset_multiprocess():
    print("--- Phase 2: Generating Voxel and Vector Dataset (Multiprocessing) ---")
    os.makedirs(OUTPUT_VOXELS_DIR, exist_ok=True)
    
    # Check for existing data to resume
    header_needed = True
    processed_originals = set()

    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            # Check if 'original_file' column exists to avoid errors on empty/corrupt files
            if 'original_file' in df_existing.columns:
                processed_originals = set(df_existing['original_file'].unique())
                print(f"Resuming... Found {len(processed_originals)} already processed original meshes.")
                header_needed = False
        except pd.errors.EmptyDataError:
            # File exists but is empty, start fresh
            pass

    search_pattern = os.path.join(DATA_DIR, "**", "*.off")
    all_files = glob.glob(search_pattern, recursive=True)
    
    # Filter out files that have already been processed
    files_to_process = [f for f in all_files if os.path.basename(f) not in processed_originals]
    print(f"Found {len(all_files)} total files. {len(files_to_process)} left to process.")
    
    if len(files_to_process) == 0:
        print("All files processed.")
        return

    # Calculate optimal thread count (Leave 2 cores free for OS stability)
    # On some systems cpu_count might be None
    cpu_count = multiprocessing.cpu_count()
    max_workers = max(1, cpu_count - 2) if cpu_count else 4
    
    print(f"Starting processing pool with {max_workers} cores...")

    # We use a batch saving mechanism so memory doesn't overload on huge datasets
    record_buffer = []
    buffer_save_limit = 500 # Save to CSV every 500 processed meshes

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        # Dictionary mapping future to file_path for tracking
        futures = {executor.submit(process_single_mesh, file_path): file_path for file_path in files_to_process}
        
        # Wrap as_completed with tqdm for a progress bar
        for future in tqdm(as_completed(futures), total=len(files_to_process), desc="Processing Meshes"):
            result_records = future.result()
            
            if result_records:
                record_buffer.extend(result_records)
            
            # Save chunk to disk and clear buffer
            if len(record_buffer) >= buffer_save_limit:
                df_chunk = pd.DataFrame(record_buffer)
                
                # Check if file exists to handle header correctly for the first batch
                file_exists = os.path.isfile(OUTPUT_CSV)
                
                # If file doesn't exist, we need a header. If it does, we don't.
                # Override header_needed based on actual file presence to be safe
                write_header = header_needed and not file_exists
                
                df_chunk.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
                
                # After first write, we never need header again
                header_needed = False 
                record_buffer = [] # Reset buffer

    # Save any remaining records in the buffer
    if len(record_buffer) > 0:
        df_chunk = pd.DataFrame(record_buffer)
        file_exists = os.path.isfile(OUTPUT_CSV)
        write_header = header_needed and not file_exists
        df_chunk.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)

    print("Generation Complete.")

if __name__ == "__main__":
    # Required for Windows Multiprocessing
    multiprocessing.freeze_support()
    generate_dataset_multiprocess()