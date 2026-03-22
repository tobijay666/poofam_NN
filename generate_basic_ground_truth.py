import trimesh
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- CONFIGURATION ---
# Removed the leading slash so it looks in your current directory
DATA_DIR = "../modelnet40" 
ALL_RESULTS_CSV = "MN40_All_Orientations.csv"
BEST_RESULTS_CSV = "MN40_Best_Orientations.csv"

# Search Parameters
STEP = 10
ANGLE_RANGE = np.arange(0, 181, STEP) 
OVERHANG_THRESHOLD = 45 

def calculate_metrics(mesh):
    height = mesh.extents[2]
    normals = mesh.face_normals
    areas = mesh.area_faces
    z_components = normals[:, 2]
    limit = -np.cos(np.radians(OVERHANG_THRESHOLD))
    overhang_mask = z_components < limit
    overhang_area = np.sum(areas[overhang_mask])
    return height, overhang_area

def process_mesh(file_path):
    # Standard ModelNet structure: ModelNet40/class/train/file.off
    path_parts = os.path.normpath(file_path).split(os.sep)
    
    # Safety check to ensure we have enough parts in the path
    if len(path_parts) < 3:
        return None
        
    class_name = path_parts[-3]
    split = path_parts[-2]
    file_base = path_parts[-1].replace('.off', '')
    
    try:
        mesh = trimesh.load(file_path, force='mesh')
        if len(mesh.faces) < 20:
            return None
        
        # Normalize scale to 1.0 for fair comparison
        mesh.apply_scale(1.0 / (mesh.extents.max() + 1e-9))
        
        mesh_results = []
        
        for rx in ANGLE_RANGE:
            for ry in ANGLE_RANGE:
                m_copy = mesh.copy()
                rx_rad, ry_rad = np.radians(rx), np.radians(ry)
                
                rot_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
                rot_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
                combined_rot = trimesh.transformations.concatenate_matrices(rot_y, rot_x)
                
                m_copy.apply_transform(combined_rot)
                h, ov = calculate_metrics(m_copy)
                
                mesh_results.append({
                    'voxel_id': file_base,
                    'class': class_name,
                    'split': split,
                    'angle_x': rx,
                    'angle_y': ry,
                    'height': h,
                    'overhang': ov
                })
        
        df_mesh = pd.DataFrame(mesh_results)
        
        # Min-Max Normalization per mesh for stable cost calculation
        h_min, h_max = df_mesh['height'].min(), df_mesh['height'].max()
        ov_min, ov_max = df_mesh['overhang'].min(), df_mesh['overhang'].max()
        
        norm_h = (df_mesh['height'] - h_min) / (h_max - h_min + 1e-9)
        norm_ov = (df_mesh['overhang'] - ov_min) / (ov_max - ov_min + 1e-9)
        
        df_mesh['cost'] = 0.5 * norm_h + 0.5 * norm_ov
        
        best_idx = df_mesh['cost'].idxmin()
        best_row = df_mesh.loc[best_idx].to_dict()
        
        return df_mesh.to_dict('records'), best_row
        
    except Exception as e:
        return None

def run_pipeline():
    # Use os.path.join for cross-platform compatibility
    search_pattern = os.path.join(DATA_DIR, "**", "*.off")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Searching in: {os.path.abspath(DATA_DIR)}")
    print(f"Found {len(files)} files in ModelNet40 structure.")

    if len(files) == 0:
        print("Error: No files found. Ensure the 'ModelNet40' folder is in the same directory as this script.")
        return

    all_best_data = []
    chunk_all_data = []
    
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    if os.path.exists(ALL_RESULTS_CSV): os.remove(ALL_RESULTS_CSV)
    if os.path.exists(BEST_RESULTS_CSV): os.remove(BEST_RESULTS_CSV)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_mesh, f): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Grid Search"):
            result = future.result()
            if result:
                full_mesh_list, best_row = result
                all_best_data.append(best_row)
                chunk_all_data.extend(full_mesh_list)
                
                # Write to disk in chunks of 50 meshes to keep I/O efficient and RAM low
                if len(all_best_data) % 50 == 0:
                    # Save "All" data
                    pd.DataFrame(chunk_all_data).to_csv(
                        ALL_RESULTS_CSV, mode='a', index=False, 
                        header=not os.path.exists(ALL_RESULTS_CSV)
                    )
                    chunk_all_data = [] # Clear chunk
                    
                    # Save "Best" data
                    pd.DataFrame(all_best_data).to_csv(
                        BEST_RESULTS_CSV, mode='a', index=False, 
                        header=not os.path.exists(BEST_RESULTS_CSV)
                    )
                    all_best_data = [] # Clear chunk

    # Final save for remaining data
    if chunk_all_data:
        pd.DataFrame(chunk_all_data).to_csv(ALL_RESULTS_CSV, mode='a', index=False, header=not os.path.exists(ALL_RESULTS_CSV))
    if all_best_data:
        pd.DataFrame(all_best_data).to_csv(BEST_RESULTS_CSV, mode='a', index=False, header=not os.path.exists(BEST_RESULTS_CSV))

    print(f"Process complete. Best results saved to {BEST_RESULTS_CSV}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_pipeline()