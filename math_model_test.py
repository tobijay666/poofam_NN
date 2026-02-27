import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm

# Configuration
DATA_DIR = "../modelnet40"
SEARCH_RESOLUTION = 15 # Degrees per step (Lower is more accurate, higher is faster)
OUTPUT_PLOT = "cost_landscape_test.png"

def calculate_print_cost(mesh, rot_x_deg, rot_y_deg, overhang_angle_threshold=45):
    rotated_mesh = mesh.copy()
    rx, ry = np.radians(rot_x_deg), np.radians(rot_y_deg)
    
    rot_matrix_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
    rot_matrix_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
    transform = trimesh.transformations.concatenate_matrices(rot_matrix_y, rot_matrix_x)
    
    rotated_mesh.apply_transform(transform)
    
    z_height = rotated_mesh.extents[2] 
    
    normals = rotated_mesh.face_normals
    face_areas = rotated_mesh.area_faces
    z_normals = normals[:, 2]
    
    threshold_rad = np.radians(overhang_angle_threshold)
    limit_z = -np.cos(threshold_rad) 
    
    overhang_mask = z_normals < limit_z
    overhang_area = np.sum(face_areas[overhang_mask])
    
    return z_height, overhang_area

def find_optimal_orientation(mesh, search_resolution):
    # Normalize mesh scale so optimization landscape is stable
    mesh = mesh.copy()
    mesh.apply_scale(1.0 / mesh.extents.max())
    
    angles_x = np.arange(0, 360, search_resolution)
    angles_y = np.arange(0, 360, search_resolution)
    
    results =[]
    total_area = mesh.area
    max_possible_z = mesh.extents.max() 
    
    # Use tqdm for nested loop progress
    total_iters = len(angles_x) * len(angles_y)
    pbar = tqdm(total=total_iters, desc="Scanning Rotations")
    
    for rx in angles_x:
        for ry in angles_y:
            z_h, ov_a = calculate_print_cost(mesh, rx, ry)
            
            norm_z = z_h / max_possible_z
            norm_ov = ov_a / total_area
            
            # Cost Function (Equal weight)
            alpha, beta = 0.5, 0.5 
            cost = (alpha * norm_z) + (beta * norm_ov)
            
            results.append({
                'rx': rx, 'ry': ry, 
                'z_height': norm_z, 'overhang': norm_ov, 
                'cost': cost
            })
            pbar.update(1)
            
    pbar.close()
    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['cost'].idxmin()]
    
    return best_row, df_results

if __name__ == "__main__":
    print("--- Phase 1.5: Mathematical Ground Truth Testing ---")
    
    search_pattern = os.path.join(DATA_DIR, "**", "*.off")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print("Error: No files found.")
        exit()

    # Pick a random file to test
    test_file = np.random.choice(files)
    print(f"Testing mathematical model on: {os.path.basename(test_file)}")
    
    test_mesh = trimesh.load(test_file, force='mesh')
    best_orientation, optimization_landscape = find_optimal_orientation(test_mesh, SEARCH_RESOLUTION)

    print("\nOptimal Orientation Found:")
    print(f"  Rotation X: {best_orientation['rx']} deg")
    print(f"  Rotation Y: {best_orientation['ry']} deg")
    print(f"  Minimum Cost: {best_orientation['cost']:.4f}")

    # Plotting the landscape
    landscape_matrix = optimization_landscape.pivot(index='rx', columns='ry', values='cost')

    plt.figure(figsize=(10, 8))
    sns.heatmap(landscape_matrix, cmap='viridis')
    plt.title(f"Print Cost Landscape for {os.path.basename(test_file)}\n(Darker is Better / Lower Cost)")
    plt.xlabel("Rotation Y (Degrees)")
    plt.ylabel("Rotation X (Degrees)")

    # Highlight the absolute minimum
    best_rx = best_orientation['rx']
    best_ry = best_orientation['ry']

    x_idx = list(landscape_matrix.index).index(best_rx)
    y_idx = list(landscape_matrix.columns).index(best_ry)

    plt.scatter(y_idx + 0.5, x_idx + 0.5, marker='*', color='red', s=200, label='Global Minimum')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"\nCost landscape heatmap saved to {OUTPUT_PLOT}")