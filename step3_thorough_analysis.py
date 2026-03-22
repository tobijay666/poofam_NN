import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import trimesh
import os
import glob

# --- CONFIGURATION ---
CSV_PATH = "MN40_Best_Orientations.csv"
DATA_DIR = "../ModelNet40"
OUTPUT_FOLDER = "Analysis_Results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def run_analysis():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Analyzing {len(df)} samples...")

    # 1. DISTRIBUTION ANALYSIS (X vs Y)
    plt.figure(figsize=(12, 10))
    # Using a hexbin plot to see density without point overlap
    hb = plt.hexbin(df['angle_x'], df['angle_y'], gridsize=20, cmap='YlGnBu', mincnt=1)
    plt.colorbar(hb, label='Count of Models')
    plt.title("Heatmap of Optimal Orientations (X vs Y)")
    plt.xlabel("Optimal Angle X (Degrees)")
    plt.ylabel("Optimal Angle Y (Degrees)")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "orientation_heatmap.png"))
    plt.close()

    # 2. BIAS QUANTIFICATION
    # Define (0,0) as the "Identity" orientation
    identity_mask = (df['angle_x'] == 0) & (df['angle_y'] == 0)
    identity_count = identity_mask.sum()
    identity_percent = (identity_count / len(df)) * 100

    # 3. CLASS-WISE BIAS
    class_bias = df[identity_mask].groupby('class').size() / df.groupby('class').size()
    class_bias = class_bias.fillna(0).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    class_bias.plot(kind='bar', color='salmon')
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% Bias Threshold')
    plt.title("Percentage of Models per Class where Optimal is (0,0)")
    plt.ylabel("Bias Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "class_bias_analysis.png"))
    plt.close()

    # 4. VISUAL AUDIT: airplane_0636
    visualize_airplane_audit(df)

    # 5. CRITICAL EVALUATION REPORT
    write_report(len(df), identity_count, identity_percent, class_bias)

def visualize_airplane_audit(df):
    target_id = "airplane_0636"
    # Find the file path
    search_pattern = os.path.join(DATA_DIR, "**", f"{target_id}.off")
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        print(f"Warning: Could not find {target_id}.off for visualization.")
        return

    mesh_path = found_files[0]
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Get the best angles from CSV
    row = df[df['voxel_id'] == target_id]
    if row.empty:
        print(f"Warning: {target_id} not found in CSV.")
        return
    
    best_x = np.radians(row.iloc[0]['angle_x'])
    best_y = np.radians(row.iloc[0]['angle_y'])

    # Setup visualization
    fig = plt.figure(figsize=(15, 7))
    
    # Subplot 1: Original (0,0)
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh_on_ax(mesh, ax1, "Original Orientation (0,0)")

    # Subplot 2: Optimal
    ax2 = fig.add_subplot(122, projection='3d')
    rot_x = trimesh.transformations.rotation_matrix(best_x, [1, 0, 0])
    rot_y = trimesh.transformations.rotation_matrix(best_y, [0, 1, 0])
    mesh_opt = mesh.copy()
    mesh_opt.apply_transform(trimesh.transformations.concatenate_matrices(rot_y, rot_x))
    plot_mesh_on_ax(mesh_opt, ax2, f"Optimal Orientation ({row.iloc[0]['angle_x']}°, {row.iloc[0]['angle_y']}°)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "airplane_0636_comparison.png"))
    plt.close()

def plot_mesh_on_ax(mesh, ax, title):
    # Simple point cloud visualization for the audit
    v = mesh.vertices
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=0.1, alpha=0.5)
    ax.set_title(title)
    # Force equal aspect ratio
    max_range = np.array([v[:,0].max()-v[:,0].min(), v[:,1].max()-v[:,1].min(), v[:,2].max()-v[:,2].min()]).max() / 2.0
    mid_x = (v[:,0].max()+v[:,0].min()) * 0.5
    mid_y = (v[:,1].max()+v[:,1].min()) * 0.5
    mid_z = (v[:,2].max()+v[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def write_report(total, id_count, id_pct, class_bias):
    report_path = os.path.join(OUTPUT_FOLDER, "critical_evaluation.txt")
    with open(report_path, "w") as f:
        f.write("CRITICAL DATASET EVALUATION\n")
        f.write("==========================\n\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Samples with (0,0) as Optimal: {id_count} ({id_pct:.2f}%)\n\n")
        
        f.write("BIAS ANALYSIS:\n")
        if id_pct > 30:
            f.write("CRITICAL WARNING: High (0,0) bias detected. The model will likely overfit to the identity rotation.\n")
        else:
            f.write("STATUS: (0,0) distribution appears manageable.\n")
            
        f.write("\nTOP BIASED CLASSES:\n")
        f.write(class_bias.head(10).to_string())
        
        f.write("\n\nRESEARCHER RECOMMENDATION:\n")
        f.write("1. If bias > 20%, we must implement 'Relative Rotation Training'.\n")
        f.write("2. Instead of predicting the absolute best angle, we should randomly rotate the input mesh\n")
        f.write("   and train the network to predict the 'Fix' rotation (Best_Angle - Current_Angle).\n")
        f.write("3. This effectively creates an infinite dataset and destroys the (0,0) bias.\n")

if __name__ == "__main__":
    run_analysis()