import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm

# Configure plotting aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)

# Configuration
DATA_DIR = "../modelnet40"
SAMPLE_SIZE = 500  # Analyze a subset to save time during EDA
OUTPUT_CSV = "dataset_audit_stats.csv"
OUTPUT_PLOT = "dataset_audit_distributions.png"

def audit_meshes(file_list, sample_size):
    audit_data =[]
    
    # Process a representative sample
    sample_files = np.random.choice(file_list, min(sample_size, len(file_list)), replace=False)
    
    for f in tqdm(sample_files, desc="Auditing Meshes"):
        try:
            # force='mesh' prevents returning complex Scene graphs
            mesh = trimesh.load(f, force='mesh')
            
            extents_max = mesh.extents.max()
            extents_min = mesh.extents.min()
            
            audit_data.append({
                'filename': os.path.basename(f),
                'class_name': os.path.basename(os.path.dirname(os.path.dirname(f))),
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'is_watertight': mesh.is_watertight,
                'volume': mesh.volume if mesh.is_watertight else np.nan,
                'bounding_box_max': extents_max,
                'bounding_box_min': extents_min,
                'aspect_ratio': extents_max / (extents_min + 1e-6)
            })
        except Exception as e:
            pass # Silently skip corrupt files during audit
            
    return pd.DataFrame(audit_data)

if __name__ == "__main__":
    print("--- Phase 1: ModelNet40 Dataset Audit ---")
    
    # Recursive globbing for ModelNet40 structure
    search_pattern = os.path.join(DATA_DIR, "**", "*.off")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} .off files.")
    
    if len(files) == 0:
        print("Error: No files found. Check your DATA_DIR path.")
        exit()

    df_audit = audit_meshes(files, SAMPLE_SIZE)
    
    # Save statistics
    df_audit.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAudit statistics saved to {OUTPUT_CSV}")
    print("\nSummary Statistics:")
    print(df_audit.describe())

    # Visualizations
    fig, axes = plt.subplots(1, 3)

    # 1. Vertices
    sns.histplot(df_audit['vertices'], bins=50, ax=axes[0], color='blue')
    axes[0].set_title('Distribution of Vertex Counts')
    axes[0].set_xscale('log') 

    # 2. Aspect Ratio
    sns.histplot(df_audit['aspect_ratio'], bins=50, ax=axes[1], color='orange')
    axes[1].set_title('Aspect Ratio (Max / Min Extent)')

    # 3. Watertightness
    sns.countplot(data=df_audit, x='is_watertight', ax=axes[2], palette='Set2')
    axes[2].set_title('Watertight vs. Non-Watertight')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Distribution plots saved to {OUTPUT_PLOT}")