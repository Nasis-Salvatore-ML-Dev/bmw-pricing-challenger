"""Configuration for visualization settings"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Plotting configuration
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
sns.set_style('whitegrid')
sns.set_palette('husl')

# Create images directory
IMAGES_PATH = Path('reports/images/feature_importance')
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    """Save figure to images directory"""
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, bbox_inches='tight')
    print(f"✅ Figure saved: {path}")
    return path