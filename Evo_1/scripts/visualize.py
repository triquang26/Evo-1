import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def visualize_standalone_heatmap(pt_path, output_path="attention_heatmap.png"):
    # Added weights_only=True to silence the PyTorch warning
    try:
        tensor_data = torch.load(pt_path, map_location='cpu', weights_only=True)
    except:
        tensor_data = torch.load(pt_path, map_location='cpu')
        
    tensor_data = tensor_data.squeeze()
    
    while len(tensor_data.shape) > 2:
        tensor_data = tensor_data.mean(dim=0)
        
    if len(tensor_data.shape) == 2 and tensor_data.shape[0] == tensor_data.shape[1]:
        tensor_data = tensor_data.mean(dim=0)
        
    num_tokens = tensor_data.shape[0]
    grid_size = int(np.sqrt(num_tokens))
    grid_h = grid_w = grid_size
        
    # FIX: Added .float() to convert BFloat16/Float16 to standard Float32 before calling .numpy()
    heatmap_2d = tensor_data.reshape(grid_h, grid_w).detach().float().numpy()
    
    heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)
    
    heatmap_resized = cv2.resize(heatmap_2d, (448, 448), interpolation=cv2.INTER_CUBIC)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(heatmap_resized, ax=ax, 
                cmap='jet', cbar=True, xticklabels=False, yticklabels=False)
    
    plt.title("Attention Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    PT_PATH = "/mnt/data/sftp/data/quangpt3/Evo-1/Evo_1/scripts/outputs/attention_map_step.pt"
    visualize_standalone_heatmap(PT_PATH)
