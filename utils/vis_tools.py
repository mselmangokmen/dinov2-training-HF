import math
import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import torch  
import torch.nn.functional as nnf 
 
 
  
 
def save_feature_maps_plot(feature_maps, output_dir, filename_prefix="register_map", cmap='viridis',figure_title='Register', 
                           patch_size=14, normalize=False, max_cols=4):
    
    # Validate input
    if feature_maps is None or feature_maps.ndim != 4 or feature_maps.shape[0] != 1 or feature_maps.shape[1] == 0:
        print(f"Invalid register maps: {None if feature_maps is None else feature_maps.shape}")
        return
    
    # Get number of registers and determine grid layout
    num_registers = feature_maps.shape[1]
    
    # Use max_cols parameter to control grid width
    cols = min(max_cols, num_registers)
    rows = math.ceil(num_registers / cols)
     
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
     
    axes_flat = np.array(axes).flatten() if hasattr(axes, 'flatten') else [axes]
     
    for i in range(num_registers): 

        map_data = feature_maps[0, i, :, :]
         
        if normalize:
            map_min, map_max = np.min(map_data), np.max(map_data)
            if map_max - map_min > 1e-8:
                map_data = (map_data - map_min) / (map_max - map_min)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = None, None
        
        # Upscale the map
        h, w = map_data.shape
        upscaled = cv2.resize(map_data, (w * patch_size, h * patch_size), interpolation=cv2.INTER_NEAREST)
        
        # Display map
        axes_flat[i].imshow(upscaled, cmap=cmap, vmin=vmin, vmax=vmax)
        axes_flat[i].set_title(f"{figure_title} {i+1}")
        axes_flat[i].axis("off")
    
    # Hide unused subplots
    for j in range(num_registers, len(axes_flat)):
        axes_flat[j].axis("off")
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    norm_suffix = "_normalized" if normalize else ""
    filepath = os.path.join(output_dir, f"{filename_prefix}{norm_suffix}.png")
    plt.tight_layout(pad=1.0)
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    
    return filepath


def visualize_attention_pca_rgb(denorm_img, filtered_pca, patch_size=8, 
                                save_dir="output_images", file_name="pca_rgb_visualization" ): 
    import os
    import torch
    import torch.nn.functional as nnf
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check if we have exactly 3 components for RGB
    n_components = filtered_pca.shape[0]
    if n_components != 3:
        raise ValueError(f"This function requires exactly 3 PCA components for RGB visualization, got {n_components}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Resize PCA components to match patch size
    filtered_pca_resized = nnf.interpolate(
        torch.tensor(filtered_pca).unsqueeze(0),  # Add batch dimension to make it (1, 3, H, W)
        scale_factor=patch_size,  # Scale factor for height and width
        mode="nearest"  # Interpolation mode
    )[0].numpy()
    # Resize original image to match PCA dimensions
    denorm_img = nnf.interpolate(
        denorm_img.unsqueeze(0),
        size=(filtered_pca_resized.shape[-2], filtered_pca_resized.shape[-1]),
    ).squeeze(0)
    
    # Convert image to numpy and transpose for visualization
    img = denorm_img.cpu().numpy().transpose(1, 2, 0)
     
    # Normalize PCA components to [0, 1] for RGB visualization
    pca_rgb = np.zeros((filtered_pca_resized.shape[-2], filtered_pca_resized.shape[-1], 3))
    for i in range(3):
        channel = filtered_pca_resized[i]
        # Normalize each channel to [0, 1]
        channel_min = channel.min()
        channel_max = channel.max()
        if channel_max > channel_min:
            pca_rgb[:, :, i] = (channel - channel_min) / (channel_max - channel_min)
        else:
            pca_rgb[:, :, i] = channel
    
    # Create figure with 2 subplots: Original Image + RGB PCA Map
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original image
    axes[0].imshow((img + 1) / 2 if np.amin(img)<0 else img )
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")
    
    # Plot RGB PCA map
    axes[1].imshow(pca_rgb)
    axes[1].set_title("RGB PCA Map\n(R=PC1, G=PC2, B=PC3)", fontsize=13)
    axes[1].axis("off")
     
    # Save the figure
    fname = f"{file_name}.png"
    fname = os.path.join(save_dir, fname)
    plt.savefig(fname=fname, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
     
    
    return pca_rgb  
 

def visualize_attention_pca(denorm_img, filtered_pca, patch_size=8, mask=None, cmap='inferno',
                          save_dir="output_images", file_name="pca_visualization", sigma=1):
    os.makedirs(save_dir, exist_ok=True)
    n_components = filtered_pca.shape[0]
    
    # Resize PCA components
    filtered_pca_resized = nnf.interpolate(
        torch.tensor(filtered_pca).unsqueeze(0),
        scale_factor=patch_size,
        mode="nearest"
    )[0].numpy()
    
    # Resize image to match PCA dimensions
    denorm_img = nnf.interpolate(
        denorm_img.unsqueeze(0),
        size=(filtered_pca_resized.shape[-2], filtered_pca_resized.shape[-1]),
    ).squeeze(0)
    
    img = denorm_img.cpu().numpy().transpose(1, 2, 0)
    
    # Process mask
    if mask is not None:
        mask = nnf.interpolate(
            mask.unsqueeze(0),
            size=(filtered_pca_resized.shape[-2], filtered_pca_resized.shape[-1]), 
            mode='nearest'
        ).squeeze(0)
        mask = mask.squeeze(0).cpu().numpy()
    else:
        mask = np.zeros((filtered_pca_resized.shape[-2], filtered_pca_resized.shape[-1]))
    
    # Calculate total plots: Original + Original with Mask + PCA Sum + Each PCA Channel
    total_plots = n_components + 3  # Original + Original with Mask + PCA Sum + Each PCA Channel
    if n_components < 2:
        total_plots = 3  # Original + Original with Mask + PCA Sum
    
    fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 5))
    
    # Plot 1: Original Image (without mask)
    plot_img = (img + 1) / 2 if np.amin(img)<0 else img 
    axes[0].imshow(plot_img, cmap='gray')
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")
    
    # Plot 2: Original Image with Mask Overlay
    img_with_mask = plot_img
    if np.any(mask > 0.5):
        # Create overlay
        alpha = 0.6
        overlay = img_with_mask.copy()
        
        # Get mask pixels
        mask_pixels = mask > 0.5
        
        # If image is grayscale, convert to RGB for colored overlay
        if img_with_mask.shape[-1] == 1:
            overlay = np.repeat(img_with_mask, 3, axis=-1)
        elif img_with_mask.shape[-1] != 3:
            # If more than 3 channels, take first 3
            overlay = overlay[..., :3]
        
        # Apply red overlay to mask areas
        overlay[mask_pixels] = alpha * np.array([1, 0, 0]) + (1 - alpha) * overlay[mask_pixels]
        
        axes[1].imshow(overlay)
        axes[1].set_title("Original Image + Mask", fontsize=13)
    else:
        axes[1].imshow(img_with_mask, cmap='gray')
        axes[1].set_title("Original Image (No Mask)", fontsize=13)
    axes[1].axis("off")
    
    # Plot 3: PCA Sum Map
    axes[2].imshow(np.sum(filtered_pca_resized, axis=0), cmap=cmap)
    axes[2].set_title("PCA Sum Map", fontsize=13)
    axes[2].axis("off")
    
    # Visualize each PCA channel
    if n_components > 1:
        for i in range(n_components):
            axes[i + 3].imshow(filtered_pca_resized[i], cmap=cmap)
            axes[i + 3].set_title(f"PCA Channel {i + 1}", fontsize=13)
            axes[i + 3].axis("off")
    
    # Save the figure
    fname = f"{file_name}.png"
    fname = os.path.join(save_dir, fname)
    plt.savefig(fname=fname, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

  