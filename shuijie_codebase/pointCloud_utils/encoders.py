"""
This is where we perform pointcloud encoding.

What we want is to perform MLP to each part of the point
"""
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .PC_utils import run_vggt_model, load_vggt_model, filter_points_by_confidence


import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

device='cuda:0'


class SimplePointCloudEmbedder(nn.Module):
    """
    Encodes point clouds into patch-based embeddings for VLA models.
    
    Divides a point cloud into patches of fixed size, applies a shared MLP to each point,
    and aggregates features within each patch via mean pooling. The resulting patch-level
    features are then projected to the target embedding dimension.
    
    This approach allows the model to process variable-length point clouds by treating
    them as sequences of patch embeddings, similar to how vision transformers process
    image patches. Supports optional masking to handle invalid/missing points.
    
    Args:
        points_per_patch (int): Number of points in each patch. Default: 1024
        hidden_dim (int): Dimension of MLP hidden layers. Default: 512
        embed_dim (int): Final embedding dimension per patch. Default: 2048
        dropout (float): Dropout probability in MLP layers. Default: 0.1
    """
    
    def __init__(self, points_per_patch=1024, hidden_dim=512, embed_dim=2048, dropout=0.1):
        super().__init__()
        self.points_per_patch = points_per_patch
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.proj = nn.Linear(hidden_dim, embed_dim)

        self.to(device)
        
    def forward(self, pointcloud, mask=None):
        """
        Args:
            pointcloud: [B, N, 3]
            mask: [B, N] optional mask
        
        Returns:
            embeddings: [B, num_patches, embed_dim]
            patch_mask: [B, num_patches]
        """
        # Convert to tensor if numpy array
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud)

        if len(pointcloud.shape) == 2:
            pointcloud = pointcloud.unsqueeze(0).float()

        pointcloud = pointcloud.to(device)

        # print(f"input pointcloud shape: {pointcloud.shape}")


        B, N, _ = pointcloud.shape
        num_patches = N // self.points_per_patch
        points_to_use = num_patches * self.points_per_patch

        # print(f"num_patches: {num_patches}, points_to_use: {points_to_use}")
        
        pointcloud = pointcloud[:, :points_to_use, :]
        pointcloud = pointcloud.reshape(B, num_patches, self.points_per_patch, 3)

        # print(f"new pointcloud shape: {pointcloud.shape}")

        pointcloud_flat = pointcloud.reshape(B * num_patches, self.points_per_patch, 3)
        
        # Apply MLP
        point_features = self.mlp(pointcloud_flat)

        # print(f"point_features shape: {point_features.shape}")

        # [B*num_patches, 1024, hidden_dim]
        
        # Mean pooling (instead of max)
        if mask is not None:
            mask_reshaped = mask[:, :points_to_use].reshape(
                B * num_patches, self.points_per_patch, 1
            ).float()
            # Masked mean: sum(features * mask) / sum(mask)
            patch_features = (point_features * mask_reshaped).sum(dim=1) / (mask_reshaped.sum(dim=1) + 1e-8)
        else:
            patch_features = point_features.mean(dim=1)
        # [B*num_patches, hidden_dim]
        
        patch_features = patch_features.reshape(B, num_patches, self.hidden_dim)
        embeddings = self.proj(patch_features)
        
        # Generate patch masks
        if mask is not None:
            mask_patches = mask[:, :points_to_use].reshape(B, num_patches, self.points_per_patch)
            patch_mask = mask_patches.sum(dim=-1) > (self.points_per_patch * 0.5)
        else:
            patch_mask = torch.ones(B, num_patches, dtype=torch.bool, device=pointcloud.device)
        
        return embeddings, patch_mask
    




if __name__ == "__main__":

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

    vggt_model = load_vggt_model()
    encoder = SimplePointCloudEmbedder()

    # in actual pipeline this will just be an array of numpy images from the sensor
    example_img_dir = "./example_input_imgs"
    input_images_list = [cv2.imread(os.path.join(example_img_dir, img_filename))
                         for img_filename in os.listdir(example_img_dir)]

    # get predictions
    predictions = run_vggt_model(vggt_model, input_images_list)

    filtered_pointCloud_vertices, filtered_pointCloud_rgb= filter_points_by_confidence(predictions, conf_thres=0.9)

    embeddings, patch_mask = encoder.forward(filtered_pointCloud_vertices)


    print(f"embeddings shape: {embeddings.shape}")
