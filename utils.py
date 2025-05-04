import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

def ensure_dir(directory):
    """Make sure the directory exists, create it if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_sample_images(content_dir, style_dir):
    """Load sample images from directories and return file paths"""
    content_images = [os.path.join(content_dir, f) for f in os.listdir(content_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    style_images = [os.path.join(style_dir, f) for f in os.listdir(style_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    return content_images, style_images

def display_images(content_path, style_path, output_path=None):
    """Display images for comparison"""
    fig, axes = plt.subplots(1, 3 if output_path else 2, figsize=(15, 5))
    
    # Display content image
    content_img = Image.open(content_path)
    axes[0].imshow(content_img)
    axes[0].set_title('Content Image')
    axes[0].axis('off')
    
    # Display style image
    style_img = Image.open(style_path)
    axes[1].imshow(style_img)
    axes[1].set_title('Style Image')
    axes[1].axis('off')
    
    # Display output image if provided
    if output_path:
        output_img = Image.open(output_path)
        axes[2].imshow(output_img)
        axes[2].set_title('Generated Image')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_checkpoint(generated_img, iteration, output_dir, filename_prefix):
    """Save checkpoint of the generated image during training"""
    ensure_dir(output_dir)
    filename = f"{filename_prefix}_iter_{iteration}.png"
    output_path = os.path.join(output_dir, filename)
    
    # Save the image
    img = generated_img.cpu().clone().squeeze(0)
    # De-normalize the image
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = denorm(img).clamp_(0, 1)
    save_image(img, output_path)
    
    return output_path

def create_gif(image_paths, output_path, duration=100):
    """Create a GIF from a series of images to show the style transfer process"""
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    return output_path

def initialize_image(content_img, init_method='content'):
    """Initialize the generated image based on the given method"""
    if init_method == 'content':
        # Initialize with content image
        return content_img.clone().requires_grad_(True)
    elif init_method == 'random':
        # Initialize with random noise
        b, c, h, w = content_img.shape
        random_img = torch.randn(b, c, h, w, device=content_img.device)
        # Scale to [0, 1]
        random_img = (random_img - random_img.min()) / (random_img.max() - random_img.min())
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=content_img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=content_img.device).view(1, 3, 1, 1)
        random_img = (random_img - mean) / std
        return random_img.requires_grad_(True)
    elif init_method == 'mixed':
        # Mix content image with random noise
        b, c, h, w = content_img.shape
        random_img = torch.randn(b, c, h, w, device=content_img.device) * 0.1
        mixed_img = content_img.clone() + random_img
        # Clamp to valid range after normalization
        return mixed_img.requires_grad_(True)
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")

def total_variation_loss(img):
    """Calculate total variation loss to enforce spatial smoothness"""
    # Compute differences along rows and columns
    row_diff = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).sum()
    col_diff = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).sum()
    
    return row_diff + col_diff

def gram_matrix(tensor):
    """
    Calculate the Gram Matrix of a given tensor
    The gram matrix is calculated by multiplying the flattened features with their transpose
    
    Args:
        tensor (torch.Tensor): Feature representation tensor
        
    Returns:
        torch.Tensor: Gram matrix
    """
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t) / (c * h * w)
    return gram 