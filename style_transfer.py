"""
Neural Style Transfer implementation.
This module provides the core functionality for neural style transfer using PyTorch.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    
    # Resize if needed
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Add batch dimension
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Function to convert tensor to image
def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

# Feature Extractor class using VGG19
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.features = nn.ModuleList(vgg)
        self.layer_indices = {'relu1_1': 1, 'relu2_1': 6, 'relu3_1': 11, 
                              'relu4_1': 20, 'relu5_1': 29}
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            for name, index in self.layer_indices.items():
                if idx == index:
                    features[name] = x
        return features

# Compute Gram Matrix for style representation
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(c * h * w)

# Content Loss
def content_loss(target_features, content_features, layer_indices):
    loss = 0
    for layer in layer_indices:
        loss += F.mse_loss(target_features[layer], content_features[layer])
    return loss

# Style Loss
def style_loss(target_features, style_features, layer_indices):
    loss = 0
    for layer in layer_indices:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        loss += F.mse_loss(target_gram, style_gram)
    return loss

# Main style transfer function
def neural_style_transfer(content_img, style_img, num_steps=300, 
                         style_weight=1000000, content_weight=1, lr=0.01):
    
    # Extract features
    feature_extractor = VGG19FeatureExtractor()
    content_features = feature_extractor(content_img)
    style_features = feature_extractor(style_img)
    
    # Initialize generated image with content image
    generated_img = content_img.clone().requires_grad_(True)
    
    # Set up optimizer
    optimizer = optim.Adam([generated_img], lr=lr)
    
    # Content and style layers
    content_layers = ['relu4_1']
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    
    # Training loop
    progress_bar = tqdm(range(num_steps))
    for step in progress_bar:
        # Get features of generated image
        generated_features = feature_extractor(generated_img)
        
        # Calculate content loss
        c_loss = content_loss(generated_features, content_features, content_layers)
        
        # Calculate style loss
        s_loss = style_loss(generated_features, style_features, style_layers)
        
        # Total loss
        total_loss = content_weight * c_loss + style_weight * s_loss
        
        # Update progress bar
        progress_bar.set_description(f"Loss: {total_loss.item():.4f}")
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Clip values to valid image range
        with torch.no_grad():
            generated_img.clamp_(0, 1)
    
    return generated_img

# Plot images for visualization
def plot_images(content_img, style_img, generated_img):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(tensor_to_image(content_img))
    ax1.set_title("Content Image")
    ax1.axis("off")
    
    ax2.imshow(tensor_to_image(style_img))
    ax2.set_title("Style Image")
    ax2.axis("off")
    
    ax3.imshow(tensor_to_image(generated_img))
    ax3.set_title("Generated Image")
    ax3.axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer with PyTorch")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output", type=str, required=True, help="Path for output image")
    parser.add_argument("--iterations", type=int, default=300, help="Number of iterations to run")
    parser.add_argument("--style-weight", type=float, default=1000000, help="Weight for style loss")
    parser.add_argument("--content-weight", type=float, default=1, help="Weight for content loss")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--plot", action="store_true", help="Plot the images during transfer")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load images
    content_img = load_image(args.content)
    style_img = load_image(args.style)
    
    # Match style image dimensions to content image
    style_img = F.interpolate(style_img, size=content_img.shape[2:])
    
    print(f"Running style transfer on {device}...")
    print(f"Content image: {args.content}")
    print(f"Style image: {args.style}")
    
    # Perform style transfer
    generated_img = neural_style_transfer(
        content_img, 
        style_img,
        num_steps=args.iterations,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        lr=args.lr
    )
    
    # Convert back to PIL Image and save
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = generated_img.cpu().clone().squeeze(0)
    img = denorm(img).clamp_(0, 1)
    save_image(img, args.output)
    
    print(f"Output image saved to {args.output}")
    
    # Plot images if requested
    if args.plot:
        plot_images(content_img, style_img, generated_img)

if __name__ == "__main__":
    main() 