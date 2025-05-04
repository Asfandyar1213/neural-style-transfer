import argparse
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import VGG19FeatureExtractor
from utils import ensure_dir, gram_matrix, initialize_image, save_checkpoint, display_images, create_gif

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
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
    optimizer = torch.optim.Adam([generated_img], lr=lr)
    
    # Content and style layers
    content_layers = ['relu4_1']
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    
    # Training loop
    progress_bar = tqdm(range(num_steps))
    for step in progress_bar:
        # Get features of generated image
        generated_features = feature_extractor(generated_img)
        
        # Calculate content loss
        c_loss = 0
        for layer in content_layers:
            c_loss += torch.nn.functional.mse_loss(generated_features[layer], content_features[layer])
        
        # Calculate style loss
        s_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(generated_features[layer])
            style_gram = gram_matrix(style_features[layer])
            s_loss += torch.nn.functional.mse_loss(target_gram, style_gram)
        
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

def demo_style_transfer(content_path, style_path, output_dir="output", iterations=300, 
                       style_weight=1000000, content_weight=1, lr=0.01, 
                       save_interval=50, create_animation=True):
    """
    Demonstrate neural style transfer and save progress images
    """
    # Create output directories
    ensure_dir(output_dir)
    temp_dir = os.path.join(output_dir, "progress")
    ensure_dir(temp_dir)
    
    # Load images
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    
    # Display original images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(content_img)
    ax1.set_title("Content Image")
    ax1.axis('off')
    ax2.imshow(style_img)
    ax2.set_title("Style Image")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original_images.png"))
    plt.close()
    
    # Load and prepare images
    content_tensor = load_image(content_path)
    style_tensor = load_image(style_path)
    
    # Match style image dimensions to content image
    style_tensor = torch.nn.functional.interpolate(style_tensor, size=content_tensor.shape[2:])
    
    print(f"Running style transfer on {device}...")
    print(f"Content image: {content_path}")
    print(f"Style image: {style_path}")
    print(f"Output directory: {output_dir}")
    
    # Set up tracking variables for animation
    checkpoint_paths = []
    start_time = time.time()
    
    # Create feature extractor
    feature_extractor = VGG19FeatureExtractor().to(device)
    
    # Extract content and style features
    content_features = feature_extractor(content_tensor)
    style_features = feature_extractor(style_tensor)
    
    # Calculate style gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize generated image with content image
    generated_img = initialize_image(content_tensor, init_method='content')
    
    # Set up optimizer
    optimizer = torch.optim.Adam([generated_img], lr=lr)
    
    # Content and style layers
    content_layers = ['relu4_1']
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    
    # Training loop with progress saving
    progress_bar = tqdm(range(iterations))
    for step in progress_bar:
        # Get features of generated image
        generated_features = feature_extractor(generated_img)
        
        # Calculate content loss
        content_loss = 0
        for layer in content_layers:
            content_loss += torch.nn.functional.mse_loss(
                generated_features[layer], content_features[layer])
        
        # Calculate style loss
        style_loss = 0
        for layer in style_layers:
            gen_gram = gram_matrix(generated_features[layer])
            style_gram = style_grams[layer]
            style_loss += torch.nn.functional.mse_loss(gen_gram, style_gram.expand_as(gen_gram))
        
        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Update progress bar
        progress_bar.set_description(f"Loss: {total_loss.item():.4f}")
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Clip values to valid image range
        with torch.no_grad():
            generated_img.clamp_(0, 1)
        
        # Save checkpoint images for animation
        if (step + 1) % save_interval == 0 or step == 0 or step == iterations - 1:
            checkpoint_path = save_checkpoint(
                generated_img, step + 1, temp_dir, "style_transfer"
            )
            checkpoint_paths.append(checkpoint_path)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Style transfer completed in {elapsed_time:.2f} seconds")
    
    # Save final image
    final_output_path = os.path.join(output_dir, "final_output.png")
    save_checkpoint(generated_img, iterations, output_dir, "final")
    
    # Create GIF animation
    if create_animation and len(checkpoint_paths) > 1:
        gif_path = os.path.join(output_dir, "style_transfer_progress.gif")
        create_gif(checkpoint_paths, gif_path)
        print(f"Animation saved to {gif_path}")
    
    # Display final comparison
    display_images(content_path, style_path, final_output_path)
    
    return final_output_path

def main():
    parser = argparse.ArgumentParser(description="Demo of Neural Style Transfer")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--iterations", type=int, default=300, help="Number of iterations")
    parser.add_argument("--style-weight", type=float, default=1000000, help="Weight for style loss")
    parser.add_argument("--content-weight", type=float, default=1, help="Weight for content loss")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--save-interval", type=int, default=50, help="Interval for saving progress images")
    parser.add_argument("--no-animation", action="store_true", help="Disable GIF animation creation")
    
    args = parser.parse_args()
    
    demo_style_transfer(
        args.content, 
        args.style, 
        args.output_dir,
        args.iterations,
        args.style_weight,
        args.content_weight,
        args.lr,
        args.save_interval,
        not args.no_animation
    )

if __name__ == "__main__":
    main() 