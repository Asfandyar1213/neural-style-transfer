import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

from models import TransformerNet, VGG19FeatureExtractor
from utils import save_checkpoint, ensure_dir, gram_matrix

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    # Create output directories
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.checkpoint_dir + '/samples')
    
    # Load style image
    style_transform = transforms.Compose([
        transforms.Resize(args.style_size),
        transforms.CenterCrop(args.style_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    style_image = style_transform(Image.open(args.style_image))
    style_image = style_image.repeat(1, 1, 1, 1).to(device)
    
    # Load dataset for content images
    content_transform = transforms.Compose([
        transforms.Resize(args.content_size),
        transforms.CenterCrop(args.content_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(args.dataset, content_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create models
    transformer = TransformerNet().to(device)
    vgg = VGG19FeatureExtractor(device=device)
    
    # Set up optimizer
    optimizer = optim.Adam(transformer.parameters(), args.lr)
    
    # Extract style features
    style_features = vgg(style_image)
    style_gram = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Layers for content and style
    content_layers = ['relu4_1']
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    
    # Training loop
    for epoch in range(args.epochs):
        transformer.train()
        epoch_loss = 0
        count = 0
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        for batch_id, (x, _) in enumerate(tqdm(train_loader)):
            count += 1
            optimizer.zero_grad()
            
            x = x.to(device)
            y = transformer(x)
            
            # Get features from VGG
            y_features = vgg(y)
            x_features = vgg(x)
            
            # Calculate content loss
            content_loss = 0
            for layer in content_layers:
                content_loss += nn.MSELoss()(y_features[layer], x_features[layer])
            
            # Calculate style loss
            style_loss = 0
            for layer in style_layers:
                y_gram = gram_matrix(y_features[layer])
                style_loss += nn.MSELoss()(y_gram, style_gram[layer].expand_as(y_gram))
            
            # Calculate total variation loss for smoothing
            tv_loss = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
                     torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
            
            # Total loss
            loss = args.content_weight * content_loss + \
                   args.style_weight * style_loss + \
                   args.tv_weight * tv_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Save sample images periodically
            if (batch_id + 1) % args.sample_interval == 0:
                transformer.eval()
                with torch.no_grad():
                    output = transformer(x)
                    output_image = output[0].cpu().clone()
                    output_image = output_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                                  torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    output_image = output_image.clamp(0, 1)
                    
                    sample_path = os.path.join(args.checkpoint_dir, 'samples', 
                                              f"epoch_{epoch+1}_batch_{batch_id+1}.jpg")
                    vutils.save_image(output_image, sample_path)
                transformer.train()
        
        # Save checkpoint
        avg_loss = epoch_loss / count
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save(transformer.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Fast Neural Style Transfer")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--style-image", type=str, required=True, help="Path to style image")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--content-size", type=int, default=256, help="Size of content images")
    parser.add_argument("--style-size", type=int, default=512, help="Size of style image")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--content-weight", type=float, default=1.0, help="Weight for content loss")
    parser.add_argument("--style-weight", type=float, default=5.0, help="Weight for style loss")
    parser.add_argument("--tv-weight", type=float, default=1e-6, help="Weight for total variation loss")
    parser.add_argument("--sample-interval", type=int, default=500, help="Interval for saving sample images")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    try:
        from PIL import Image
        import torchvision.utils as vutils
    except ImportError:
        print("Error: PIL or torchvision.utils not installed")
        sys.exit(1)
    
    main() 