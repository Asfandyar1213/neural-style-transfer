import argparse
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import TransformerNet
from utils import ensure_dir

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stylize(args):
    # Load transformer model
    print(f"Loading model from {args.model}")
    with torch.no_grad():
        style_model = TransformerNet().to(device)
        state_dict = torch.load(args.model, map_location=device)
        # For loading a checkpoint vs a final model
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            style_model.load_state_dict(state_dict['model_state_dict'])
        else:
            style_model.load_state_dict(state_dict)
        style_model.eval()
        
        # Create output directory
        ensure_dir(os.path.dirname(args.output))
        
        # Prepare content image
        content_image = Image.open(args.content).convert('RGB')
        content_transform = transforms.Compose([
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        content_image = content_transform(content_image).unsqueeze(0).to(device)
        
        # Generate stylized image
        output = style_model(content_image)
        
        # Save the output image
        output_image = output.cpu()[0]
        output_image = output_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                      torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        output_image = output_image.clamp(0, 1)
        save_image(output_image, args.output)
        
        print(f"Stylized image saved to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Apply style transfer using a trained model")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--output", type=str, required=True, help="Path for output image")
    parser.add_argument("--size", type=int, default=512, help="Size of the output image")
    
    args = parser.parse_args()
    stylize(args)

if __name__ == "__main__":
    main() 