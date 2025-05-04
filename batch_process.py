import argparse
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
import glob

from models import TransformerNet
from utils import ensure_dir

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(content_path, model, output_path, size=512):
    """Process a single image with the style transfer model"""
    try:
        # Load and transform content image
        content_image = Image.open(content_path).convert('RGB')
        content_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        content_image = content_transform(content_image).unsqueeze(0).to(device)
        
        # Generate stylized image
        with torch.no_grad():
            output = model(content_image)
        
        # Save the output image
        output_image = output.cpu()[0]
        output_image = output_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                      torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        output_image = output_image.clamp(0, 1)
        save_image(output_image, output_path)
        
        return True
    except Exception as e:
        print(f"Error processing {content_path}: {e}")
        return False

def batch_process(args):
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
        ensure_dir(args.output_dir)
        
        # Get list of images to process
        if os.path.isdir(args.input_dir):
            image_paths = []
            for ext in ['jpg', 'jpeg', 'png']:
                image_paths.extend(glob.glob(os.path.join(args.input_dir, f'*.{ext}')))
                image_paths.extend(glob.glob(os.path.join(args.input_dir, f'*.{ext.upper()}')))
        else:
            # Single file
            image_paths = [args.input_dir]
        
        print(f"Found {len(image_paths)} images to process")
        if len(image_paths) == 0:
            print("No images found. Make sure your input path is correct.")
            return
        
        # Process each image
        successful = 0
        for image_path in tqdm(image_paths):
            # Create output filename
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output_dir, f"{name}_stylized{ext}")
            
            # Process the image
            if process_image(image_path, style_model, output_path, args.size):
                successful += 1
        
        print(f"Processing complete. {successful}/{len(image_paths)} images processed successfully.")
        print(f"Stylized images saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Batch process images with style transfer")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing images or a single image")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save stylized images")
    parser.add_argument("--size", type=int, default=512, help="Size of the output images")
    
    args = parser.parse_args()
    batch_process(args)

if __name__ == "__main__":
    main() 