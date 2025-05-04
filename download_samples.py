import os
import urllib.request
import zipfile
import argparse
from tqdm import tqdm

def ensure_dir(directory):
    """Make sure the directory exists, create it if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_file(url, destination, desc=None):
    """Download a file with progress bar"""
    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
            def report_hook(count, block_size, total_size):
                if total_size == -1:  # If total size unknown
                    t.total = None
                    t.update(block_size)
                else:
                    t.total = total_size
                    t.update(count * block_size - t.n)  # Updates with the increment
            
            urllib.request.urlretrieve(url, destination, reporthook=report_hook)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_and_extract_images(content_dir, style_dir):
    """Download sample content and style images"""
    # Create directories
    ensure_dir(content_dir)
    ensure_dir(style_dir)
    
    # URLs for sample image collections
    content_url = "https://github.com/pytorch/examples/raw/main/fast_neural_style/images/content-images.zip"
    style_url = "https://github.com/pytorch/examples/raw/main/fast_neural_style/images/style-images.zip"
    
    # Download content images
    content_zip = os.path.join(os.path.dirname(content_dir), "content-images.zip")
    print("Downloading content images...")
    if download_file(content_url, content_zip, "Content images"):
        # Extract content images
        with zipfile.ZipFile(content_zip, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    zip_ref.extract(file, os.path.dirname(content_dir))
                    # Move files from extracted directory to content_dir
                    extracted_file = os.path.join(os.path.dirname(content_dir), file)
                    if os.path.exists(extracted_file):
                        filename = os.path.basename(extracted_file)
                        os.rename(extracted_file, os.path.join(content_dir, filename))
        
        # Clean up zip file and extracted directory
        os.remove(content_zip)
        extracted_dir = os.path.join(os.path.dirname(content_dir), "content-images")
        if os.path.exists(extracted_dir):
            try:
                os.rmdir(extracted_dir)
            except:
                pass
        
        print(f"Content images downloaded to {content_dir}")
    else:
        print("Failed to download content images")
    
    # Download style images
    style_zip = os.path.join(os.path.dirname(style_dir), "style-images.zip")
    print("Downloading style images...")
    if download_file(style_url, style_zip, "Style images"):
        # Extract style images
        with zipfile.ZipFile(style_zip, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    zip_ref.extract(file, os.path.dirname(style_dir))
                    # Move files from extracted directory to style_dir
                    extracted_file = os.path.join(os.path.dirname(style_dir), file)
                    if os.path.exists(extracted_file):
                        filename = os.path.basename(extracted_file)
                        os.rename(extracted_file, os.path.join(style_dir, filename))
        
        # Clean up zip file and extracted directory
        os.remove(style_zip)
        extracted_dir = os.path.join(os.path.dirname(style_dir), "style-images")
        if os.path.exists(extracted_dir):
            try:
                os.rmdir(extracted_dir)
            except:
                pass
        
        print(f"Style images downloaded to {style_dir}")
    else:
        print("Failed to download style images")
    
    # Display image counts
    content_count = len([f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    style_count = len([f for f in os.listdir(style_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Downloaded {content_count} content images and {style_count} style images")

def main():
    parser = argparse.ArgumentParser(description="Download sample content and style images")
    parser.add_argument("--content-dir", type=str, default="images/content", help="Directory to save content images")
    parser.add_argument("--style-dir", type=str, default="images/style", help="Directory to save style images")
    
    args = parser.parse_args()
    download_and_extract_images(args.content_dir, args.style_dir)

if __name__ == "__main__":
    main() 