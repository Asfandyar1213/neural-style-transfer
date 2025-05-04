# Neural Style Transfer

This project implements Neural Style Transfer using PyTorch. Neural Style Transfer is a technique that takes a content image and a style image and generates a new image that combines the content of the first image with the style of the second.

## Features

- **Optimization-based style transfer**: Apply style transfer by optimizing an image directly
- **Fast neural style transfer**: Use pre-trained models for instant stylization
- **Web application**: User-friendly interface for style transfer
- **Multiple style transfer options**: Control style weight, content weight, and iterations
- **Batch processing**: Process multiple images with one command
- **Animation generation**: Create GIFs showing the style transfer process

## Requirements
- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download sample images (optional):
```
python download_samples.py
```

## Usage

### Command-line Style Transfer

Run the main style transfer script:
```
python style_transfer.py --content images/content/your_content_image.jpg --style images/style/your_style_image.jpg --output output/result.jpg
```

#### Parameters
- `--content`: Path to content image
- `--style`: Path to style image
- `--output`: Path for saving the output image
- `--iterations`: Number of optimization steps (default: 300)
- `--style-weight`: Weight for style loss (default: 1000000)
- `--content-weight`: Weight for content loss (default: 1)
- `--lr`: Learning rate (default: 0.01)
- `--plot`: Show a plot of the content, style, and output images

### Demo with Progress Visualization

Run the demo script to see the style transfer process with visualization:
```
python demo.py --content images/content/your_content_image.jpg --style images/style/your_style_image.jpg --output-dir output/demo
```

#### Parameters
- `--content`: Path to content image
- `--style`: Path to style image
- `--output-dir`: Directory to save result images
- `--iterations`: Number of optimization steps (default: 300)
- `--style-weight`: Weight for style loss (default: 1000000)
- `--content-weight`: Weight for content loss (default: 1)
- `--lr`: Learning rate (default: 0.01)
- `--save-interval`: Interval for saving progress images (default: 50)
- `--no-animation`: Disable creation of GIF animation

### Batch Processing

Process multiple images at once:
```
python batch_process.py --input-dir images/content --model models/starry_night.pth --output-dir output/batch
```

#### Parameters
- `--input-dir`: Directory containing images or a single image path
- `--model`: Path to a saved model for fast style transfer
- `--output-dir`: Directory to save stylized images
- `--size`: Size of the output images (default: 512)

### Training Your Own Style Model

Train a fast neural style transfer model with your own style:
```
python train.py --dataset path/to/dataset --style-image images/style/your_style.jpg --checkpoint-dir checkpoints/your_style
```

#### Parameters
- `--dataset`: Path to training dataset (should contain content images)
- `--style-image`: Path to style image
- `--checkpoint-dir`: Directory to save checkpoints
- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Batch size for training (default: 4)
- `--content-weight`: Weight for content loss (default: 1.0)
- `--style-weight`: Weight for style loss (default: 5.0)

### Web Application

Run the Flask web app:
```
python app.py
```

Then open your browser and go to http://localhost:5000/

The web app allows you to:
- Upload content and style images
- Choose between optimization-based or fast style transfer methods
- Adjust style transfer parameters
- Download stylized images

## Implementation Details

The implementation uses the VGG19 model pretrained on ImageNet to extract features from content and style images. Two main approaches are implemented:

1. **Optimization-based style transfer**: Directly optimizes a generated image to match the content features of the content image and the style features (Gram matrices) of the style image.

2. **Fast neural style transfer**: Uses a trained transformation network (TransformerNet) that can stylize images in a single forward pass, providing much faster results after training.

## Examples

Example command:
```
python style_transfer.py --content images/content/landscape.jpg --style images/style/starry_night.jpg --output output/landscape_starry.jpg --iterations 500
```

## Project Structure

- `style_transfer.py`: Main implementation of neural style transfer
- `models.py`: Neural network models for feature extraction and transformation
- `utils.py`: Utility functions for image processing and visualization
- `demo.py`: Script for demonstrating style transfer with visualization
- `train.py`: Script for training fast neural style transfer models
- `stylize.py`: Script for applying trained style models to images
- `batch_process.py`: Script for batch processing multiple images
- `app.py`: Flask web application for style transfer
- `download_samples.py`: Script to download sample content and style images
- `templates/`: HTML templates for the web application
- `static/`: Static files for the web application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The implementation is inspired by the paper ["A Neural Algorithm of Artistic Style" by Gatys et al.](https://arxiv.org/abs/1508.06576)
- Fast neural style transfer is based on ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution" by Johnson et al.](https://arxiv.org/abs/1603.08155) 