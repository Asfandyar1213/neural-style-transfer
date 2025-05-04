import os
import sys
import uuid
import torch
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, url_for, redirect, send_from_directory
import torchvision.transforms as transforms
from io import BytesIO
import base64

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from models import TransformerNet, VGG19FeatureExtractor
from utils import ensure_dir
from style_transfer import neural_style_transfer, load_image

# Configure app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MODELS_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['MODELS_FOLDER']]:
    ensure_dir(folder)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
def load_models():
    # Load VGG19 for optimization-based style transfer
    vgg = VGG19FeatureExtractor().to(device)
    return vgg

# Get list of pretrained style models (if any)
def get_pretrained_models():
    models = []
    model_dir = app.config['MODELS_FOLDER']
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pth'):
                name = os.path.splitext(file)[0].replace('_', ' ').title()
                models.append({'name': name, 'file': file})
    return models

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load a pretrained style model
def load_pretrained_model(model_file):
    model_path = os.path.join(app.config['MODELS_FOLDER'], model_file)
    model = TransformerNet().to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Handle different save formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model {model_file}: {e}")
        return None

# Generate a unique filename
def get_unique_filename(extension='.jpg'):
    return str(uuid.uuid4()) + extension

# Process using optimization-based method
def process_optimization_style(content_img, style_img, iterations=300, style_weight=1000000, content_weight=1):
    # Load the images
    content_tensor = load_image(content_img)
    style_tensor = load_image(style_img)
    
    # Resize style to match content dimensions
    style_tensor = torch.nn.functional.interpolate(style_tensor, size=content_tensor.shape[2:])
    
    # Run style transfer
    result = neural_style_transfer(
        content_tensor, 
        style_tensor,
        num_steps=iterations,
        style_weight=style_weight,
        content_weight=content_weight
    )
    
    # Convert to PIL Image and return
    output_image = result.cpu().clone().squeeze(0)
    output_image = output_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                  torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    output_image = output_image.clamp(0, 1)
    
    # Convert to PIL image
    output_image = output_image.detach().cpu().numpy()
    output_image = output_image.transpose(1, 2, 0) * 255.0
    output_image = output_image.clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_image)

# Process using fast neural style transfer with a pretrained model
def process_fast_style(content_img, model):
    # Prepare the image
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform the image
    content_image = Image.open(content_img).convert('RGB')
    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    
    # Generate stylized image
    with torch.no_grad():
        output = model(content_tensor)
    
    # Convert back to image
    output_image = output.cpu()[0]
    output_image = output_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                  torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    output_image = output_image.clamp(0, 1)
    
    # Convert to PIL image
    output_image = output_image.detach().cpu().numpy()
    output_image = output_image.transpose(1, 2, 0) * 255.0
    output_image = output_image.clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_image)

# Convert image to base64 for display in the browser
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

# Routes
@app.route('/', methods=['GET'])
def index():
    pretrained_models = get_pretrained_models()
    return render_template('index.html', pretrained_models=pretrained_models)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/process', methods=['POST'])
def process():
    try:
        # Check if files were uploaded
        if 'content_image' not in request.files:
            return render_template('error.html', message='No content image uploaded')
        
        content_file = request.files['content_image']
        if content_file.filename == '':
            return render_template('error.html', message='No content image selected')
        
        if not allowed_file(content_file.filename):
            return render_template('error.html', message='Invalid content image format')
        
        # Create unique filenames
        content_filename = get_unique_filename(os.path.splitext(content_file.filename)[1])
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
        content_file.save(content_path)
        
        # Get parameters
        method = request.form.get('method', 'optimization')
        
        # Process according to method
        if method == 'pretrained':
            # Fast style transfer with pretrained model
            model_file = request.form.get('pretrained_model')
            if not model_file:
                return render_template('error.html', message='No style model selected')
                
            # Load the model
            model = load_pretrained_model(model_file)
            if model is None:
                return render_template('error.html', message=f'Failed to load model {model_file}')
                
            # Process the image
            result_image = process_fast_style(content_path, model)
            
        else:  # optimization method
            # Check for style image
            if 'style_image' not in request.files:
                return render_template('error.html', message='No style image uploaded')
                
            style_file = request.files['style_image']
            if style_file.filename == '':
                return render_template('error.html', message='No style image selected')
                
            if not allowed_file(style_file.filename):
                return render_template('error.html', message='Invalid style image format')
                
            # Save style image
            style_filename = get_unique_filename(os.path.splitext(style_file.filename)[1])
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            style_file.save(style_path)
            
            # Get additional parameters
            iterations = int(request.form.get('iterations', 300))
            style_weight = float(request.form.get('style_weight', 1000000))
            content_weight = float(request.form.get('content_weight', 1))
            
            # Limit iterations for web app to prevent timeouts
            iterations = min(iterations, 500)
            
            # Process the images
            result_image = process_optimization_style(
                content_path, 
                style_path, 
                iterations=iterations,
                style_weight=style_weight,
                content_weight=content_weight
            )
        
        # Save the result
        output_filename = get_unique_filename('.jpg')
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        result_image.save(output_path)
        
        # Return result page with images
        content_url = url_for('static', filename=f'uploads/{content_filename}')
        output_url = url_for('static', filename=f'outputs/{output_filename}')
        
        # If style file exists, include it
        style_url = None
        if method == 'optimization' and 'style_file' in locals():
            style_url = url_for('static', filename=f'uploads/{style_filename}')
        
        return render_template(
            'result.html',
            content_url=content_url,
            style_url=style_url,
            output_url=output_url,
            method=method
        )
        
    except Exception as e:
        return render_template('error.html', message=f'Error processing images: {str(e)}')

if __name__ == '__main__':
    # Load models when app starts
    vgg = load_models()
    print(f"Running on {device}")
    app.run(debug=True, host='0.0.0.0', port=5000) 