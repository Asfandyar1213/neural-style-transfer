import torch
import torch.nn as nn
import torchvision.models as models

class VGG19FeatureExtractor(nn.Module):
    """
    Feature extractor using VGG19 pretrained model.
    Extracts features from specific layers for content and style representation.
    """
    def __init__(self, device="cpu"):
        super(VGG19FeatureExtractor, self).__init__()
        
        # Load pretrained VGG19 model
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        
        # Create a module list from VGG19 features
        self.features = nn.ModuleList(vgg)
        
        # Define layer indices for feature extraction
        # Format: {layer_name: layer_index}
        self.layer_indices = {
            'relu1_1': 1,  # After first conv layer
            'relu2_1': 6,  # After second conv block
            'relu3_1': 11, # After third conv block
            'relu4_1': 20, # After fourth conv block
            'relu5_1': 29  # After fifth conv block
        }
        
        # Freeze parameters of the model
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through the network, extracting features at specified layers.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            dict: Dictionary of feature maps at specified layers
        """
        features = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            for name, index in self.layer_indices.items():
                if idx == index:
                    features[name] = x
        return features


class TransformerNet(nn.Module):
    """
    Fast Neural Style Transfer transformation network.
    This network can be trained to perform style transfer in a single forward pass.
    """
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        
        # Residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        
        # Non-linearities
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(nn.Module):
    """
    Custom convolutional layer with reflection padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions and instance normalization
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    """
    Upsampling layer with convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out 