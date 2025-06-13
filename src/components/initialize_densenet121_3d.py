import torch
import torchvision.models as models
import torch.nn as nn
import copy
from src.exception import CustomException
from src.logger import logging
import sys

def inflate_densenet121_to_3d():

    try:
        
        # Loading pre-trained model
        model_2d = models.densenet121(pretrained=True)
        print(model_2d.features)
        model_2d.eval()

        #Defining the 3D model class
        class DenseNet1213D(nn.Module):
            def __init__(self, model_2d):
                super(DenseNet1213D, self).__init__()


                def inflate_module(module, is_conv0=False):

                    # Is current layer nn.Conv2d? 
                    if isinstance(module, nn.Conv2d): 

                        # Extracting parameters from 2d
                        out_channels = module.out_channels
                        in_channels = 1 if is_conv0 else module.in_channels
                        kernel_size = module.kernel_size[0] # For square kernel
                        stride = module.stride[0]
                        padding = module.padding[0]
                        
                        weight_2d = module.weight.data # Shape: (out_channel, in_channel, h, w)
                        if is_conv0:
                            # Use first channel of RGB weights for grayscale
                            weight_2d = weight_2d[:, :1, :, :]

                        depth = kernel_size # Depth dimension

                        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth # adding new dim at 2nd index -> (out_channels, in_channels, depth, h, w) -> Normalization

                        # Defining the conv3d block
                        conv3d = nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(kernel_size, kernel_size, kernel_size),
                            stride=(stride, stride, stride),
                            padding=(padding,padding,padding)
                        )

                        conv3d.weight.data = weight_3d
                        
                        # For bias
                        if module.bias is not None:
                            conv3d.bias.data = module.bias.data

                        return conv3d
                    

                    # Handling non-convolutional layers

                    # MaxPool layer
                    elif isinstance(module, nn.MaxPool2d):
                        return nn.MaxPool3d(
                            kernel_size=module.kernel_size,
                            stride=module.stride,
                            padding=module.padding
                        )

                    # AvgPool layer
                    elif isinstance(module, nn.AvgPool2d):
                        return nn.AvgPool3d(
                            kernel_size=module.kernel_size,
                            stride=module.stride,
                            padding=module.padding
                        )    
                        
                    # BatchNorm Layer    
                    elif isinstance(module, nn.BatchNorm2d):
                        # Convert BatchNorm2d to BatchNorm3d
                        batchnorm3d = nn.BatchNorm3d(module.num_features)
                        if hasattr(module, 'weight') and module.weight is not None:
                            batchnorm3d.weight.data = module.weight.data
                        if hasattr(module, 'bias') and module.bias is not None:
                            batchnorm3d.bias.data = module.bias.data
                        if hasattr(module, 'running_mean'):
                            batchnorm3d.running_mean = module.running_mean
                        if hasattr(module, 'running_var'):
                            batchnorm3d.running_var = module.running_var
                        return batchnorm3d

                    elif isinstance(module, (nn.Sequential, nn.Module)):
                        # Recursively inflate submodules
                        new_module = copy.deepcopy(module)
                        for name, child in new_module.named_children():
                            # Pass is_conv0=True for conv0
                            setattr(new_module, name, inflate_module(child, is_conv0=(name == 'conv0')))
                        return new_module
                    else:
                        # Return unchanged layers (e.g., ReLU)
                        return module
                
                # Create features by inflating the 2D model's features
                self.features = inflate_module(model_2d.features)

            def forward(self, x):
                return self.features(x)
        
        return DenseNet1213D(model_2d)
    
    except Exception as e:
        logging.info("Error during Inflation 2D --> 3D")
        raise CustomException(e, sys) 


def initialize_model():
    """
    Initialize the 3D DenseNet121 model, move it to the appropriate device,
    and verify its structure and functionality.
    """
    print("Initializing 3D DenseNet121 model...")
    
    # Create model instance
    model_3d = inflate_densenet121_to_3d()
    
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Move model to device and set to evaluation mode
    model_3d = model_3d.to(device)
    model_3d.eval()
    print("Model initialized and moved to device.")

    return model_3d, device


if __name__ == "__main__":
    # Initialize and verify model
    model_3d, device = initialize_model()
    print("\nModel is ready for use in Task III pipeline.")