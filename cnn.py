import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.

'''

class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        # First convolutional layer (input channels=1 for grayscale, output channels=32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer (2x2 window)
        
        # Second convolutional layer (32 input channels, 64 output channels)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64 channels with 7x7 spatial size after pooling
        self.fc2 = nn.Linear(128, 10)          # Output layer with 10 classes

    def forward(self, x):
        # Pass through the first convolutional layer, then apply ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass through the second convolutional layer, then apply ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Pass through fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        
        # Final output layer (logits for each class)
        x = self.fc2(x)
        return x
        
