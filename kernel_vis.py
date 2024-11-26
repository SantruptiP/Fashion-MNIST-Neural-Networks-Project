import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
import matplotlib.pyplot as plt
import torchvision.utils as vutils


conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network
first_conv_layer = conv_net.conv1  # Adjust this based on the actual layer name in Conv_Net
kernels = first_conv_layer.weight.data.clone()  # Clone to avoid modifying the model
kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())  # Normalize to [0, 1]

''' YOUR CODE HERE '''


# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, 
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be 
# between 0 and 1 before plotting.
# Arrange kernels in a grid
grid = vutils.make_grid(kernels, nrow=8, padding=1)  # Adjust nrow based on number of kernels

# Plot and save the grid
plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title("Kernels at the First Conv Layer")

''' YOUR CODE HERE '''

# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.
plt.savefig('kernel_grid.png')
''' YOUR CODE HERE '''


# Apply the kernel to the provided sample image.

img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0					# Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image
with torch.no_grad():
    output = first_conv_layer(img)  # Pass through only the first conv layer

print("Output feature maps shape:", output.shape)

output = output.squeeze(0).unsqueeze(1)
# Normalize the output feature maps to [0, 1] for visualization
output = (output - output.min()) / (output.max() - output.min())


# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.

#output = output.squeeze(0)
#output = output.unsqueeze(1)


# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.
grid = vutils.make_grid(output, nrow=8, padding=1)  # Adjust nrow based on number of kernels

# Plot and save the grid of feature maps
plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title("Feature Maps Extracted from Sample Image")

# Save the grid to a file named 'image_transform_grid.png'
plt.savefig('image_transform_grid.png')

''' YOUR CODE HERE '''















