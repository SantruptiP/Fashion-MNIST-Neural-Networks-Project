# Fashion-MNIST-Neural-Networks-Project
## Overview
In this project, I have worked with the **Fashion-MNIST** dataset to design and train two neural networks for image classification. The task is to classify each image into one of 10 possible output classes using:

1. A **Fully Feedforward Neural Network (FFN)**.
2. A **Convolutional Neural Network (CNN)**.

Experimenting with various hyperparameters and architectural decisions to optimize performance.

## Dataset
**Fashion-MNIST** is a dataset of grayscale images, each 28x28 pixels, representing different clothing items. It includes:
- **60,000 training images**.
- **10,000 testing images**.
- **10 classes**, such as T-shirt, coat, bag, etc.

This dataset can serve as a direct replacement for the MNIST handwritten digits dataset.  
For more details, visit the [official repository](https://github.com/zalandoresearch/fashion-mnist).

## Getting Started
The data is automatically downloaded within the script using PyTorch. 
- Preprocess the data as required for training.
- Design and implement the two neural networks (FFN and CNN).

## Project Structure
The project includes the following files:

### `fashionmnist.py`
Overall training and evaluation pipeline.

### `ffn.py`
Code for the feedforward neural network.

### `cnn.py`
Code for the convolutional neural network.

## Tasks
1. Implementation of the **FFN** and **CNN** models in their respective files.
2. Train and evaluate both models using the Fashion-MNIST dataset.
3. Experiment with hyperparameters, including but not limited to:
   - Number of layers.
   - Number of neurons per layer.
   - Convolutional kernel sizes and counts.
   - Learning rates, optimizers, and batch sizes.

## Deliverables
- Trained FFN and CNN models achieving optimal performance.
- Documentation of the experimental process and results.

## Requirements
- **PyTorch**: Ensure PyTorch is installed in your development environment.  
  You can install it by following the instructions at [pytorch.org](https://pytorch.org/).
