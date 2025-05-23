# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: research
#     language: python
#     name: python3
# ---

# %% 
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.notebook import tqdm

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# # Introduction to Convolutional Neural Networks (CNNs)
# 
# In previous lectures, we've worked with Multi-Layer Perceptrons (MLPs) for image classification tasks. While MLPs can learn to classify images, they have significant limitations:
# 
# 1. **Loss of spatial information**: MLPs flatten the image into a 1D vector, losing the spatial relationships between pixels
# 2. **Parameter explosion**: For high-resolution images, MLPs require huge numbers of parameters (e.g., a 224×224×3 image would need 150,528 weights for just the first layer)
# 3. **No translation invariance**: MLPs don't naturally handle shifted versions of the same pattern
# 
# Convolutional Neural Networks (CNNs) are specifically designed to address these limitations and have become the standard approach for most computer vision tasks.

# %% [markdown]
# ## Key Building Blocks of CNNs
# 
# ### 1. Convolutional Layers
# 
# Convolutional layers are the core building block of CNNs. Unlike fully connected layers in MLPs, convolutional layers:
# 
# - Apply a set of learnable filters (kernels) to the input
# - Each filter slides (convolves) across the width and height of the input
# - Compute dot products between the filter weights and the input at each position
# - Produce a 2D activation map of that filter's responses
# 
# **Benefits of convolutional layers:**
# 
# - **Parameter sharing**: The same filter is applied to every position in the image
# - **Local connectivity**: Each neuron is only connected to a small region of the input
# - **Translation invariance**: The same feature can be detected regardless of its position

# %% [markdown]
# **Exercise 1:** Create a simple convolution example. Complete the code below to:
# 1. Create a 3×3 edge detection filter
# 2. Apply it to a simple image (a white square on a black background)
# 3. Visualize the input and output

# %%
# Your task: Complete the code to create a convolution example

# Create a simple edge detection filter
edge_filter = torch.tensor([
    # Your code here: define a 3×3 edge detection filter
    # Hint: A common edge detection filter has -1 in all cells except the center, which is 8
]).float().view(1, 1, 3, 3)  # (out_channels, in_channels, height, width)

# Create a simple input (white square on black background)
input_image = torch.zeros(1, 1, 8, 8)  # (batch_size, channels, height, width)
# Your code here: Set a region in the middle to be white (value 1.0)

# Create a convolutional layer with our predefined filter
conv_layer = nn.Conv2d(1, 1, kernel_size=3, bias=False)
with torch.no_grad():
    conv_layer.weight = nn.Parameter(edge_filter)

# Apply convolution
output = conv_layer(input_image)

# Visualize input and output
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(input_image[0, 0].numpy(), cmap='gray')
axes[0].set_title('Input Image')
axes[0].axis('off')

axes[1].imshow(output[0, 0].detach().numpy(), cmap='gray')
axes[1].set_title('After Edge Detection Convolution')
axes[1].axis('off')

plt.show()

# %% [markdown]
# **Solution:**

# %%
# Create a simple edge detection filter
edge_filter = torch.tensor([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]).float().view(1, 1, 3, 3)  # (out_channels, in_channels, height, width)

# Create a simple input (white square on black background)
input_image = torch.zeros(1, 1, 8, 8)  # (batch_size, channels, height, width)
input_image[0, 0, 2:6, 2:6] = 1.0  # Set middle 4×4 square to white

# Create a convolutional layer with our predefined filter
conv_layer = nn.Conv2d(1, 1, kernel_size=3, bias=False)
with torch.no_grad():
    conv_layer.weight = nn.Parameter(edge_filter)

# Apply convolution
output = conv_layer(input_image)

# Visualize input and output
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(input_image[0, 0].numpy(), cmap='gray')
axes[0].set_title('Input Image')
axes[0].axis('off')

axes[1].imshow(output[0, 0].detach().numpy(), cmap='gray')
axes[1].set_title('After Edge Detection Convolution')
axes[1].axis('off')

plt.show()

# %% [markdown]
# In this example, the edge detection filter highlights the boundaries of the square. In a CNN, these filters are learned during training rather than being manually defined.
# 
# **Key parameters in PyTorch's Conv2d layer:**
# 
# - `kernel_size`: Size of the convolutional filter (e.g., 3×3, 5×5)
# - `stride`: Step size when sliding the filter (default=1)
# - `padding`: Zero-padding added to input (default=0)
# - `in_channels`: Number of input channels
# - `out_channels`: Number of output channels (number of filters)

# %% [markdown]
# ### 2. Pooling Layers
# 
# Pooling layers reduce the spatial dimensions of the feature maps, which:
# 
# - Reduces computation in the network
# - Controls overfitting
# - Makes the detection more robust to the position of features
# 
# The most common type is **max pooling**, which takes the maximum value in each window.

# %% [markdown]
# **Exercise 2:** Implement max pooling. Complete the code below to:
# 1. Create a sample feature map
# 2. Apply max pooling to it
# 3. Visualize the result

# %%
# Your task: Implement max pooling

# Create a sample feature map (4×4)
feature_map = torch.tensor([
    # Your code here: create a 4×4 matrix with some values
    # Choose values that will show the effect of max pooling clearly
]).float().view(1, 1, 4, 4)  # (batch_size, channels, height, width)

# Create max pooling layer (2×2 window with stride 2)
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

# Apply max pooling
output = pool_layer(feature_map)

# Visualize input and output
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(feature_map[0, 0].numpy(), cmap='viridis')
axes[0].set_title('Original Feature Map')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f"{feature_map[0, 0, i, j]:.1f}", 
                   ha="center", va="center", color="w", fontsize=12)
axes[0].axis('off')

axes[1].imshow(output[0, 0].detach().numpy(), cmap='viridis')
axes[1].set_title('After Max Pooling (2×2)')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{output[0, 0, i, j]:.1f}", 
                   ha="center", va="center", color="w", fontsize=12)
axes[1].axis('off')

plt.show()

# %% [markdown]
# **Solution:**

# %%
# Create a sample feature map (4×4)
feature_map = torch.tensor([
    [1.0, 2.0, 5.0, 6.0],
    [3.0, 4.0, 7.0, 8.0],
    [9.0, 8.0, 5.0, 4.0],
    [7.0, 6.0, 3.0, 2.0]
]).float().view(1, 1, 4, 4)  # (batch_size, channels, height, width)

# Create max pooling layer (2×2 window with stride 2)
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

# Apply max pooling
output = pool_layer(feature_map)

# Visualize input and output
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(feature_map[0, 0].numpy(), cmap='viridis')
axes[0].set_title('Original Feature Map')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f"{feature_map[0, 0, i, j]:.1f}", 
                   ha="center", va="center", color="w", fontsize=12)
axes[0].axis('off')

axes[1].imshow(output[0, 0].detach().numpy(), cmap='viridis')
axes[1].set_title('After Max Pooling (2×2)')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{output[0, 0, i, j]:.1f}", 
                   ha="center", va="center", color="w", fontsize=12)
axes[1].axis('off')

plt.show()

# %% [markdown]
# In this example, the max pooling layer takes a 2×2 window and outputs the maximum value in each window. Note how the output size is reduced from 4×4 to 2×2.

# %% [markdown]
# ### 3. Activation Functions (ReLU)
# 
# Activation functions introduce non-linearity into the network. The most common activation in CNNs is the Rectified Linear Unit (ReLU), which is defined as:
# 
# $$f(x) = \max(0, x)$$
# 
# ReLU passes through all positive values unchanged and sets all negative values to zero.

# %% [markdown]
# **Exercise 3:** Implement ReLU activation. Complete the code below to:
# 1. Create a feature map with both positive and negative values
# 2. Apply ReLU activation to it
# 3. Visualize the result

# %%
# Your task: Implement ReLU activation

# Create a feature map with positive and negative values
feature_map = torch.tensor([
    # Your code here: create a 4×4 matrix with positive and negative values
]).float().view(1, 1, 4, 4)  # (batch_size, channels, height, width)

# Create ReLU layer
relu_layer = nn.ReLU()

# Apply ReLU
output = relu_layer(feature_map)

# Visualize input and output
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im1 = axes[0].imshow(feature_map[0, 0].numpy(), cmap='coolwarm')
axes[0].set_title('Original Feature Map')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f"{feature_map[0, 0, i, j]:.1f}", 
                   ha="center", va="center", color="k", fontsize=12)
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(output[0, 0].detach().numpy(), cmap='coolwarm')
axes[1].set_title('After ReLU Activation')
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f"{output[0, 0, i, j]:.1f}", 
                   ha="center", va="center", color="k", fontsize=12)
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1])

plt.show()

# %% [markdown]
# **Solution:**

# %%
# Create a feature map with positive and negative values
feature_map = torch.tensor([
    [ 1.0, -2.0,  3.0, -4.0],
    [-5.0,  6.0, -7.0,  8.0],
    [ 9.0, -8.0,  7.0, -6.0],
    [-5.0,  4.0, -3.0,  2.0]
]).float().view(1, 1, 4, 4)  # (batch_size, channels, height, width)

# Create ReLU layer
relu_layer = nn.ReLU()

# Apply ReLU
output = relu_layer(feature_map)

# Visualize input and output
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im1 = axes[0].imshow(feature_map[0, 0].numpy(), cmap='coolwarm')
axes[0].set_title('Original Feature Map')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f"{feature_map[0, 0, i, j]:.1f}", 
                   ha="center", va="center", color="k", fontsize=12)
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(output[0, 0].detach().numpy(), cmap='coolwarm')
axes[1].set_title('After ReLU Activation')
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f"{output[0, 0, i, j]:.1f}", 
                   ha="center", va="center", color="k", fontsize=12)
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1])

plt.show()

# %% [markdown]
# In this example, ReLU keeps the positive values unchanged but sets all negative values to zero. This non-linearity is crucial for learning complex patterns in deep networks.

# %% [markdown]
# ## Putting It All Together: CNN Architecture
# 
# A typical CNN consists of several convolutional layers, each followed by ReLU activation, with occasional pooling layers to reduce the spatial dimensions. The final layers are usually fully connected (like in an MLP) for classification.
# 
# Here's a typical CNN architecture for image classification:

# %%
# Simple CNN architecture diagram
plt.figure(figsize=(12, 4))
plt.axis('off')
plt.text(0.5, 0.5, '''
Input Image → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → ... → Flatten → FC → ReLU → FC → Output
''', ha='center', va='center', fontsize=14)
plt.title('Typical CNN Architecture', fontsize=16)
plt.show()

# %% [markdown]
# ## CIFAR-10 Image Classification Task
# 
# In this notebook, we'll be working with the CIFAR-10 dataset, which consists of 60,000 32×32 color images across 10 classes:
# - Airplane
# - Automobile
# - Bird
# - Cat
# - Deer
# - Dog
# - Frog
# - Horse
# - Ship
# - Truck
# 
# Our goal is to build a CNN that can accurately classify these images.

# %%
# Load and explore the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                       shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# Visualize some example images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Get random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
plt.figure(figsize=(10, 4))
imshow(torchvision.utils.make_grid(images[:8]))
plt.axis('off')
# Print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))

# %% [markdown]
# ## AlexNet Architecture for CIFAR-10
# 
# Now, we'll implement the AlexNet architecture for CIFAR-10 classification. AlexNet is a pioneering CNN architecture that achieved breakthrough performance on the ImageNet competition in 2012.
# 
# We've adapted AlexNet to work with the smaller 32×32 CIFAR-10 images while maintaining its key architectural features.

# %% [markdown]
# **Exercise 4:** Implement AlexNet for CIFAR-10. Complete the code below to create the AlexNet architecture:

# %%
# Your task: Implement AlexNet for CIFAR-10

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # TODO: Implement the feature extraction layers
        self.features = nn.Sequential(
            # Conv1: 64 kernels of size 11×11, stride 4, followed by ReLU
            # Modified for CIFAR-10's 32×32 images
            # YOUR CODE HERE
            
            # Add more layers according to the AlexNet architecture:
            # - Max Pooling after Conv1
            # - Conv2 with 192 kernels of size 5×5
            # - Max Pooling after Conv2
            # - Conv3 with 384 kernels of size 3×3
            # - Conv4 with 384 kernels of size 3×3
            # - Conv5 with 256 kernels of size 3×3
            # - Max Pooling after Conv5
            # YOUR CODE HERE
        )
        
        # TODO: Implement the classifier (fully connected layers)
        self.classifier = nn.Sequential(
            # FC6: 4096 units + ReLU + Dropout(0.5)
            # YOUR CODE HERE
            
            # FC7: 4096 units + ReLU + Dropout(0.5)
            # YOUR CODE HERE
            
            # FC8: num_classes units (10 for CIFAR-10)
            # YOUR CODE HERE
        )
        
    def forward(self, x):
        # TODO: Implement the forward pass
        # YOUR CODE HERE
        return x

# %% [markdown]
# **Solution:**

# %%
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1: 64 kernels of size 11×11, stride 4, followed by ReLU
            # Modified stride and added padding to work with 32x32 CIFAR-10 images
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),
            # Max Pooling: 3×3 window, stride 2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv2: 192 kernels of size 5×5, padding 2, followed by ReLU
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # Max Pooling: 3×3 window, stride 2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv3: 384 kernels of size 3×3, padding 1, followed by ReLU
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 384 kernels of size 3×3, padding 1, followed by ReLU
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 256 kernels of size 3×3, padding 1, followed by ReLU
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Max Pooling: 3×3 window, stride 2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            # FC6: 4096 units + ReLU + Dropout(0.5)
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            
            # FC7: 4096 units + ReLU + Dropout(0.5)
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # FC8: num_classes units (10 for CIFAR-10)
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the feature maps
        x = self.classifier(x)
        return x

# Let's check the model structure
model = AlexNet().to(device)
print(model)

# %% [markdown]
# ## Training and Evaluation Functions
# 
# Now, let's define functions for training our model and evaluating its performance:

# %%
def train_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=5):
    """
    Train the model for a specified number of epochs.
    
    Args:
        model: The neural network model
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        criterion: Loss function
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        
    Returns:
        train_losses: List of training losses for each epoch
        train_accs: List of training accuracies for each epoch
        test_losses: List of test losses for each epoch
        test_accs: List of test accuracies for each epoch
    """
    # TODO: Initialize lists to store metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # TODO: Train the model for one epoch
        # Set model to training mode
        model.train()
        
        # Initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iterate over batches
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        for i, (inputs, targets) in loop:
            # TODO: Move inputs and targets to device
            # TODO: Zero the parameter gradients
            # TODO: Forward pass
            # TODO: Compute loss
            # TODO: Backward pass
            # TODO: Optimizer step
            # TODO: Update metrics
            # TODO: Update progress bar
            
            # Set model to evaluation mode
            model.eval()
            
            # Initialize test metrics
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            # Disable gradient calculation for evaluation
            with torch.no_grad():
                # TODO: Iterate over test data
                # TODO: Move inputs and targets to device
                # TODO: Forward pass
                # TODO: Compute loss
                # TODO: Update metrics
            
            # TODO: Calculate epoch metrics
            # TODO: Update learning rate
            # TODO: Record statistics
            # TODO: Print epoch summary
            # TODO: Return training and test metrics
            
    return train_losses, train_accs, test_losses, test_accs

# %% [markdown]
# ## Helper Function for Visualizing Conv1 Kernels

# %%
def plot_conv1_kernels(model):
    """
    Plot the kernels of the first convolutional layer.
    
    Args:
        model: Trained AlexNet model
    """
    # Extract weights from the first convolutional layer
    # Shape is [64, 3, 11, 11] - [out_channels, in_channels, height, width]
    weights = model.features[0].weight.data.cpu()
    
    # Normalize weights for better visualization
    min_val = weights.min()
    max_val = weights.max()
    weights = (weights - min_val) / (max_val - min_val)
    
    # Create a grid of all kernels
    num_kernels = weights.size(0)
    grid_size = int(np.ceil(np.sqrt(num_kernels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    # Plot each kernel
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            # Get the kernel and permute dimensions for display
            # From [3, 11, 11] (C, H, W) to [11, 11, 3] (H, W, C)
            kernel = weights[i].permute(1, 2, 0)
            
            # Display the kernel
            ax.imshow(kernel)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle("First Convolutional Layer Kernels", fontsize=16)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Exercise 5:** Complete the training and evaluation functions. Fill in the missing parts in the train_model function:

# %%
def train_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=5):
    """
    Train the model for a specified number of epochs.
    
    Args:
        model: The neural network model
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        criterion: Loss function
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        
    Returns:
        train_losses: List of training losses for each epoch
        train_accs: List of training accuracies for each epoch
        test_losses: List of test losses for each epoch
        test_accs: List of test accuracies for each epoch
    """
    # Initialize lists to store metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iterate over batches
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        for i, (inputs, targets) in loop:
            # TODO: Move inputs and targets to device
            
            # TODO: Zero the parameter gradients
            
            # TODO: Forward pass
            
            # TODO: Compute loss
            
            # TODO: Backward pass
            
            # TODO: Optimizer step
            
            # TODO: Update metrics
            
            # TODO: Update progress bar
        
        # Calculate training metrics for this epoch
        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = 100.0 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                # TODO: Move inputs and targets to device
                
                # TODO: Forward pass
                
                # TODO: Compute loss
                
                # TODO: Update metrics
        
        # Calculate test metrics for this epoch
        epoch_test_loss = test_loss / len(testloader)
        epoch_test_acc = 100.0 * correct / total
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
    
    return train_losses, train_accs, test_losses, test_accs

# %% [markdown]
# **Solution:**

# %%
def train_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=5):
    """
    Train the model for a specified number of epochs.
    
    Args:
        model: The neural network model
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        criterion: Loss function
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        
    Returns:
        train_losses: List of training losses for each epoch
        train_accs: List of training accuracies for each epoch
        test_losses: List of test losses for each epoch
        test_accs: List of test accuracies for each epoch
    """
    # Initialize lists to store metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iterate over batches
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        for i, (inputs, targets) in loop:
            # Move inputs and targets to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            loop.set_postfix(loss=running_loss/(i+1), acc=100.*correct/total)
        
        # Calculate training metrics for this epoch
        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = 100.0 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                # Move inputs and targets to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, targets)
                
                # Update metrics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate test metrics for this epoch
        epoch_test_loss = test_loss / len(testloader)
        epoch_test_acc = 100.0 * correct / total
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
    
    return train_losses, train_accs, test_losses, test_accs

# %% [markdown]
# **Exercise 6:** Train the AlexNet model on CIFAR-10. Complete the code below to:
# 1. Create an instance of the AlexNet model
# 2. Define loss function, optimizer, and learning rate scheduler
# 3. Train the model using the train_model function
# 4. Plot the training and test curves (loss and accuracy)

# %%
# Your task: Train AlexNet on CIFAR-10

# TODO: Create an instance of AlexNet

# TODO: Define loss function, optimizer, and learning rate scheduler

# TODO: Train the model

# TODO: Plot the training and test curves

# %% [markdown]
# **Solution:**

# %%
# Create an instance of AlexNet
model = AlexNet().to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Train the model
train_losses, train_accs, test_losses, test_accs = train_model(
    model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=5
)

# Plot the training and test curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss')
plt.plot(range(1, len(test_losses)+1), test_losses, 'r-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accs)+1), train_accs, 'b-', label='Train Accuracy')
plt.plot(range(1, len(test_accs)+1), test_accs, 'r-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Test Accuracy')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualizing the Conv1 Kernels
# 
# Now, let's visualize the first convolutional layer kernels to see what the network has learned:

# %%
# Visualize Conv1 kernels
plot_conv1_kernels(model)

# %% [markdown]
# ## Analysis of Conv1 Kernels
# 
# The kernels in the first convolutional layer have learned to detect various low-level features:
# 
# - **Edge detectors**: Kernels that highlight horizontal, vertical, or diagonal edges
# - **Color detectors**: Kernels that respond to specific colors (red, green, blue patches)
# - **Texture detectors**: Kernels that respond to specific patterns
# 
# Each of the 64 filters in the first layer specializes in detecting a particular pattern. These low-level features are then combined in deeper layers to detect more complex patterns such as shapes, textures, and eventually entire objects.
# 
# ## Conclusion
# 
# In this notebook, we've:
# 
# 1. Learned about the fundamental building blocks of CNNs (convolution, pooling, and ReLU)
# 2. Understood how these components work together in a CNN architecture
# 3. Implemented AlexNet for CIFAR-10 image classification
# 4. Trained the model and evaluated its performance
# 5. Visualized the learned filters from the first convolutional layer

# Convolutional Neural Networks have revolutionized computer vision and remain the backbone of many vision-based applications. The principles you've learned in this notebook extend to more modern architectures like VGG, ResNet, and beyond.