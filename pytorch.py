# Import writehost module from utils package
from utils.writehost import *
from colorama import Fore, Back

clear_screen()
print_header("Importing Modules")

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
# -------------------------------------------------------------------------------
print_header("Path Definitions")    

# path definitions
data_dir = '.\\datasets'
model_dir = '.\\models'

train_data_path = f'{data_dir}\\train_data.npy'
train_labels_path = f'{data_dir}\\train_labels.npy'
test_data_path = f'{data_dir}\\test_data.npy'
test_labels_path = f'{data_dir}\\test_labels.npy'

model_path = f'{model_dir}\\pytorch_model.pth'

# Create model directory if it doesn't exist
if not os.path.exists(model_dir): os.makedirs(model_dir)

# Print confirmation message and data shapes
print(f"Training data path:   {Fore.GREEN}{train_data_path:>35}{Fore.RESET}")
print(f"Training labels path: {Fore.GREEN}{train_labels_path:>35}{Fore.RESET}")
print(f"Test data path:       {Fore.GREEN}{test_data_path:>35}{Fore.RESET}")
print(f"Test labels path:     {Fore.GREEN}{test_labels_path:>35}{Fore.RESET}")
print(f"\nModel path:          {Fore.GREEN}{model_path:>35}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Loading MNIST Dataset") 

# Example of loading the data using the same path variables
train_data = np.load(train_data_path)
train_labels = np.load(train_labels_path)
test_data = np.load(test_data_path)
test_labels = np.load(test_labels_path)

# Print confirmation message and data shapes
print(f"Training data shape:   {Fore.GREEN}{train_data.shape}{Fore.RESET}")
print(f"Training labels shape: {Fore.GREEN}{train_labels.shape}{Fore.RESET}") 
print(f"Test data shape:       {Fore.GREEN}{test_data.shape}{Fore.RESET}")
print(f"Test labels shape:     {Fore.GREEN}{test_labels.shape}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Normalizing Data") 

# normalize the data
train_data = train_data / 255.0
test_data = test_data / 255.0   

# Print confirmation message and data ranges
print(f"{Fore.WHITE}Data normalized to range [{Fore.GREEN}0,1{Fore.WHITE}]{Fore.RESET}")
print(f"Training data range:   [{Fore.GREEN}{train_data.min():>8.6f}, {Fore.GREEN}{train_data.max():>8.6f}{Fore.RESET}]")
print(f"Test data range:       [{Fore.GREEN}{test_data.min():>8.6f}, {Fore.GREEN}{test_data.max():>8.6f}{Fore.RESET}]")

# -------------------------------------------------------------------------------
print_header("Converting Labels to One-Hot Encoded Format")

# Convert labels to one-hot encoded format using numpy
train_labels_onehot = np.zeros((len(train_labels), 10))
test_labels_onehot = np.zeros((len(test_labels), 10))

# Fill the one-hot encoded arrays - set 1 at the index corresponding to the label
for i, label in enumerate(train_labels):
    train_labels_onehot[i, label] = 1
    
for i, label in enumerate(test_labels):
    test_labels_onehot[i, label] = 1

# Print confirmation message and data shapes
print(f"Training labels one-hot shape: {Fore.GREEN}{train_labels_onehot.shape}{Fore.RESET}")
print(f"Test labels one-hot shape:     {Fore.GREEN}{test_labels_onehot.shape}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Reshaping Data from 2D images (28x28) to 1D vectors (784)")

# reshape data
train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)

# Print confirmation message and data shapes
print(f"Training data shape:   {Fore.GREEN}{train_data.shape}{Fore.RESET}")
print(f"Test data shape:       {Fore.GREEN}{test_data.shape}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Defining the Model")

# Set random seed for reproducibility
torch.manual_seed(112358)

# Create sequential model
model = nn.Sequential(
    # Input layer to first hidden layer: 784 -> 256 with ReLU activation
    nn.Linear(784, 128),
    nn.ReLU(),
    
    # Second hidden layer to third hidden layer: 128 -> 64 with ReLU activation
    nn.Linear(128, 128),
    nn.ReLU(),
    
    # Third hidden layer to output layer: 64 -> 10 with Softmax activation
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

# Print model architecture
print(f"{Fore.WHITE}Model architecture:{Fore.RESET}")
print(f"{Fore.GREEN}{model}{Fore.RESET}")

# Print parameter summary
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters:      {Fore.GREEN}{total_params:>15}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Training Configuration")

# Define loss function - CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Create optimizer with learning rate 0.001
optimizer = Adam(model.parameters(), lr=0.001)

# Set training hyperparameters
epochs = 10
batch_size = 32

# Print optimizer and loss function configuration
print(f"{Fore.WHITE}Training configuration:{Fore.RESET}")
print(f"Optimizer:            {Fore.GREEN}{'Adam':>15}{Fore.RESET}")
print(f"Learning rate:        {Fore.GREEN}{0.001:>15.3f}{Fore.RESET}")
print(f"Loss function:        {Fore.GREEN}{'CrossEntropy':>15}{Fore.RESET}")
print(f"Epochs:             {Fore.GREEN}{epochs:>15}{Fore.RESET}")
print(f"Batch size:         {Fore.GREEN}{batch_size:>15}{Fore.RESET}")


# -------------------------------------------------------------------------------
print_header("Converting Data to PyTorch Tensors")


# Convert data to PyTorch tensors
train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels)
test_data = torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels)

# Print confirmation message and data shapes
print(f"Training data shape:   {Fore.GREEN}{train_data.shape}{Fore.RESET}")
print(f"Training labels shape: {Fore.GREEN}{train_labels.shape}{Fore.RESET}") 
print(f"Test data shape:       {Fore.GREEN}{test_data.shape}{Fore.RESET}")
print(f"Test labels shape:     {Fore.GREEN}{test_labels.shape}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Training the Model")

# Training loop
for epoch in range(epochs):
    # Initialize metrics for this epoch
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create mini-batches for training
    for i in range(0, len(train_data), batch_size):
        # Get mini-batch
        batch_x = train_data[i:i + batch_size]
        batch_y = train_labels[i:i + batch_size]
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x)
        
        # Calculate loss
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    # Calculate epoch metrics
    epoch_loss = running_loss / (len(train_data) / batch_size)
    epoch_acc = 100 * correct / total
    
    # Print epoch results
    print(f"Epoch {epoch + 1}/{epochs}:")
    print(f"Loss:                 {epoch_loss:>15.4f}")
    print(f"Accuracy:             {epoch_acc:>14.2f}%")
    print("-" * 45)

# -------------------------------------------------------------------------------
print_header("Evaluating the Model")

# Set model to evaluation mode
model.eval()

# Initialize metrics for evaluation
test_loss = 0.0
correct = 0
total = 0

# Disable gradient calculation for evaluation
with torch.no_grad():
    # Create mini-batches for testing
    for i in range(0, len(test_data), batch_size):
        # Get mini-batch
        batch_x = test_data[i:i + batch_size]
        batch_y = test_labels[i:i + batch_size]
        
        # Forward pass
        outputs = model(batch_x)
        
        # Calculate loss
        loss = criterion(outputs, batch_y)
        
        # Update metrics
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

# Calculate final metrics        
test_loss = test_loss / (len(test_data) / batch_size)
test_acc = 100 * correct / total

# Print final metrics
print(f"Test loss:             {test_loss:>15.4f}")
print(f"Test accuracy:        {test_acc:>14.2f}%")


# -------------------------------------------------------------------------------
print_header("Saving the Model")

# Save the model
torch.save(model, model_path)

# Print confirmation message
print(f"Model saved to {Fore.GREEN}{model_path}{Fore.RESET}")

