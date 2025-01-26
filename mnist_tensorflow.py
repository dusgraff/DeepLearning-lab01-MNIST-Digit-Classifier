# Import writehost module from utils package
from utils.writehost import *
from colorama import Fore, Back

clear_screen()
print_header("Importing Modules")

import os
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

# -------------------------------------------------------------------------------
print_header("Path Definitions")    

# path definitions
data_dir = '.\\datasets'
model_dir = '.\\models'

train_data_path = f'{data_dir}\\train_data.npy'
train_labels_path = f'{data_dir}\\train_labels.npy'
test_data_path = f'{data_dir}\\test_data.npy'
test_labels_path = f'{data_dir}\\test_labels.npy'

model_path = f'{model_dir}\\tensorflow_model.keras'

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

# Convert labels to one-hot encoded format using to_categorical and store in new variables
train_labels_categorical = to_categorical(train_labels)
test_labels_categorical = to_categorical(test_labels)

# Print confirmation message and label shapes
print("Labels converted to one-hot encoded format")
print(f"Training labels shape: {str(train_labels_categorical.shape):>15}")
print(f"Test labels shape:     {str(test_labels_categorical.shape):>15}")

# -------------------------------------------------------------------------------
print_header("Reshaping Data from 2D images (28x28) to 1D vectors (784)")

# Get the original dimensions of the training data
original_shape = train_data.shape

# Reshape training data from (60000, 28, 28) to (60000, 784) - flattening each 28x28 image into a 784 length vector
train_input = train_data.reshape(original_shape[0], -1)

# Reshape test data in the same way
test_input = test_data.reshape(test_data.shape[0], -1)

# Print confirmation message and reshaped data dimensions
print("Data reshaped to flat vectors")
print(f"Training data shape: {str(train_input.shape):>15}")
print(f"Test data shape:     {str(test_input.shape):>15}")

# -------------------------------------------------------------------------------
print_header("Defining the Model")

# Set random seed for reproducibility
np.random.seed(112358)
tf.random.set_seed(112358)

# Create a sequential model
model = tf.keras.Sequential()

# Add input layer that matches flattened input shape of 784 nodes (28x28 pixels)
model.add(tf.keras.layers.Dense(784, input_shape=(784,)))

# Add hidden layer with 128 nodes and ReLU activation
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add output layer with 10 nodes (one per digit) and softmax activation
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# -------------------------------------------------------------------------------
print_header("Training Configuration")

# Compile the model with Adam optimizer and categorical crossentropy loss function
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Set training parameters
epochs = 10
batch_size = 32

# Print optimizer and loss function configuration
print(f"{Fore.WHITE}Training configuration:{Fore.RESET}")
print(f"Optimizer:          {Fore.GREEN}{'Adam':>15}{Fore.RESET}")
print(f"Loss function:      {Fore.GREEN}{'Categorical Crossentropy':>15}{Fore.RESET}")
print(f"Epochs:             {Fore.GREEN}{epochs:>15}{Fore.RESET}")
print(f"Batch size:         {Fore.GREEN}{batch_size:>15}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Training the Model")
# Train the model and store training history
history = model.fit(
    train_input, train_labels_categorical,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)

# -------------------------------------------------------------------------------
print_header("Evaluating the Model")

# Evaluate model performance on test data
test_loss, test_accuracy = model.evaluate(test_input, test_labels_categorical, verbose=0)

# Print test results with aligned formatting
print("Model evaluation on test data:")
print(f"Test loss:     {test_loss:>15.4f}")
print(f"Test accuracy: {test_accuracy:>15.4f}")

# -------------------------------------------------------------------------------
print_header("Saving the Model")

# Save the model to a file
model.save(model_path)

# Print confirmation message
print(f"Model saved to {model_path}")
