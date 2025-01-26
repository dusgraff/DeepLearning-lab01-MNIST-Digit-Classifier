
# Import writehost module from utils package
from utils.writehost import *
from colorama import Fore, Back

clear_screen()
print_header("Importing Modules")

import numpy as np
import os
import shutil

# for dataset ONLY
from tensorflow.keras.datasets import mnist

# -------------------------------------------------------------------------------
print_header("Path Definitions")    

# path definitions
data_dir = '.\\datasets'

train_data_path = f'{data_dir}\\train_data.npy'
train_labels_path = f'{data_dir}\\train_labels.npy'
test_data_path = f'{data_dir}\\test_data.npy'
test_labels_path = f'{data_dir}\\test_labels.npy'

# Print confirmation message and data shapes
print(f"Training data path:   {Fore.MAGENTA}{train_data_path:>35}{Fore.RESET}")
print(f"Training labels path: {Fore.MAGENTA}{train_labels_path:>35}{Fore.RESET}")
print(f"Test data path:       {Fore.MAGENTA}{test_data_path:>35}{Fore.RESET}")
print(f"Test labels path:     {Fore.MAGENTA}{test_labels_path:>35}{Fore.RESET}")

# clear datasets directory
if os.path.exists('.\\datasets'):  
    print_text("Clearing datasets directory", Fore.RED, 1)
    shutil.rmtree('.\\datasets')

# create datasets directory
os.makedirs('.\\datasets')

# -------------------------------------------------------------------------------
print_header("Downloading and Loading MNIST Dataset")  
# Download and load the MNIST dataset
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Save training data using path variables
np.save(train_data_path, train_data)
np.save(train_labels_path, train_labels)

# Save test data using path variables
np.save(test_data_path, test_data)
np.save(test_labels_path, test_labels)

print_text("MNIST dataset download complete", Fore.GREEN, 1)