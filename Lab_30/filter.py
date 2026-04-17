import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Hide GPU/CUDA warnings for a clean terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Extract Weights ---
print("Task 1: Loading Pre-trained VGG16 Model...")
# We load the model trained on 'imagenet' to see meaningful filters
model = VGG16(weights='imagenet', include_top=False)

# Identify the first convolutional layer
layer_name = 'block1_conv1'
layer = model.get_layer(name=layer_name)

# Extract weights (filters) and biases
# filters shape: (width, height, channels, num_filters)
# For VGG16 block1_conv1: (3, 3, 3, 64)
weights, biases = layer.get_weights()
filters = weights

# --- Task 2: Plot Filter Patterns ---
print("Task 2: Normalizing and Plotting Filters...")

# Normalize filter values to [0, 1] so they display correctly as images
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# We will plot the first 6 filters
n_filters = 6
index = 1

plt.figure(figsize=(10, 12))
for i in range(n_filters):
    # Get the i-th filter
    f = filters[:, :, :, i]
    
    # Each filter has 3 channels (Red, Green, Blue)
    for j in range(3):
        ax = plt.subplot(n_filters, 3, index)
        ax.set_xticks([])
        ax.set_yticks([])
        # Show each channel's contribution to the filter
        plt.imshow(f[:, :, j], cmap='gray')
        if i == 0:
            plt.title(f"Channel {j+1}")
        if j == 0:
            plt.ylabel(f"Filter {i+1}", rotation=0, labelpad=40, fontsize=10)
        index += 1

plt.tight_layout()
plt.suptitle(f"Visualizing Filters from '{layer_name}'", fontsize=16, y=1.02)
plt.savefig('cnn_filters.png')
print("✓ Success: Filter visualization saved as 'cnn_filters.png'")
