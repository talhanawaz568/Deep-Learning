import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Hide GPU/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Prepare Model and Image ---
print("Task 1: Loading VGG16 and Image...")
model = VGG16(weights='imagenet', include_top=True)

# Note: Using a placeholder image if 'elephant.jpg' isn't in your directory
# You can replace this with any image path on your Ubuntu system
img_path = 'elephant.jpg' 

# If file doesn't exist, we'll download a sample image automatically
if not os.path.exists(img_path):
    import requests
    url = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Elephant_f_600.jpg"
    img_data = requests.get(url).content
    with open(img_path, 'wb') as f:
        f.write(img_data)

img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)

# --- Task 2: Build Activation Model ---
print("Task 2: Extracting Intermediate Layers...")

# We'll take the first 8 layers (convolutional and pooling layers)
layer_outputs = [layer.output for layer in model.layers[:8]]
layer_names = [layer.name for layer in model.layers[:8]]

# This model returns the activations of the intermediate layers
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor, verbose=0)

# --- Task 3: Plot and Analyze ---
print("Task 3: Visualizing Feature Maps...")

# Plotting the activations of the first few layers
# We will look at the 1st channel of each of the first 8 layers
fig, axes = plt.subplots(1, 8, figsize=(25, 5))

for i, (act, name) in enumerate(zip(activations, layer_names)):
    # act shape is (1, width, height, num_filters)
    # We display the first channel [0, :, :, 0]
    axes[i].matshow(act[0, :, :, 0], cmap='viridis')
    axes[i].set_title(name, fontsize=10)
    axes[i].axis('off')

plt.suptitle("How VGG16 'Sees' the Image through Intermediate Layers", fontsize=16)
plt.savefig('activations_flow.png')
print("✓ Success: Activations saved as 'activations_flow.png'")

# Display a single detailed map from the first layer
plt.figure(figsize=(5,5))
plt.matshow(activations[0][0, :, :, 15], cmap='viridis') # Looking at the 16th channel
plt.title(f"Layer: {layer_names[0]} (Detailed Channel 15)")
plt.savefig('first_layer_detail.png')
print("✓ Success: Detailed map saved as 'first_layer_detail.png'")
