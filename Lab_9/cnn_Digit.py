import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Suppress TensorFlow startup logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Import MNIST from Keras ---
print("Task 1: Loading MNIST Dataset...")

# Load MNIST dataset - Keras automatically splits it into train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display data shapes
# Result: (60000, 28, 28) means 60k images, each 28x28 pixels
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape:  {test_images.shape}")
print(f"Test labels shape:  {test_labels.shape}")

# --- Task 2: Normalize Image Pixel Values ---
print("\nTask 2: Normalizing Pixels...")

# Images are stored as integers from 0 (black) to 255 (white).
# We convert to float32 and divide by 255 to get a range of [0, 1].
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Verify the normalization
print(f"Max pixel value after normalization: {train_images.max()}")
print(f"Min pixel value after normalization: {train_images.min()}")

# --- Task 3: Visualize Sample Digits ---
print("\nTask 3: Visualizing Samples...")

def visualize_samples(images, labels):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        # We use cmap='gray' because these are single-channel images
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    
    # Save the plot for viewing in your file explorer
    plt.savefig('mnist_samples.png')
    print("✓ Visualization saved as 'mnist_samples.png'")

# Visualize first 10 images with their labels
visualize_samples(train_images, train_labels)

print("\n--- Lab 8 Complete ---")
