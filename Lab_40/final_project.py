import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Hide extra logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Dataset Selection & Problem Definition ---
# Problem: Classify 32x32 color images into 10 categories (CIFAR-10)
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Class names for reference
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- Task 2: Data Preprocessing ---
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# --- Task 2.2: Model Construction (CNN) ---
model = models.Sequential([
    # First Convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Dense Classifier
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# --- Task 2.3: Model Training ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nStarting model training...")
# Running for 10 epochs as per lab requirements
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# --- Task 2.4: Model Evaluation ---
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest accuracy achieved: {test_acc*100:.2f}%')

# --- Task 3: Presenting Results (Visualization) ---
def plot_results(history):
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label = 'Test Accuracy', color='orange')
    plt.title('Training vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label = 'Test Loss', color='orange')
    plt.title('Training vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Run the visualization
plot_results(history)

# Save the final model for potential future deployment (Lab 35 style!)
model.save('final_cifar10_model.keras')
print("\n✓ Project complete. Model saved as 'final_cifar10_model.keras'")
