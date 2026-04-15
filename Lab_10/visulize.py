import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Record Training History ---
print("Task 1: Loading Data and Preparing Model...")

# 1.1 Load and Preprocess MNIST for the Sequential model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]

# Define a simple model
model = Sequential([
    Flatten(input_shape=(28, 28)), # Flattens 28x28 images to 784 pixels
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 1.2 Train the Model and Record History
print("\nStarting Training (10 Epochs)...")
# validation_split=0.2 tells Keras to use 20% of data to check for overfitting
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)

print("\nHistory keys found:", history.history.keys())

# --- Task 2: Plot Training and Validation Curves ---
print("\nTask 2: Generating Visualization Plots...")

# Create a figure with two subplots: one for Loss and one for Accuracy
plt.figure(figsize=(12, 5))

# Plot 1: Loss Curves
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='#3498db')
plt.plot(history.history['val_loss'], label='Validation Loss', color='#e67e22', linestyle='--')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 2: Accuracy Curves
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#2ecc71')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#e74c3c', linestyle='--')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Save the plots to a file
plt.tight_layout()
plt.savefig('training_metrics.png')
print("✓ Success: Metrics saved as 'training_metrics.png'")

# --- Task 3: Analyze Convergence Trends ---
print("\n--- Task 3: Convergence Analysis ---")
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

if final_val_loss > final_loss + 0.05:
    print("STATUS: Potential Overfitting detected.")
    print("Action: Consider adding Dropout or more data.")
else:
    print("STATUS: Model is converging well and generalizing to validation data.")
