import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 2: Load and Preprocess Data ---
print("Task 1: Loading CIFAR-10 Dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Step 2.2: Normalization (Scale pixels from 0-255 to 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-Hot Encoding labels for categorical_crossentropy
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- Task 1: Construct CNN Model ---
print("Task 2: Defining CNN Architecture...")

model = Sequential([
    # First Convolutional block: finds simple edges
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    
    # Second Convolutional block: finds more complex patterns
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Transition from 2D features to 1D classification
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax') # 10 neurons for 10 classes
])

# Step 2.3: Compile the Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# --- Task 2.4: Train the Model ---
print("\nStarting Training (10 Epochs)...")
# validation_split takes 10% of training data to monitor for overfitting
history = model.fit(x_train, y_train, epochs=10, batch_size=64, 
                    validation_split=0.1, verbose=1)

# --- Task 3: Evaluate the Model ---
print("\nTask 3: Evaluating Performance on Test Data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

# Visualize the first prediction
prediction = model.predict(x_test[:1], verbose=0)
predicted_class = class_names[np.argmax(prediction)]
actual_class = class_names[np.argmax(y_test[0])]

print(f"\nSample Prediction:")
print(f"Actual: {actual_class} | Predicted: {predicted_class}")

# Save training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.legend()

plt.savefig('cnn_performance.png')
print("\n✓ Success: Performance plot saved as 'cnn_performance.png'")
