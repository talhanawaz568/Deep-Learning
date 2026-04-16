import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Suppress TensorFlow logs for a clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1.2: Load and Preprocess Data ---
print("Task 1: Loading and Preprocessing MNIST Data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# --- Task 1.3 & 2.1: Define Model Architectures ---

def create_model(use_dropout=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    
    if use_dropout:
        model.add(Dropout(0.25)) # Dropout after Convolution/Pooling
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    if use_dropout:
        model.add(Dropout(0.5)) # Heavy dropout before final classification
        
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# --- Task 2: Training and Comparison ---

# Train Model WITHOUT Dropout
print("\n--- Training Model WITHOUT Dropout ---")
model_no_drop = create_model(use_dropout=False)
history_no_drop = model_no_drop.fit(x_train, y_train, epochs=10, 
                                    batch_size=128, validation_data=(x_test, y_test), 
                                    verbose=1)

# Train Model WITH Dropout
print("\n--- Training Model WITH Dropout ---")
model_drop = create_model(use_dropout=True)
history_drop = model_drop.fit(x_train, y_train, epochs=10, 
                              batch_size=128, validation_data=(x_test, y_test), 
                              verbose=1)

# --- Task 2.3: Visualize and Compare ---

def plot_results(h1, h2):
    plt.figure(figsize=(14, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(h1.history['accuracy'], 'b--', label='Train (No Dropout)')
    plt.plot(h1.history['val_accuracy'], 'b', label='Test (No Dropout)')
    plt.plot(h2.history['accuracy'], 'r--', label='Train (Dropout)')
    plt.plot(h2.history['val_accuracy'], 'r', label='Test (Dropout)')
    plt.title('Training & Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(h1.history['loss'], 'b--', label='Train (No Dropout)')
    plt.plot(h1.history['val_loss'], 'b', label='Test (No Dropout)')
    plt.plot(h2.history['loss'], 'r--', label='Train (Dropout)')
    plt.plot(h2.history['val_loss'], 'r', label='Test (Dropout)')
    plt.title('Training & Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dropout_comparison.png')
    print("\n✓ Comparison plot saved as 'dropout_comparison.png'")

plot_results(history_no_drop, history_drop)


## pipinstall keras numpy matplotlib pandas 
