import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt

# Hide GPU/CUDA warnings for a clean terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Define the Residual Block ---

def residual_block(x, filters, kernel_size=3):
    """
    Implements a basic Residual Block: Output = Activation(F(x) + x)
    """
    shortcut = x  # The "Highway" connection
    
    # Layer 1
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Layer 2
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # The Skip Connection: Add the original input back to the processed output
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

# --- Task 2: Build the Models ---

def build_resnet(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)
    # Initial Convolution
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Stack 2 Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="ResNet_Model")

def build_standard_cnn(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="Standard_CNN")

# --- Task 3: Train and Compare ---

print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3.1 Train ResNet
print("\nTraining ResNet...")
resnet = build_resnet()
resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_res = resnet.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# 3.2 Train Standard CNN
print("\nTraining Standard CNN...")
cnn = build_standard_cnn()
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cnn = cnn.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.plot(history_res.history['val_accuracy'], label='ResNet Val Accuracy', linewidth=2)
plt.plot(history_cnn.history['val_accuracy'], label='Standard CNN Val Accuracy', linestyle='--')
plt.title('ResNet vs. Standard CNN Performance')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('resnet_comparison.png')
print("\n✓ Comparison plot saved as 'resnet_comparison.png'")
