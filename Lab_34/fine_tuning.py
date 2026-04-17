import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, Input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Unfreeze Select Layers ---
# Create a base model (usually one you've already added a head to)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze everything first
base_model.trainable = False

# Unfreeze the last block of VGG16 (Block 5)
# This allows the model to adjust its high-level feature detection
for layer in base_model.layers:
    if "block5" in layer.name:
        layer.trainable = True

# --- Task 2: Re-build and Compile ---
inputs = Input(shape=(32, 32, 3))
x = base_model(inputs, training=True) # training=True is important for BatchNormalization
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)

# Use a very SMALL learning rate for fine-tuning
# We don't want to "wreck" the pre-trained weights with big updates
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("✓ Model ready for Fine-Tuning.")
for i, layer in enumerate(base_model.layers):
    print(f"Layer {i}: {layer.name} | Trainable: {layer.trainable}")

# --- Task 3: Retrain on Small Dataset ---
# (Using CIFAR-10 as a placeholder)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("\nStarting Fine-Tuning...")
# Training usually takes fewer epochs for fine-tuning
model.fit(x_train[:5000], y_train[:5000], epochs=3, batch_size=32, validation_split=0.1)
