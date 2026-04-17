import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
import numpy as np

# Hide GPU/Warnings for a clean run
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Load Pre-Trained Model ---
print("Task 1: Loading VGG16 Base...")
# We use include_top=False to remove the final 1000-class classifier
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# --- Task 2: Modify Architecture ---
# 1. Freeze the base model (don't change the expert's existing knowledge)
for layer in base_model.layers:
    layer.trainable = False

# 2. Add custom layers for our specific task (e.g., 10 classes)
num_classes = 10
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x) # Prevents overfitting
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# --- Task 3: Compile and Prep ---
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("✓ Model ready for Transfer Learning.")
print(f"Total layers: {len(model.layers)}")
print(f"Trainable layers: {len([l for l in model.layers if l.trainable])}")

# Note: To run training, you would use:
# model.fit(train_data, epochs=10, validation_data=test_data)
