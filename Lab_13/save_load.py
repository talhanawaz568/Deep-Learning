import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Input

# This hides the CUDA/GPU warnings so your output is clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Create and Save ---
print("Task 1: Creating and Saving Model Architecture & Weights...")

# UPDATED: Using Input layer as recommended by the warning
model = Sequential([
    Input(shape=(784,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Save Architecture
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)
print("✓ Architecture saved to 'model_architecture.json'")

# FIXED: Changed extension to .weights.h5 to satisfy Keras 3
model.save_weights('model_weights.weights.h5')
print("✓ Weights saved to 'model_weights.weights.h5'")


# --- Task 2: Reload ---
print("\nTask 2: Reloading Model...")

with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# FIXED: Must match the new filename used above
loaded_model.load_weights('model_weights.weights.h5')
print("✓ Model and Weights reloaded successfully.")


# --- Task 3: Verify ---
example_data = np.random.random((1, 784))
original_pred = model.predict(example_data, verbose=0)
loaded_pred = loaded_model.predict(example_data, verbose=0)

if np.allclose(original_pred, loaded_pred):
    print("\nSUCCESS: Original and Loaded predictions match!")
else:
    print("\nERROR: Predictions do not match.")
