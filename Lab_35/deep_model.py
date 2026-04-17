import os
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Hide GPU logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Train & Save ---
print("Training model for deployment...")
(x_train, y_train), _ = datasets.mnist.load_data()
x_train = x_train / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, verbose=1) # 1 epoch for demo

# Save in the modern Keras v3 format (replaces .h5)
model.save('mnist_deployment_model.keras')
print("✓ Model saved as 'mnist_deployment_model.keras'")

# --- Task 2: Document Architecture ---
model_json = model.to_json()
with open("model_config.json", "w") as f:
    f.write(model_json)
print("✓ Architecture saved to 'model_config.json'")

# --- Task 3: Load & Inference ---
# This part represents what happens on the "Client" or "Server" side
new_model = tf.keras.models.load_model('mnist_deployment_model.keras')
print("✓ Model loaded successfully for inference.")

# Dummy inference (one blank image)
import numpy as np
dummy_data = np.zeros((1, 28, 28))
prediction = new_model.predict(dummy_data, verbose=0)
print(f"Inference Test: Predicted Class {np.argmax(prediction)}")
