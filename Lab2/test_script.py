# test_script.py
import tensorflow as tf
from tensorflow import keras

# Get TensorFlow and Keras version
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Simple test model
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

# Add sample data
import numpy as np
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500)

# Make a prediction
# Convert the list to a NumPy array first
prediction_input = np.array([[10.0]], dtype=float)
print(model.predict(prediction_input))
