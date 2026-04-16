import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# 1. Define Custom Loss (Manual Mean Squared Error)
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 2. Build Model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss=custom_mse)

# 3. Dummy Data
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)

print("Training with Custom MSE Loss...")
model.fit(X_train, y_train, epochs=5, verbose=1)
print("✓ Successfully trained with custom loss function.")
