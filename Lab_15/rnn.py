import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1 & 2: Data Preparation ---
print("Task 1 & 2: Generating and Preparing Sequence Data...")

# Creating a simple sequence: [0,1,2] -> 3, [1,2,3] -> 4, etc.
# We have 100 samples, each with 3 timesteps
X = np.array([[i, i+1, i+2] for i in range(100)])
y = np.array([i+3 for i in range(100)])

# Reshape data into (samples, timesteps, features)
# RNNs REQUIRE this 3D shape. Here: (100, 3, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Task 1.3: Building the RNN Model ---
print("Building Simple RNN Model...")

timesteps = X.shape[1] # 3
features = X.shape[2]  # 1

model = Sequential([
    # SimpleRNN processes the sequence one step at a time
    SimpleRNN(units=50, activation='relu', input_shape=(timesteps, features)),
    Dense(units=1) # Predicting a single numerical value
])

# --- Task 2.2: Training ---
print("\nCompiling and Training (200 Epochs)...")

model.compile(optimizer='adam', loss='mse')

# Training the model
history = model.fit(X, y, epochs=200, batch_size=10, verbose=0)
print("✓ Training complete.")

# --- Task 3: Visualize and Discuss ---
print("\nTask 3: Making Predictions and Visualizing...")

# Predict the sequence based on X
y_pred = model.predict(X, verbose=0)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(y, label='Actual Sequence (Ground Truth)', color='blue', linewidth=2)
plt.plot(y_pred, label='RNN Predicted Sequence', color='red', linestyle='--', linewidth=2)
plt.title('RNN Sequence Prediction Performance')
plt.xlabel('Sample Index')
plt.ylabel('Numerical Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot for terminal viewing
plt.savefig('rnn_predictions.png')
print("✓ Success: Plot saved as 'rnn_predictions.png'")

# --- Task 3.3 Discussion ---
print("\n--- Discussion Points ---")
print("1. Effectiveness: The RNN is highly effective at linear sequences.")
print("2. Memory: The 'SimpleRNN' can struggle with very long sequences due to the 'vanishing gradient' problem.")
print("3. Improvement: For complex patterns, LSTMs or GRUs are usually preferred over SimpleRNN.")
