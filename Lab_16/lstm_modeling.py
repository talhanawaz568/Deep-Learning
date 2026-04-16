import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 2.1: Load and Prepare Dataset ---
print("Task 1: Preparing Time-Series Data...")

# Generate a dummy sine wave dataset (500 points)
data = np.sin(np.linspace(0, 50, 500))

# LSTM expects 3D input: (samples, timesteps, features)
# Here, features = 1 because we only have one value (the sine wave)
n_features = 1
data = data.reshape((len(data), n_features))

# Define sequence length (n_steps) - looking back at 3 points to predict the 4th
n_steps = 3

# TimeseriesGenerator automates the windowing of data
generator = TimeseriesGenerator(data, data, length=n_steps, batch_size=1)

# --- Task 1.2: Define the LSTM Model ---
print("Task 2: Building and Compiling LSTM Model...")

model = Sequential([
    # LSTM layer with 50 units
    # activation='relu' is used here, though 'tanh' is the standard default for LSTMs
    LSTM(units=50, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# --- Task 2.2: Train the LSTM Model ---
print("\nStarting Training (50 Epochs)...")
# We use the generator directly for training
model.fit(generator, epochs=50, verbose=1)

# --- Task 3: Evaluate Sequence Prediction Performance ---
print("\nTask 3: Evaluating and Visualizing Results...")

# Make predictions on the training data to see how well it learned the pattern
predictions = model.predict(generator, verbose=0)

# Step 3.2: Visualize the Results
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual Data (Sine Wave)', color='blue', alpha=0.5)

# We shift predictions by n_steps because the first prediction starts after the first 'n_steps'
plt.plot(range(n_steps, len(predictions) + n_steps), predictions, 
         label='LSTM Predicted Data', color='red', linestyle='--')

plt.title('LSTM Time-Series Prediction Performance')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot for your file explorer
plt.savefig('lstm_results.png')
print("✓ Success: Result plot saved as 'lstm_results.png'")
