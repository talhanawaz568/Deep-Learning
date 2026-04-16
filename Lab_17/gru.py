import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, LSTM

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1.2: Data Preparation ---
print("Task 1: Generating and Preprocessing Data...")

# Creating synthetic time-series data (a complex sine wave)
t = np.linspace(0, 100, 1000)
data_values = np.sin(t) + 0.5 * np.cos(2*t) + np.random.normal(0, 0.1, 1000)
df = pd.DataFrame(data_values, columns=['value'])

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['value']])

# Function to create sequences (Sliding Window)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 50
X, y = create_sequences(scaled_data, sequence_length)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Task 1.3 & 3: Build and Compare Models ---

def build_model(layer_type='GRU'):
    model = Sequential()
    if layer_type == 'GRU':
        model.add(GRU(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(GRU(units=50))
    else:
        model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(LSTM(units=50))
        
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 2. Train the GRU Model
print("\nTraining GRU Model...")
gru_model = build_model('GRU')
gru_history = gru_model.fit(X_train, y_train, epochs=20, batch_size=32, 
                            validation_data=(X_test, y_test), verbose=1)

# (Optional for Task 3) Train an LSTM Model for Comparison
print("\nTraining LSTM Model for Comparison...")
lstm_model = build_model('LSTM')
lstm_history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, 
                              validation_data=(X_test, y_test), verbose=1)

# --- Task 3: Compare Results ---
print("\nTask 3: Evaluation and Visualization...")

plt.figure(figsize=(12, 5))

# Plotting Loss Comparison
plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss', color='blue')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss', color='red', linestyle='--')
plt.title('GRU vs LSTM Performance')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('gru_vs_lstm.png')
print("✓ Comparison chart saved as 'gru_vs_lstm.png'")

# Final Evaluation
gru_loss = gru_model.evaluate(X_test, y_test, verbose=0)
lstm_loss = lstm_model.evaluate(X_test, y_test, verbose=0)

print(f"\nFinal GRU Test Loss:  {gru_loss:.6f}")
print(f"Final LSTM Test Loss: {lstm_loss:.6f}")
