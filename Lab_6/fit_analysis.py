import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Train a Model on a Dataset ---
print("Task 1: Loading California Housing Data...")

# Step 1.1: Load and Prepare the Dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1.2: Train a Simple Linear Regression Model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# --- Task 2: Plot Training and Validation Losses ---
print("\nTask 2: Calculating and Visualizing Losses...")

# Step 2.1: Make Predictions and Calculate Mean Squared Error
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

train_loss = mean_squared_error(y_train, train_predictions)
val_loss = mean_squared_error(y_val, val_predictions)

print(f"Training Loss (MSE):   {train_loss:.4f}")
print(f"Validation Loss (MSE): {val_loss:.4f}")

# Step 2.2: Visualize the Losses
plt.figure(figsize=(8, 6))
losses = [train_loss, val_loss]
labels = ['Training Loss', 'Validation Loss']

plt.bar(labels, losses, color=['#3498db', '#e67e22'])
plt.title('Training vs Validation Loss (Linear Regression)')
plt.ylabel('Mean Squared Error')
plt.savefig('loss_comparison.png')
print("✓ Chart saved as 'loss_comparison.png'")

# --- Task 3: Addressing Overfitting with Neural Networks (Dropout) ---
print("\nTask 3: Demonstrating Dropout Architecture...")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    # This model uses Dropout to prevent the network from becoming 
    # overly dependent on specific neurons (reducing overfitting).
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5), # 50% of neurons are randomly disabled during each training step
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1) # Final price prediction
    ])

    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    print("✓ Neural Network with Dropout layers initialized successfully.")
except ImportError:
    print("TensorFlow not detected. Skipping Task 3.2 neural network setup.")

print("\n--- Lab Conclusion ---")
print("Check 'loss_comparison.png' to see if your model generalizes well.")
