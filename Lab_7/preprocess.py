import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# --- Task 1: Normalize and Scale Sample Data ---

# 1.2 Normalizing Data
# Goal: Scale features to a fixed range, typically [0, 1].
print("Task 1.2: Normalizing Data (Min-Max Scaling)")

data = np.array([[10, 2.7, 3.6],
                 [-100, 5, -2],
                 [120, 20, 40]], dtype=np.float64)

# MinMaxScaler: (x - min) / (max - min)
min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data)

print("Original Data:\n", data)
print("Normalized Data (Range 0-1):\n", normalized_data)
print("-" * 30)

# 1.3 Standardizing Data
# Goal: Scale data to have mean=0 and standard deviation=1.
print("Task 1.3: Standardizing Data (Z-score Scaling)")

# StandardScaler: (x - mean) / std_dev
std_scaler = StandardScaler()
standardized_data = std_scaler.fit_transform(data)

print("Standardized Data (Mean 0, Std 1):\n", standardized_data)
print("-" * 30)


# --- Task 2: Split Data into Training and Testing Sets ---

# 2.2 Implementing Data Split
# Goal: Create a 'Hold-out' set to verify the model on unseen data.
print("Task 2: Splitting Data")

# Create a synthetic dataset (100 samples, 5 features)
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100) # Binary target (0 or 1)

# Split Ratio: 70% Training, 30% Testing
# random_state ensures the split is reproducible every time you run the script.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Total Samples: {X.shape[0]}")
print(f"Training Features Shape: {X_train.shape} (70%)")
print(f"Testing Features Shape:  {X_test.shape} (30%)")


# --- Task 3: Documentation Summary ---
"""
DOCUMENTATION SUMMARY:
1. Normalization: Used when the distribution of data is unknown or 
   when the algorithm does not assume a Gaussian distribution (like KNN or NNs).
2. Standardization: Useful when data follows a Gaussian (Normal) distribution. 
   It is less sensitive to outliers than Min-Max scaling.
3. Train/Test Split: Prevents overfitting by ensuring the model 
   is evaluated on data it has never seen during the training phase.
"""

print("\n✓ Lab 7 Complete: Data is cleaned, scaled, and split.")
