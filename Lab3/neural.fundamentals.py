import numpy as np

# --- Task 2.2 & 2.3: Activation Functions ---
# Sigmoid: Squashes values between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU: Returns 0 for negative values, keeps positive values
def relu(x):
    return np.maximum(0, x)

# --- Task 3: The Feedforward Process ---
def feedforward(input_data, weights_input_hidden, weights_hidden_output):
    """
    Simulates the flow of data from input to output.
    """
    # 1. Input Layer to Hidden Layer
    # Dot product of inputs and weights
    hidden_input = np.dot(input_data, weights_input_hidden)
    # Apply activation function to introduce non-linearity
    hidden_output = sigmoid(hidden_input)
    
    # 2. Hidden Layer to Output Layer
    final_input = np.dot(hidden_output, weights_hidden_output)
    # Final squashing for the prediction
    final_output = sigmoid(final_input)
    
    return final_output

# --- Task 1 & 2.1: Defining the Architecture ---
print("Task 1 & 2: Initializing Network Structure...")

# Input features (e.g., 3 sensors or 3 pixels)
# Shape: (1 row, 3 columns)
input_features = np.array([[1.0, 2.0, 3.0]])

# Define layer sizes
input_size = 3
hidden_size = 4
output_size = 2

# Initialize weights randomly
# weights_input_hidden connects 3 inputs to 4 hidden nodes
weights_ih = np.random.rand(input_size, hidden_size)
# weights_hidden_output connects 4 hidden nodes to 2 outputs
weights_ho = np.random.rand(hidden_size, output_size)

print(f"Weights (Input -> Hidden) shape: {weights_ih.shape}")
print(f"Weights (Hidden -> Output) shape: {weights_ho.shape}")

# --- Task 3.2: Implementation ---
print("\nTask 3: Performing Feedforward Pass...")
output = feedforward(input_features, weights_ih, weights_ho)

print("-" * 30)
print("Input Data:       ", input_features)
print("Predicted Output: ", output)
print("-" * 30)
