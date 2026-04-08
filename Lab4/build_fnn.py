import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Task 1: Create a Sequential Model ---
# Sequential means layers are stacked one after another
model = Sequential()

# --- Task 2: Add Dense Layers with Activation Functions ---

# Input Layer: 4 features (like the Iris dataset)
# 'units=8' means 8 neurons in this first hidden layer
model.add(Dense(units=8, input_shape=(4,), activation='relu'))

# Hidden Layer: 16 neurons
# ReLU is the standard activation for hidden layers
model.add(Dense(units=16, activation='relu'))

# Output Layer: 3 neurons (one for each Iris class)
# 'softmax' ensures the 3 outputs add up to 100% (probability)
model.add(Dense(units=3, activation='softmax'))

# --- Task 3: Compile and Summarize the Model ---

# Optimizer: 'adam' is the best all-around choice for beginners
# Loss: 'categorical_crossentropy' is used for multi-class classification
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Display the architecture
print("\n" + "="*30)
print("MODEL SUMMARY")
print("="*30)
model.summary()

print("\n✓ Neural Network Architecture built and compiled successfully.")
