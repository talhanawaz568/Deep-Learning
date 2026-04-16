import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Construct the Autoencoder ---
print("Task 1: Loading and Preprocessing Data...")

# 1.3 Load and Preprocess (We don't need labels 'y' for unsupervised learning)
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the 28x28 images into 784-element vectors
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 1.4 Define Architecture
input_dim = x_train.shape[1]  # 784
encoding_dim = 32             # 32 floats -> compression factor of 24.5

# Input Layer
input_img = Input(shape=(input_dim,))

# "Encoded" is the compressed representation
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "Decoded" is the reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# The Full Autoencoder Model
autoencoder = Model(input_img, decoded)

# --- Task 2: Train the Autoencoder ---
print("\nTask 2: Compiling and Training (50 Epochs)...")

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Notice we pass x_train as both the input AND the target!
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=1)

# --- Task 3: Compare Original vs. Reconstructed ---
print("\nTask 3: Reconstructing and Visualizing Results...")

# Predict (Reconstruct) the test images
decoded_imgs = autoencoder.predict(x_test, verbose=0)

# Visualization
n = 10  # Number of digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    plt.title("Original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display Reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.title("Reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig('autoencoder_results.png')
print("✓ Success: Comparison plot saved as 'autoencoder_results.png'")
