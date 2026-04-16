import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# This line hides the CUDA/GPU warnings for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Create Generator and Discriminator ---

def build_generator():
    model = tf.keras.Sequential([
        Input(shape=(100,)),  # FIXED: Proper Keras 3 Input layer
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(28 * 28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        Input(shape=(28, 28, 1)), # FIXED: Proper Keras 3 Input layer
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Initialize models
generator = build_generator()
discriminator = build_discriminator()

# Compile Discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Task 2: Build and Compile GAN ---

# Freeze discriminator weights during GAN training
discriminator.trainable = False

gan_input = Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

print("✓ GAN Architecture Ready (Warnings Cleaned).")

# --- Task 3: Training Prep ---
print("Loading MNIST for training...")
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
# Normalize to [-1, 1] because generator uses 'tanh'
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

print(f"Ready to train on {x_train.shape[0]} images.")
print("To start training, call: train_gan(gan, generator, discriminator, x_train)")
