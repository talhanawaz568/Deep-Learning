import os
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import numpy as np
import matplotlib.pyplot as plt

# Hide GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Construct Networks ---
input_dim = 784
latent_dim = 2

# Encoder
inputs = layers.Input(shape=(input_dim,))
h = layers.Dense(512, activation='relu')(inputs)
h = layers.Dense(256, activation='relu')(h)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(latent_inputs)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(input_dim, activation='sigmoid')(x)
decoder = models.Model(latent_inputs, outputs, name='decoder')

# VAE Model
vae_outputs = decoder(encoder(inputs)[2])
vae = models.Model(inputs, vae_outputs, name='vae')

# --- Task 2: Custom Loss & Training ---
reconstruction_loss = losses.mse(inputs, vae_outputs) * input_dim
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
vae.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
vae.compile(optimizer='adam')

# Load MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, input_dim).astype('float32') / 255.0
x_test = x_test.reshape(-1, input_dim).astype('float32') / 255.0

print("Training VAE...")
vae.fit(x_train, epochs=10, batch_size=128, validation_data=(x_test, None), verbose=1)

# --- Task 3: Visualize Latent Space ---
n = 10
figure = np.zeros((28 * n, 28 * n))
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        digit = decoder.predict(z_sample, verbose=0)[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

plt.imshow(figure, cmap='Greys_r')
plt.savefig('vae_generated.png')
print("✓ VAE output saved as 'vae_generated.png'")
