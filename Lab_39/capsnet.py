import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils
import numpy as np
import matplotlib.pyplot as plt

# --- Task 1: Construct the Capsule Network ---

def squash(vectors, axis=-1):
    """
    The non-linear activation function for capsules.
    Squashes the length of vectors to between 0 and 1.
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-7)
    return scale * vectors

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings

    def build(self, input_shape):
        # Weight matrix to transform input capsules to output capsules
        self.kernel = self.add_weight(name='capsule_kernel',
                                     shape=(self.num_capsules, input_shape[1], 
                                            self.dim_capsules, input_shape[2]),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, inputs):
        # inputs shape: (None, input_num_capsules, input_dim_capsules)
        # We perform a transformation to get predicted vectors (u_hat)
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        u_hat = tf.matmul(self.kernel, inputs_expanded)
        u_hat = tf.squeeze(u_hat, [ -1])
        
        # In a full CapsNet, we would perform Dynamic Routing here.
        # For this minimal lab, we take the mean and squash it.
        v_j = tf.reduce_mean(u_hat, axis=2)
        return squash(v_j)

def create_capsnet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # 1. Standard Convolution to detect basic features
    x = layers.Conv2D(64, 9, activation='relu')(inputs)
    
    # 2. Primary Capsules: Group features into vectors
    x = layers.Conv2D(64, 9, strides=2, activation='relu')(x)
    x = layers.Reshape((-1, 8))(x) # 8D vectors
    
    # 3. Digit Capsules: One capsule per class (0-9)
    capsule = CapsuleLayer(num_capsules=10, dim_capsules=16)(x)
    
    # 4. Output: Length of the capsule vector represents probability
    outputs = layers.Lambda(lambda z: tf.sqrt(tf.reduce_sum(tf.square(z), axis=-1)))(capsule)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CapsNet')
    return model

# --- Task 2: Data Preparation & Training ---

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Preprocess
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# Categorical labels
train_labels_cat = utils.to_categorical(train_labels, 10)
test_labels_cat = utils.to_categorical(test_labels, 10)

capsnet = create_capsnet((28, 28, 1))
capsnet.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

print("Starting training...")
capsnet.fit(train_images[:10000], train_labels_cat[:10000], # Subset for speed
            batch_size=128, epochs=3, validation_split=0.1)

# --- Task 3: Visualization ---

def plot_examples(images, labels, predictions):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        true_label = labels[i]
        predicted_label = np.argmax(predictions[i])
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}')
        axes[i].axis('off')
    plt.show()

predictions = capsnet.predict(test_images[:5])
plot_examples(test_images[:5], test_labels[:5], predictions)
