import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load and Normalize MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Model WITH Batch Normalization
model_bn = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(), # Normalizes the output of the dense layer
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

model_bn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training Model WITH Batch Normalization...")
history_bn = model_bn.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)

# Summary of result
test_acc = model_bn.evaluate(x_test, y_test, verbose=0)[1]
print(f"\n✓ Test accuracy with BN: {test_acc:.4f}")
