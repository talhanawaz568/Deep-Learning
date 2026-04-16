import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def train_and_evaluate(lr, bs):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=2, batch_size=bs, verbose=0)
    return model.evaluate(x_test, y_test, verbose=0)[1]

# Grid Search
lrs = [0.01, 0.001]
bss = [32, 64]

for lr in lrs:
    for bs in bss:
        acc = train_and_evaluate(lr, bs)
        print(f"LR: {lr} | Batch: {bs} | Accuracy: {acc:.4f}")
