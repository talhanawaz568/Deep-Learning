import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, utils

# Data Prep
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = utils.to_categorical(y_train), utils.to_categorical(y_test)

def build_simple_model():
    return models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

# Compare Optimizers
results = {}
for opt in ['sgd', 'adam', 'rmsprop']:
    print(f"Training with {opt} optimizer...")
    model = build_simple_model()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    h = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=0)
    results[opt] = h.history['accuracy']

# Plotting
for opt, acc in results.items():
    plt.plot(acc, label=opt)
plt.title('Optimizer Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('optimizer_lab.png')
print("✓ Comparison plot saved to 'optimizer_lab.png'")
