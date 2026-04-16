import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Synthetic Data
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))

# 2. Build Model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Setup Callbacks
# Stop if val_loss doesn't improve for 5 epochs
stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Save the best model weights only
save_best = callbacks.ModelCheckpoint('best_weights.weights.h5', monitor='val_accuracy', save_best_only=True)

print("Training with Early Stopping and Checkpointing...")
model.fit(X_train, y_train, epochs=50, validation_split=0.2, 
          callbacks=[stop_early, save_best], verbose=1)

print("✓ Training finished. Best weights saved to 'best_weights.weights.h5'")
