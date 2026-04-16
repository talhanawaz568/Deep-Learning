import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import requests

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 1: Prepare and Preprocess Text ---
print("Task 1: Loading and Preprocessing Corpus...")

# Download Alice in Wonderland if it doesn't exist
file_path = 'alice_in_wonderland.txt'
if not os.path.exists(file_path):
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    r = requests.get(url)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(r.text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read().lower()

# Basic cleaning: Replace newlines and extra spaces
text = text.replace('\n', ' ').replace('\r', ' ')
# Limit text length for faster training during the lab
text = text[5000:55000] 

print(f'Text length: {len(text)} characters')

# --- Task 2: Build and Train Character-Level RNN ---

# Mapping chars to indices
chars = sorted(list(set(text)))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}
num_chars = len(chars)
print(f'Unique characters: {num_chars}')

# Create sequences
seq_length = 40
step = 3
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

print(f'Number of sequences: {len(sequences)}')

# Vectorization (One-Hot Encoding)
X = np.zeros((len(sequences), seq_length, num_chars), dtype=bool)
y = np.zeros((len(sequences), num_chars), dtype=bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Build Model
model = Sequential([
    LSTM(128, input_shape=(seq_length, num_chars)),
    Dense(num_chars, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train Model (Using 10 epochs for lab demonstration)
print("\nStarting Training (10 Epochs)...")
model.fit(X, y, batch_size=128, epochs=10)

# --- Task 3: Generate Text ---

def generate_text(seed_text, length=200):
    generated = seed_text
    sentence = seed_text[:seq_length].lower()
    
    print(f"\n--- Seed: '{sentence}' ---")
    
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, num_chars))
        for t, char in enumerate(sentence):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        
        # Diversity/Temperature trick: Instead of just argmax, 
        # we can sample to get more creative results
        next_index = np.argmax(preds)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
        
    return generated

# Select a random seed from the text
start_index = np.random.randint(0, len(text) - seq_length - 1)
seed = text[start_index: start_index + seq_length]

print("\nGenerating Sample Text...")
result = generate_text(seed, 200)
print("\nRESULT:\n", result)
