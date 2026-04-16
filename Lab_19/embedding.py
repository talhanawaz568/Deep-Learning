import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Task 2: Prepare the Text Data ---
print("Task 1: Preparing Text Sequences...")

# Larger sample dataset for better visualization
texts = [
    "I love this product", "This is excellent", "Very good quality",
    "I hate this", "This is terrible", "Poor quality and bad",
    "Amazing experience", "Wonderful and great", "Disgusting and awful",
    "Not worth the money", "Best purchase ever", "I am very unhappy"
]
# 1 = Positive, 0 = Negative
y_data = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0])

vocab_size = 100
embedding_dim = 8  # Small dimension for small data
max_length = 5

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences so they are all the same length
X_data = pad_sequences(sequences, maxlen=max_length)

# --- Task 1: Create Model with Embedding Layer ---
print("Task 2: Building Neural Network...")

model = Sequential([
    # Input: Vocab index -> Output: Dense vector of size 8
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=(max_length,)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
print("\nTraining Model...")
model.fit(X_data, y_data, epochs=50, batch_size=2, verbose=0)
print("✓ Training complete.")

# --- Task 3: Visualize the Embedding Space ---
print("\nTask 3: Extracting and Visualizing Embeddings...")

# 3.1 Extract Weights from the Embedding layer (Layer 0)
embeddings = model.layers[0].get_weights()[0]

# 3.2 t-SNE Dimensionality Reduction
# t-SNE reduces our 8D vectors down to 2D so we can plot them on a screen
tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate='auto')
# We only visualize the words that were actually in our 'texts'
relevant_vocab_size = len(tokenizer.word_index) + 1
reduced_embeddings = tsne.fit_transform(embeddings[1:relevant_vocab_size])

# Plotting
plt.figure(figsize=(10, 8))
for i, word in enumerate(tokenizer.word_index.keys()):
    x, y = reduced_embeddings[i, 0], reduced_embeddings[i, 1]
    plt.scatter(x, y)
    plt.annotate(word, (x, y), alpha=0.7)

plt.title('t-SNE Visualization of Learned Word Embeddings')
plt.grid(True, alpha=0.3)
plt.savefig('embeddings_plot.png')
print("✓ Success: Embedding plot saved as 'embeddings_plot.png'")
