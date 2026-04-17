import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Task 2: Define Attention Mechanism ---

class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SimpleAttention, self).__init__()
        self.W_a = tf.keras.layers.Dense(units) # Weight for Encoder states
        self.U_a = tf.keras.layers.Dense(units) # Weight for Decoder hidden state
        self.V_a = tf.keras.layers.Dense(1)     # Final score weight

    def call(self, encoder_states, decoder_hidden):
        # decoder_hidden shape: (batch_size, hidden_size)
        # encoder_states shape: (batch_size, seq_len, hidden_size)
        
        # Expand decoder_hidden to (batch_size, 1, hidden_size) for broadcasting
        decoder_hidden_with_time = tf.expand_dims(decoder_hidden, 1)
        
        # Calculate Alignment Scores
        # score shape: (batch_size, seq_len, 1)
        score = self.V_a(tf.nn.tanh(self.W_a(encoder_states) + self.U_a(decoder_hidden_with_time)))
        
        # Softmax gives Attention Weights (sum to 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Create Context Vector (weighted sum of encoder states)
        context_vector = attention_weights * encoder_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

# --- Task 4: Visualization Function ---

def plot_attention(attention, sentence, translation):
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention, cmap='viridis', annot=True, fmt=".2f",
                xticklabels=sentence.split(),
                yticklabels=translation.split())
    plt.xlabel('Input Sequence (Encoder)')
    plt.ylabel('Output Sequence (Decoder)')
    plt.title('Attention Weights Heatmap')
    plt.show()

# --- Dummy Data & Execution for Testing ---

# Parameters
batch_size = 1
seq_len = 4
hidden_units = 64

# Create dummy encoder states and decoder hidden state
# Example Input: "The black cat" (padded to 4 tokens)
encoder_output = tf.random.normal((batch_size, seq_len, hidden_units))
decoder_hidden = tf.random.normal((batch_size, hidden_units))

# Initialize Attention
attention_layer = SimpleAttention(hidden_units)
context, weights = attention_layer(encoder_output, decoder_hidden)

print("Context Vector Shape:", context.shape) # (1, 64)
print("Attention Weights Shape:", weights.shape) # (1, 4, 1)

# Visualization Example
input_sent = "The black cat <PAD>"
output_word = "Noir" # Example French word for 'Black'
# Reshape weights for plotting: (4, 1)
plot_attention(weights.numpy().reshape(seq_len, 1), input_sent, output_word)
