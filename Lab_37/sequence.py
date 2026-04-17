import torch
import torch.nn as nn
import torch.optim as optim
import random

# --- Constants & Helpers ---
SOS_token = 0  # Start of Sentence
EOS_token = 1  # End of Sentence
MAX_LENGTH = 5

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

# --- Task 1: Construct Encoder & Decoder ---

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# --- Task 2: Training Logic ---

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
          decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0

    # Encoder loop
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # Decoder starts with SOS_token and encoder's last hidden state
    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden

    # Decoder loop
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# --- Task 3: Evaluation ---

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_ids = [input_lang.word2index[w] for w in sentence.split(' ')]
        input_ids.append(EOS_token)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).view(-1, 1)
        
        encoder_hidden = encoder.initHidden()
        for ei in range(input_tensor.size(0)):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words

# --- Main Execution ---

# 1. Prepare small dummy dataset
input_lang = Lang("digits")
output_lang = Lang("words")
pairs = [("1", "One"), ("2", "Two"), ("3", "Three"), ("4", "Four"), ("5", "Five")]

for s1, s2 in pairs:
    input_lang.addSentence(s1)
    output_lang.addSentence(s2)

# 2. Initialize Models
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = DecoderRNN(hidden_size, output_lang.n_words)

# 3. Training Loop
encoder_opt = optim.SGD(encoder.parameters(), lr=0.01)
decoder_opt = optim.SGD(decoder.parameters(), lr=0.01)
criterion = nn.NLLLoss()

print("Training Seq2Seq Model...")
for epoch in range(500):
    pair = random.choice(pairs)
    input_tensor = torch.tensor([input_lang.word2index[pair[0]], EOS_token]).view(-1, 1)
    target_tensor = torch.tensor([output_lang.word2index[pair[1]], EOS_token]).view(-1, 1)
    
    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_opt, decoder_opt, criterion)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 4. Final Evaluation
print("\nEvaluation:")
for test_val in ["1", "3", "5"]:
    result = evaluate(encoder, decoder, test_val, input_lang, output_lang)
    print(f"Input: {test_val} -> Output: {' '.join(result)}")
