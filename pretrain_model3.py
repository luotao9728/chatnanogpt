import torch
import random
import numpy as np
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
import json
import re
import pandas as pd
from tqdm import tqdm
import Levenshtein
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config = {
    "epochs": 2,
    "lr": 1e-4,
    "seq_len": 500,
    "batch_size": 20,
    "hidden_size": 512,
    "encoder_nhead": 4,
    "decoder_nhead": 4,
    "cross_nhead": 4
}

# Load dictionaries and data
stoi_comm = json.load(open('stoi_comm.json', 'r'))
itos_comm = json.load(open('itos_comm.json', 'r'))
text = np.load('test.npy', allow_pickle=True)

vocab_size = len(stoi_comm)
print('Vocab size:', vocab_size)
print('Number of sequences:', len(text))

# Split dataset
def split_train_valid_test(arr, proportion1, proportion2):
    # Ensure proportions are valid
    assert 0 <= proportion1 <= 1, "Proportion1 must be between 0 and 1"
    assert 0 <= proportion2 <= 1, "Proportion2 must be between 0 and 1"
    assert proportion1 + proportion2 <= 1, "The sum of proportions must not exceed 1"

    # Shuffle the array along axis=0
    shuffled_arr = np.random.permutation(arr)

    # Calculate split indices
    split_index1 = int(proportion1 * arr.shape[0])
    split_index2 = split_index1 + int(proportion2 * arr.shape[0])

    # Split the array
    return np.split(shuffled_arr, [split_index1, split_index2])

train_data, val_data, test_data = split_train_valid_test(text, 0.9, 0.05) 

# Print the results
print("Train:", len(train_data))
print("Validate:", len(val_data))
print("Test:", len(test_data))

# Define dataloader
class DataLoaderForLanguageModeling(torch.utils.data.DataLoader): # Inherit from torch.utils.data.DataLoader

    def __init__(self, dataset, seq_len, batch_size, num_workers, shuffle=True, drop_last=False):

        self.dataset     = dataset
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.drop_last   = drop_last
        self.num_workers = num_workers

    def __len__(self):
        return np.concatenate(self.dataset).shape[0] // (self.seq_len * self.batch_size) - 2

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.dataset)

        # Concatenate all data
        dataset_concat = np.concatenate(self.dataset)
        num_batches = dataset_concat.shape[0] // (self.seq_len * self.batch_size)

        batch_idx = 0

        if self.drop_last:
            num_batches -= 1

        residual = dataset_concat.shape[0] % (num_batches * self.seq_len)
        leftover_residual = residual % self.seq_len - 1
        leftover_x = dataset_concat[-residual:-leftover_residual-1].reshape(residual // self.seq_len, self.seq_len)
        leftover_y = dataset_concat[-residual+1:-leftover_residual].reshape(residual // self.seq_len, self.seq_len)
        dataset_concat = dataset_concat[:-residual].reshape(self.batch_size, -1)

        while batch_idx < num_batches - 1:
            start_idx = batch_idx * self.seq_len

            if batch_idx == num_batches - 1 and not self.drop_last:
                yield torch.tensor(leftover_x), torch.tensor(leftover_y)
            else:
                end_idx = start_idx + self.seq_len
                x_out = dataset_concat[:,start_idx:end_idx]
                y_out = dataset_concat[:,start_idx+1:end_idx+1]
                yield torch.tensor(x_out, dtype=torch.long), torch.tensor(y_out, dtype=torch.long)

            batch_idx += 1

# Initialize dataloaders
train_loader = DataLoaderForLanguageModeling(
    dataset     = train_data,
    seq_len     = config["seq_len"],
    batch_size  = config["batch_size"],
    shuffle     = True,
    drop_last   = True,
    num_workers = 16
)

val_loader = DataLoaderForLanguageModeling(
    dataset     = val_data[:5000],
    seq_len     = config["seq_len"],
    batch_size  = config["batch_size"],
    shuffle     = False,
    drop_last   = True,
    num_workers = 16
)

test_loader = DataLoaderForLanguageModeling(
    dataset     = test_data[:500],
    seq_len     = config["seq_len"],
    batch_size  = config["batch_size"],
    shuffle     = False,
    drop_last   = True,
    num_workers = 16
)

inputs, targets = next(iter(train_loader))
print(inputs.shape, targets.shape)

# Tokenizer
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Define encode and decode functions
def encode(s, stoi):
    encoded = []
    for word in tokenize(s):
        UNKNOWN_TOKEN = len(stoi) - 1
        try:
            encoded.append(int(stoi[word]))
        except KeyError:
            encoded.append(UNKNOWN_TOKEN)
    return encoded

def decode(l, itos):
    decoded = []
    for i in l:
        try:
            decoded.append(itos[str(i)])
        except KeyError:
            continue
    return decoded

for x, y in train_loader:
    print("x: ", ' '.join(decode(np.array(x[0, :]), itos_comm)))
    print("y: ", ' '.join(decode(np.array(y[0, :]), itos_comm)))
    break

import math

class PositionalEncoding(torch.nn.Module):

    def __init__(self, projection_size, max_seq_len=176):
        super().__init__()

        self.projection_size = projection_size
        self.max_seq_len = max_seq_len

        pos = torch.arange(0, self.max_seq_len).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, self.projection_size, 2) * - math.log(10000.0) / self.projection_size)
        
        P = torch.zeros(self.max_seq_len, self.projection_size)
        P[:,0::2] = torch.sin(pos * denominator)
        P[:,1::2] = torch.cos(pos * denominator)
        self.P = P

        self.register_buffer('encoding_matrix', self.P)

    def forward(self, x):       
        return x + self.P[:x.size(1)].to(device)

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        attn = torch.nn.Softmax(dim=-1)(scores)
        output = torch.matmul(attn, value)
        return output, attn

class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.linear_q = torch.nn.Linear(embed_size, embed_size)
        self.linear_k = torch.nn.Linear(embed_size, embed_size)
        self.linear_v = torch.nn.Linear(embed_size, embed_size)
        self.linear_out = torch.nn.Linear(embed_size, embed_size)

        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projection and split into heads
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        attn, _ = self.attention(query, key, value)

        # Concatenate heads and apply final linear projection
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        output = self.linear_out(attn)

        return output

class TransformerEncoder(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        # create the key, query and value weights
        self.W = torch.nn.Linear(hidden_size, hidden_size)
        # Compute multihead attention. You are free to use the version provided by pytorch
        self.attention  = MultiheadAttention(hidden_size, num_heads)
        self.bn1        = torch.nn.LayerNorm(hidden_size)
        self.bn2        = torch.nn.LayerNorm(hidden_size)

        # Feed forward neural network
        self.MLP        = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        
    def forward(self, x):
        # compute the key, query and value
        x = self.W(x)
        # compute the output of the attention module
        out1    = self.attention(x, x, x)
        # Create a residual connection between the input and the output of the attention module
        out1    = out1 + x
        out1 = self.bn1(out1)
        # Apply the output of the feed forward network
        out2    = self.MLP(out1)
        # Apply a residual connection between the input and output of the  FFN
        out2 = out2 + out1
        # Apply batch norm to the output
        out2    = self.bn2(out2)

        return out2

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, encoder_hidden_size, tf_block, nhead):
        super(Encoder, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, encoder_hidden_size)
        self.multihead_attn = MultiheadAttention(encoder_hidden_size, nhead)
        self.positional_encoding = PositionalEncoding(encoder_hidden_size, 500)
        self.layer_norm = torch.nn.LayerNorm(encoder_hidden_size)
        self.transformer_encoder    = torch.nn.Sequential()
        for _ in range(tf_block):
            self.transformer_encoder.append(TransformerEncoder(encoder_hidden_size, nhead))

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Shape: [seq_len, batch_size, features]
        # Positional Encoding
        x = self.positional_encoding(x)
        # Apply Multihead Attention with Residual Connection
        attn_output = self.multihead_attn(x, x, x)
        x = x + attn_output  # Adding residual connection

        return x

class Decoder(torch.nn.Module):
    def __init__(self, decoder_hidden_size, vocab_size, nhead):
        super(Decoder, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, decoder_hidden_size)
        self.multihead_attn = MultiheadAttention(decoder_hidden_size, nhead)
        self.layer_norm = torch.nn.LayerNorm(decoder_hidden_size)

    def forward(self, x, y=None, tf_rate=1.0):
        p = torch.rand(1).item()
        if p < tf_rate and y is not None:
            x = self.embedding(y)
            x = x.transpose(0, 1)
        # Apply Multihead Attention with Residual Connection
        attn_output = self.multihead_attn(x, x, x)
        x = x + attn_output  # Adding residual connection
        # Layer Normalization
        x = self.layer_norm(x)
        return x.transpose(0, 1)

class NanoGPT(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, tf_block, encoder_nhead, decoer_nhead, cross_nhead):
        super(NanoGPT, self).__init__()
        
        self.encoder = Encoder(vocab_size, hidden_size, tf_block, encoder_nhead)
        self.decoder = Decoder(hidden_size, vocab_size, decoer_nhead)
        self.enc_dec_attn = MultiheadAttention(hidden_size, cross_nhead)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.CDN = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, vocab_size)
        )
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, y=None, tf_rate=1.0):
        # Encoding
        encoder_output = self.encoder(x)
        # Decoding
        decoder_output = self.decoder(encoder_output, y, tf_rate)
        # Transpose for attention
        encoder_output = encoder_output.transpose(0, 1)  # Shape: [seq_len, batch_size, features]
        # decoder_output = decoder_output.transpose(0, 1)  # Shape: [seq_len, batch_size, features]
        # Attention between encoder and decoder
        attn_output = self.enc_dec_attn(decoder_output, encoder_output, encoder_output)
        # Adding residual connection and layer normalization
        attn_output = self.layer_norm(attn_output + decoder_output)     # changed from encoder_output
        # Convert to probability distribution
        output = self.CDN(attn_output)
        output = self.softmax(output)

        return output

    def predict(self, x):
        with torch.no_grad():  # Inference mode
            output = self.forward(x)
            output = output[-1]  # Get the last output
            probabilities = F.softmax(output, dim=-1)
            _, predicted = torch.max(probabilities, dim=-1)
            return predicted

    def generate(self, x, max_length):
        with torch.no_grad():  # Inference mode
            generated_seq = x
            for _ in range(max_length):
                output = self.forward(generated_seq)
                output = output[-1]  # Get the last output
                probabilities = F.softmax(output, dim=-1)
                _, next_token = torch.max(probabilities, dim=-1)
                generated_seq = torch.cat((generated_seq, next_token.unsqueeze(0)), dim=0)
                if next_token.item() == self.end_token_id:  # Assuming you have an end token id
                    break
            return generated_seq

model = NanoGPT(
    vocab_size=vocab_size,
    hidden_size=config['hidden_size'],
    tf_block=config['tf_block'],
    encoder_nhead=config['encoder_nhead'],
    decoer_nhead=config['decoder_nhead'],
    cross_nhead=config['cross_nhead']
).to(device)

print(model)
summary(model, x.to(device))

optimizer   = torch.optim.AdamW(model.parameters(), lr=config["lr"])
criterion   = torch.nn.CrossEntropyLoss()
scaler      = torch.cuda.amp.GradScaler()
scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                         factor=0.5, 
                                                         patience=1,
                                                         threshold=0.4,
                                                         cooldown=1)

def train(model, dataloader, criterion, optimizer, tf_rate=1.0):

    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    running_loss        = 0.0
    running_perplexity  = 0.0

    for i, (x, y) in enumerate(dataloader):

        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            raw_predictions = model(x, y, tf_rate)
            loss        = criterion(raw_predictions.transpose(1, 2), y)

        perplexity  = torch.exp(loss) # Perplexity is defined the exponential of the loss
        running_loss        += loss.item()
        running_perplexity  += perplexity.item()

        # Backward on the masked loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(i+1)),
            perplexity="{:.04f}".format(running_perplexity/(i+1)),
            lr="{:.08f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()

        del x, y
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)
    batch_bar.close()

    return running_loss, running_perplexity

def decode(l, itos):
    decoded = []
    for i in l:
        try:
            decoded.append(itos[str(i)])
        except KeyError:
            continue
    return decoded

def calc_edit_distance(predictions, y, itos_comm, print_example=False):

    dist                = 0
    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size):
        y_sliced    = decode(np.array(y[batch_idx]), itos_comm)
        pred_sliced = decode(np.array(predictions[batch_idx]), itos_comm)

        # Strings - When you are using characters from the AudioDataset
        y_string    = ' '.join(y_sliced)
        pred_string = ' '.join(pred_sliced)

        dist        += Levenshtein.distance(pred_string, y_string)

    if print_example:
        # Print y_sliced and pred_sliced if you are using the toy dataset
        print("\nGround Truth : ", y_string)
        print("Prediction   : ", pred_string)

    dist    /= batch_size
    return dist

def validate(model, dataloader):

    model.eval()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    running_lev_dist = 0.0

    for i, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            raw_predictions = model(x)

        # Greedy Decoding
        predictions   =  torch.argmax(raw_predictions, dim=-1)

        # Calculate Levenshtein Distance
        predictions, y = predictions.to('cpu'), y.to('cpu')
        running_lev_dist    += calc_edit_distance(predictions, y, itos_comm, print_example = False)

        batch_bar.set_postfix(
            dist="{:.04f}".format(running_lev_dist/(i+1)))
        batch_bar.update()

        del x, y
        torch.cuda.empty_cache()

    batch_bar.close()
    running_lev_dist /= len(dataloader)

    return running_lev_dist

best_lev_dist = float("inf")
tf_rate = 0.1

for epoch in range(0, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))
    curr_lr = optimizer.param_groups[0]['lr']

    # Call train and validate, get attention weights from training
    running_loss, running_perplexity = train(model, train_loader, criterion, optimizer, tf_rate)
    
    valid_dist = validate(model, val_loader)

    # Print your metrics
    print("\nEpoch {}/{}: \nPerplexity {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.08f}\t Validate Distance {:.04f}".format(
        epoch + 1,
        config['epochs'],
        running_perplexity,
        running_loss,
        curr_lr,
        valid_dist))

    if valid_dist <= best_lev_dist:
        best_lev_dist = valid_dist
        torch.save(model.state_dict(), 'pretrain_model3.pth')

def test(model, dataloader):

    predicted = []
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Test")

    for i, (x, _) in enumerate(dataloader):

        x = x.to(device)

        with torch.inference_mode():
            raw_predictions = model(x)

        raw_predictions = raw_predictions.argmax(dim=-1)
        batch_size, _ = raw_predictions.shape
        raw_predictions = raw_predictions.to('cpu')

        curr_pred = []
        for batch_idx in range(batch_size):
            pred_sliced = decode(np.array(raw_predictions[batch_idx]), itos_comm)
            pred_string = ' '.join(pred_sliced)
            curr_pred.append(pred_string)

        predicted.append(curr_pred)

        del x
        torch.cuda.empty_cache()
        batch_bar.update()

    batch_bar.close()

    return predicted

prediction = test(model, test_loader)
for i in range(5):
    print(prediction[i])
