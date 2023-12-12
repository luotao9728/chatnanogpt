# Import packages
import torch
import numpy as np
import json
import re
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
config = {
    "hidden_size": 1024,
    "tf_block": 2,
    "encoder_nhead": 8,
    "decoder_nhead": 8,
    "cross_nhead": 8
}

# Load dictionaries and finetune data
stoi_comm = json.load(open('stoi_comm.json', 'r'))
itos_comm = json.load(open('itos_comm.json', 'r'))

# Create model
import math

class PositionalEncoding(torch.nn.Module):

    def __init__(self, projection_size, max_seq_len=176):
        super().__init__()
        # Read the Attention Is All You Need paper to learn how to code the positional encoding
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
        out1    =  x + out1
        # Apply batch norm to out1
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
        for tf_block in self.transformer_encoder:
            x  = tf_block(x)

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
        # Attention between encoder and decoder
        attn_output = self.enc_dec_attn(decoder_output, encoder_output, encoder_output)
        # Adding residual connection and layer normalization
        attn_output = self.layer_norm(attn_output + decoder_output)    # changed from encoder_output
        # Convert to probability distribution
        output = self.CDN(attn_output)
        output = self.softmax(output)

        return output

    def generate(self, x, max_length):
        with torch.no_grad():  # Inference mode
            generated_seq = x
            for _ in range(max_length):
                output = self.forward(generated_seq)
                output = output[-1]  # Get the last output
                _, next_token = torch.max(output, dim=-1)
                generated_seq = torch.cat((generated_seq, next_token.unsqueeze(0)), dim=0)

            return generated_seq[1:,-1]
        
# Create ChatNanoGPT
class ChatNanoGPT(torch.nn.Module):
    def __init__(self, 
                 stoi=stoi_comm, 
                 itos=itos_comm, 
                 hidden_size=config["hidden_size"], 
                 tf_block=config["tf_block"], 
                 encoder_nhead=config["encoder_nhead"], 
                 decoer_nhead=config["decoder_nhead"], 
                 cross_nhead=config["cross_nhead"]):
        super(ChatNanoGPT, self).__init__()

        self.stoi, self.itos = stoi, itos
        self.model = NanoGPT(vocab_size=len(stoi),
                             hidden_size=hidden_size,
                             tf_block=tf_block,
                             encoder_nhead=encoder_nhead,
                             decoer_nhead=decoer_nhead,
                             cross_nhead=cross_nhead).to(device)
        # Load pretrained model
        self.model.load_state_dict(torch.load('pretrain_best.pth'))

    # Mode selector
    def mode_selector(self, input):
        if input["mode"] == "summarization":
            return random.randint(15, 25), input["article"]
        elif input["mode"] == "qa":
            return random.randint(5, 15), input["context"] + input["question"]
        else:
            assert False, "Mode should be either summarization or qa!"

    # Tokenizer
    def tokenize(self, text):
        return re.findall(r"\b\w+'?\w*|[^\w\s]", text.lower())

    # Encoding and decoding functions
    def encode(self, s, stoi):
        encoded = []
        for word in self.tokenize(s):
            UNKNOWN_TOKEN = len(stoi) - 1
            try:
                encoded.append(int(stoi[word]))
            except KeyError:
                encoded.append(UNKNOWN_TOKEN)
        return encoded
    
    def decode(self, l):
        decoded = []
        for i in l:
            try:
                curr_token = self.itos[str(i[0])]
                if curr_token == ".":
                    continue
                else:
                    decoded.append(curr_token)
            except KeyError:
                continue
        return decoded
    
    # Chat function
    def chat(self, input):
        l, x = self.mode_selector(input)

        # encode input
        encoded = self.encode(x, self.stoi)
        encoded = torch.tensor(encoded).view(1, -1).to(device)

        # generate output
        if input["mode"] == "summarization":
            with torch.no_grad():
                output = self.model.forward(encoded).argmax(-1)[0]
        else:
            output = self.model.generate(encoded, l)

        # decode output
        output = self.decode(np.array(output.to('cpu').unsqueeze(0).T))

        return ' '.join(output)
