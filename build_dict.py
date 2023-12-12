import os
import glob
import re
import numpy as np
import json

def tokenize(text):
    return re.findall(r"\b\w+'?\w*|[^\w\s]", text.lower())

# Directory path
dir_path = "subset"

# Find all .txt files in the directory
txt_files = glob.glob(os.path.join(dir_path, "*.txt"))

# Read and tokenize all files
all_text = []
stoi, itos = {}, {}
index = 0
for file in txt_files:
    with open(file, 'r', encoding='utf-8') as f:
        # Build vocabulary
        for word in tokenize(f.read()):
            if word not in stoi:
                stoi[word] = [index, 1]
                itos[index] = word
                index += 1
            else:
                stoi[word][1] += 1
    f.close()

def dict_comm(stoi, n):
    stoi_comm, itos_comm = {}, {}
    ind = 0

    # Sort the dictionary by the second element of the tuple and take the top n items
    sorted_items = sorted(stoi.items(), key=lambda x: x[1][1], reverse=True)[:n]

    # Construct the new dictionaries
    for key, _ in sorted_items:
        stoi_comm[key] = ind
        itos_comm[ind] = key
        ind += 1

    # Add the <UNKNOWN> token
    stoi_comm['<UNKNOWN>'] = ind
    itos_comm[ind] = ''

    return stoi_comm, itos_comm

print("Start building dictionaries!")
stoi_comm, itos_comm = dict_comm(stoi, 50000)
print("Finished building dictionaries!")

# Save dictionaries
json.dump(stoi_comm, open('stoi_comm.json', 'w'))
json.dump(itos_comm, open('itos_comm.json', 'w'))
print("Dictionaries saved!")

# Encoding and decoding functions
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
            decoded.append(itos[int(i)])
        except KeyError:
            continue
    return decoded

# Encoding all text files
print("Start encoding!")
text = []
for file in txt_files:
    with open(file, 'r', encoding='utf-8') as f:
        text.append(np.array(encode(f.read(), stoi_comm), dtype=int))
    f.close()
        
text = np.array(text, dtype=object)
print("Finished encoding!")

print('Vocab size:', len(stoi_comm))
print('Number of sequences:', len(text))

# Save encoded data
np.save('test.npy', text)
print("Encoded data saved!")