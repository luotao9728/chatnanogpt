import datasets
import re
import numpy as np
import json

# Download cnn_dailymail_dataset and SQuAD_dataset
cnn_dailymail_dataset = datasets.load_dataset("cnn_dailymail", '1.0.0', download_mode="force_redownload")
SQuAD_dataset = dataset = datasets.load_dataset("squad", download_mode="force_redownload")

# Load dictionaries and data
stoi_comm = json.load(open('stoi_comm.json', 'r'))

# Tokenizer
def tokenize(text):
    return re.findall(r"\b\w+'?\w*|[^\w\s]", text.lower())

# Encoding function
def encode(s, stoi):
    encoded = []
    for word in tokenize(s):
        UNKNOWN_TOKEN = len(stoi) - 1
        try:
            encoded.append(int(stoi[word]))
        except KeyError:
            encoded.append(UNKNOWN_TOKEN)
    return encoded

# Encode and save cnn_dailymail_dataset
cnn_dailymail_splits = ['train', 'validation', 'test']
for split in cnn_dailymail_splits:
    cnn_dailymail = []
    for data in enumerate(cnn_dailymail_dataset[split]):
        pair = np.array([np.array(encode(data[1]['article'], stoi_comm), dtype=int),
                         np.array(encode(data[1]['highlights'], stoi_comm), dtype=int)],
                        dtype=object)

        cnn_dailymail.append(pair)

    cnn_dailymail = np.array(cnn_dailymail, dtype=object)
    np.save('cnn_dailymail_%s.npy' % split, cnn_dailymail)
    print('cnn_dailymail_%s.npy saved!' % split)

# Encode and save SQuAD_dataset
SQuAD_splits = ['train', 'validation']
for split in SQuAD_splits:
    SQuAD = []
    for data in enumerate(SQuAD_dataset[split]):
        pair = np.array([np.array(encode(data[1]['title'], stoi_comm), dtype=int),
                         np.array(encode(data[1]['context'], stoi_comm), dtype=int),
                         np.array(encode(data[1]['question'], stoi_comm), dtype=int),
                         np.array(encode(data[1]['answers']['text'][0], stoi_comm), dtype=int)],
                        dtype=object)

        SQuAD.append(pair)

    SQuAD = np.array(SQuAD, dtype=object)
    np.save('SQuAD_%s.npy' % split, SQuAD)
    print('SQuAD_%s.npy saved!' % split)