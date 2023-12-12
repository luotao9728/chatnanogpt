# ChatNanoGPT
> Carnegie Mellon University \
> Fall 2023 HW5 \
> Tao Luo, Yiyang Zheng

## Data preprocessing:
1. Build vocabulary dictionaries and encode data from OpenWebText dataset:
> build_dict.py
2. Encode data from CNN Dailymail and SQuAD datasets:
> encode_finetune.py

## Dictionaries:
1. String to index dictionary:
> stoi_comm.json
2. Index to string dictionary:
> itos_comm.json

## 3 stages for this project:
1. Pretrain
> pretrain_model1.py (126.6M parameters) \
> pretrain_model2.py (57.3M parameters) \
> pretrain_model3.py (54.7M parameters) \
> pretrain_model4.py (26.5M parameters)

Link to download <pretain_best.pth> (for model 1):
> https://drive.google.com/file/d/1zfdaTa1Ky_ag5OLHuqb2Xb5F4FXq5DI-/view?usp=drive_link

2. Finetune
> finetune.py (126.6M parameters)

Link to download <finetune.pth>:
> https://drive.google.com/file/d/1rDawxNDpJJgD9RK1bwhCyuU3jzQLM07N/view?usp=sharing

3. ChatNanoGPT model:
> chatnanogpt.py

4. API-like access:
> from chatnanogpt import ChatNanoGPT \
> chat = ChatNanoGPT()

> Note that the input must be python dictionaries or in JSON format. 
>> For summarization task, the format of the input should be: \
>> { "mode": "summuarization", "article": "article contents"}
>
>> For QA task, the format of the input should be: \
>> { "mode": "qa", "context": "context for the question", "question": "your question here"}
>
> output = chat.chat(input) 

Detailed demos could be found on: 
> lets_chat.ipynb
