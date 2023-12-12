<div align="center">
  <h1 class="title">LLM: ChatNanoGPT</h1>
  <br> Carnegie Mellon University
  <br> 11685 - Introduction to Deep Learning
  <br> Fall 2023 HW5
  <br> Tao Luo, Yiyang Zheng
</div>

## Step 1: Data preprocessing:
1. Build vocabulary dictionaries and encode data from OpenWebText dataset:
> build_dict.py
2. Encode data from CNN Dailymail and SQuAD datasets:
> encode_finetune.py
3. String to index dictionary:
> stoi_comm.json
4. Index to string dictionary:
> itos_comm.json

## 3 stages for this project:
1. Pretrain

> pretrain_model1.py (126.6M parameters) \
> pretrain_model2.py (57.3M parameters) \
> pretrain_model3.py (54.7M parameters) \
> pretrain_model4.py (26.5M parameters)

* Link to download <pretain_best.pth> (for model 1):
> https://drive.google.com/file/d/1zfdaTa1Ky_ag5OLHuqb2Xb5F4FXq5DI-/view?usp=drive_link

<div align="center">
  <img src="https://github.com/luotao9728/chatnanogpt/blob/main/architectures.png" alt="image" width="100%" height="auto">
  <img src="https://github.com/luotao9728/chatnanogpt/blob/main/table.png" alt="image" width="50%" height="auto">
</div>

2. Finetune
> finetune.py (126.6M parameters)

* Link to download <finetune.pth>:
> https://drive.google.com/file/d/1rDawxNDpJJgD9RK1bwhCyuU3jzQLM07N/view?usp=sharing

3. ChatNanoGPT model:

> chatnanogpt.py

<div align="center">
  <img src="https://github.com/luotao9728/chatnanogpt/blob/main/wrapper.png" alt="image" width="50%" height="auto">
</div>

4. API-like access:

> Note that the input must be python dictionaries or in JSON format. 
> * For summarization task, the format of the input should be: \
> --> { "mode": "summuarization", "article": "article contents"} 
> 
> * For QA task, the format of the input should be: \
> -->  { "mode": "qa", "context": "context for the question", "question": "your question here"}

* Quick Start:
> from chatnanogpt import ChatNanoGPT 
> 
> chat = ChatNanoGPT() \
> output = chat.chat(input) 

* Detailed demos could be found on: 
> lets_chat.ipynb
