#%%
import numpy as np
import pandas as pd
import dask.dataframe as dd
import datatable as dt
from time import time
from tqdm import tqdm
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchsummary import summary
# %%
'''
Download Kaggle data using kaggle API:
kaggle competitions download riiid-test-answer-prediction --path ./data
'''
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
start = time()
data = dt.fread(DATA_DIR+"train.csv")
print(f"Readding train.csv in {time()-start} seconds")
# df_train = data.to_pandas()

# %%
TRAIN_DTYPES = {
    # 'row_id': np.uint32,
    'timestamp': np.int64,
    'user_id': np.int32,
    'content_id': np.int16,
    'content_type_id': np.int8,
    'task_container_id': np.int16,
    'user_answer': np.int8,
    'answered_correctly': np.int8,
    'prior_question_elapsed_time': np.float32,
    'prior_question_had_explanation': 'boolean'
}
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
start = time()
df_train = pd.read_csv(DATA_DIR+'train.csv', 
                    #    nrows=40_00_000, 
                       dtype=TRAIN_DTYPES, 
                       usecols=TRAIN_DTYPES.keys())
print(f"Readding train.csv in {time()-start} seconds")

# %%
LAST_N = 100 # this parameter denotes how many last seen content_ids I am going to consider <aka the max_seq_len or the window size>.
FILLNA_VAL = 100 # fillers for the values

df_questions = pd.read_csv(DATA_DIR+'questions.csv')

df_train['prior_question_had_explanation'] = df_train['prior_question_had_explanation'].astype(np.float16).fillna(-1).astype(np.int8)
df_train = df_train[df_train.content_type_id == 0]

part_ids_map = dict(zip(df_questions.question_id, df_questions.part))
df_train['part_id'] = df_train['content_id'].map(part_ids_map)
# %%
from collections import Counter
df_train["prior_question_elapsed_time"].fillna(FILLNA_VAL, inplace=True) # different than all current values
df_train["prior_question_elapsed_time"] = df_train["prior_question_elapsed_time"] // 1000
# %%
from collections import deque

# we will be using a deque as it automatically limits the max_size as per the Data Strucutre's defination itself
# so we don't need to manage that...

d = {}
user_id_to_idx = {}

PAD = 0

grp = df_train.groupby("user_id").tail(LAST_N) # Select last_n rows of each user.

for idx, row in tqdm(grp.groupby("user_id").agg({
    "content_id":list, "answered_correctly":list, "task_container_id":list, 
    "part_id":list, "prior_question_elapsed_time":list
    }).reset_index().iterrows()):
    # here we make a split whether a user has more than equal to 100 entries or less than that
    # if it's less than LAST_N, then we need to PAD it using the PAD token defined as 0 by me in this cell block
    # also, padded will be True where we have done padding obviously, rest places it's False.
    if len(row["content_id"]) >= 100:
        d[idx] = {
            "user_id": row["user_id"],
            "content_id" : deque(row["content_id"], maxlen=LAST_N),
            "answered_correctly" : deque(row["answered_correctly"], maxlen=LAST_N),
            "task_container_id" : deque(row["task_container_id"], maxlen=LAST_N),
            "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"], maxlen=LAST_N),
            "part_id": deque(row["part_id"], maxlen=LAST_N),
            "padded" : deque([False]*100, maxlen=LAST_N)
        }
    else:
        # we have to pad...
        # (max_batch_len - len(seq))
        d[idx] = {
            "user_id": row["user_id"],
            "content_id" : deque(row["content_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
            "answered_correctly" : deque(row["answered_correctly"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
            "task_container_id" : deque(row["task_container_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
            "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
            "part_id": deque(row["part_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
            "padded" : deque([False]*len(row["content_id"]) + [True]*(100-len(row["content_id"])), maxlen=LAST_N)
        }
    user_id_to_idx[row["user_id"]] = idx
    # if in future a new user comes, we will just increase the counts as of now... <WIP>
# %%
print(user_id_to_idx[115], d[0]) # user_id 115; I encourage you to match the same with the dataframes.
# %%
class TransformerModel(nn.Module):

    def __init__(self, ninp:int=32, nhead:int=2, nhid:int=64, nlayers:int=2, dropout:float=0.3):
        '''
        nhead -> number of heads in the transformer multi attention thing.
        nhid -> the number of hidden dimension neurons in the model.
        nlayers -> how many layers we want to stack.
        '''
        super(TransformerModel, self).__init__()
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.exercise_embeddings = nn.Embedding(num_embeddings=13523, embedding_dim=ninp) # exercise_id
        self.pos_embedding = nn.Embedding(ninp, ninp) # positional embeddings
        self.part_embeddings = nn.Embedding(num_embeddings=7+1, embedding_dim=ninp) # part_id_embeddings
        self.prior_question_elapsed_time = nn.Embedding(num_embeddings=301, embedding_dim=ninp) # prior_question_elapsed_time
        self.device = "cpu"
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # init embeddings
        self.exercise_embeddings.weight.data.uniform_(-initrange, initrange)
        self.part_embeddings.weight.data.uniform_(-initrange, initrange)
        self.prior_question_elapsed_time.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, content_id, part_id, prior_question_elapsed_time=None, mask_src=None):
        '''
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        padding mask is (N, S) with boolean True/False.
        SRC_MASK is (S, S) with float(’-inf’) and float(0.0).
        '''

        embedded_src = self.exercise_embeddings(content_id) + \
        self.pos_embedding(torch.arange(0, content_id.shape[1]).to(self.device).unsqueeze(0).repeat(content_id.shape[0], 1)) + \
        self.part_embeddings(part_id) + self.prior_question_elapsed_time(prior_question_elapsed_time) # (N, S, E)
        embedded_src = embedded_src.transpose(0, 1) # (S, N, E)
        
        _src = embedded_src * np.sqrt(self.ninp)
        
        output = self.transformer_encoder(src=_src, src_key_padding_mask=mask_src)
        output = self.decoder(output)
        output = output.transpose(1, 0)
        return output

def pad_seq(seq: List[int], max_batch_len: int = LAST_N, pad_value: int = True) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]
# %%
# pytorch dataset class

class Riiid(torch.utils.data.Dataset):
    
    def __init__(self, d):
        self.d = d
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        # you can return a dict of these as well etc etc...
        # remember the order
        return idx, self.d[idx]["content_id"], self.d[idx]["task_container_id"], \
    self.d[idx]["part_id"], self.d[idx]["prior_question_elapsed_time"], self.d[idx]["padded"], \
    self.d[idx]["answered_correctly"]

def collate_fn(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded, labels = zip(*batch)
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    labels = torch.Tensor(labels)
    # remember the order
    return content_id, task_id, part_id, prior_question_elapsed_time, padded, labels

dataset = Riiid(d=d)
print(dataset[0]) # sample dataset
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
sample = next(iter(torch.utils.data.DataLoader(dataset=dataset, 
                batch_size=1, collate_fn=collate_fn, num_workers=8))) # dummy check
# createing the mdoel
model = TransformerModel(ninp=LAST_N, nhead=4, nhid=128, nlayers=3, dropout=0.3)
model = model.to(device)
# %%
BATCH_SIZE = 32
losses = []
criterion = nn.BCEWithLogitsLoss()
lr = 1e-3 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.train()


for idx,batch in tqdm(enumerate(torch.utils.data.DataLoader(dataset=dataset, 
                                                            batch_size=BATCH_SIZE,             collate_fn=collate_fn, 
                                                            num_workers=8))):
    content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
    content_id = content_id.cuda()
    part_id = part_id.cuda()
    prior_question_elapsed_time = prior_question_elapsed_time.cuda()
    mask = mask.cuda()
    labels = labels.cuda()
    break
    optimizer.zero_grad()
    with torch.set_grad_enabled(mode=True):
        output = model(content_id, part_id, prior_question_elapsed_time, mask)
        # output is (N,S,2) # i am working on it
        loss = criterion(output[:,:,1], labels)
        loss.backward()
        losses.append(loss.detach().data.numpy())
        optimizer.step()
# %%
