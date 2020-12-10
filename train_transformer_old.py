#%%
import numpy as np
import pandas as pd
import dask.dataframe as dd
import datatable as dt
from time import time
from tqdm import tqdm
from typing import List
from collections import deque, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable
from torchsummary import summary

from sklearn.metrics import roc_auc_score

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
LAST_N = 100 # this parameter denotes how many last seen content_ids I am going to consider <aka the max_seq_len or the window size>.
FILLNA_VAL = 100 # fillers for the values (a unique value)
TQDM_INT = 15 # tqdm update interval
PAD = 0
BATCH_SIZE = 128
VAL_BATCH_SIZE = 512
NROWS_TRAIN = 40_000_000
NROWS_VALID = 10_000_000
EPOCHS = 10

def get_feats_tranformer(data_df):
    '''
    Using a deque as it automatically limits the max_size as per the Data Strucutre's defination itself
    so we don't need to manage that...
    '''
    df = {}
    user_id_to_idx = {}
    grp = data_df.groupby("user_id").tail(LAST_N) # Select last_n rows of each user.
    grp_user = grp.groupby("user_id")
    num_user_id_grp = len(grp_user)

    with tqdm(total=num_user_id_grp) as pbar:
        for idx, row in grp_user.agg({
            "content_id":list, "answered_correctly":list, "task_container_id":list, 
            "part_id":list, "prior_question_elapsed_time":list
            }).reset_index().iterrows():
            # here we make a split whether a user has more than equal to 100 entries or less than that
            # if it's less than LAST_N, then we need to PAD it using the PAD token defined as 0 by me in this cell block
            # also, padded will be True where we have done padding obviously, rest places it's False.
            if len(row["content_id"]) >= 100:
                df[idx] = {
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
                df[idx] = {
                    "user_id": row["user_id"],
                    "content_id" : deque(row["content_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "answered_correctly" : deque(row["answered_correctly"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "task_container_id" : deque(row["task_container_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "part_id": deque(row["part_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "padded" : deque([False]*len(row["content_id"]) + [True]*(100-len(row["content_id"])), maxlen=LAST_N)
                }
            user_id_to_idx[row["user_id"]] = idx
            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)
        # if in future a new user comes, we will just increase the counts as of now... <WIP>
    return df, user_id_to_idx


def roc_auc_compute_fn(y_targets, y_preds):
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred)

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


# %%
start = time()
df_train = pd.read_csv(DATA_DIR+'train.csv', 
                       nrows=NROWS_TRAIN, 
                       dtype=TRAIN_DTYPES, 
                       usecols=TRAIN_DTYPES.keys())
print(f"Readding train.csv in {time()-start} seconds\n\n")

df_questions = pd.read_csv(DATA_DIR+'questions.csv')

start = time()
df_train['prior_question_had_explanation'] = df_train['prior_question_had_explanation'].astype(np.float16).fillna(-1).astype(np.int8)
df_train = df_train[df_train.content_type_id == 0]

part_ids_map = dict(zip(df_questions.question_id, df_questions.part))
df_train['part_id'] = df_train['content_id'].map(part_ids_map)

df_train["prior_question_elapsed_time"].fillna(FILLNA_VAL, inplace=True) # different than all current values
df_train["prior_question_elapsed_time"] = df_train["prior_question_elapsed_time"] // 1000


d, user_id_to_idx = get_feats_tranformer(df_train)
print(f"Processing train.csv in {time()-start} seconds\n\n")
print(user_id_to_idx[115], d[0]) # user_id 115; I encourage you to match the same with the dataframes.


# %%

dataset = Riiid(d=d)
# print(dataset[0]) # sample dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
sample = next(iter(torch.utils.data.DataLoader(dataset=dataset, 
                batch_size=1, collate_fn=collate_fn, num_workers=8))) # dummy check
# createing the mdoel
model = TransformerModel(ninp=LAST_N, nhead=4, nhid=128, nlayers=3, dropout=0.3)
model = model.to(device)
# %% training
losses = []
history = []

auc_max = -np.inf


# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
lr = 1e-3 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.train()
dataset_train = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=BATCH_SIZE,
                                            collate_fn=collate_fn, 
                                            num_workers=8)

len_dataset = len(dataset_train)


for epoch in range(1, EPOCHS+1):
    with tqdm(total=len_dataset) as pbar:
        for idx,batch in enumerate(dataset_train):
            content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
            content_id = Variable(content_id.cuda())
            part_id = Variable(part_id.cuda())
            prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
            mask = Variable(mask.cuda())
            labels = Variable(labels.cuda().long())
            optimizer.zero_grad()
            with torch.set_grad_enabled(mode=True):
                output = model(content_id, part_id, prior_question_elapsed_time, mask)
                # output is (N,S,2) # i am working on it
                
                # loss = criterion(output[:,:,1], labels) # BCEWithLogitsLoss
                loss = criterion(output.reshape(-1, 2), labels.reshape(-1)) # Flatten and use crossEntropy
                loss.backward()
                losses.append(loss.cpu().detach().data.numpy())
                optimizer.step()
            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)
# %% The current validation set will have leakage
df_valid = pd.read_csv(DATA_DIR+"train.csv", nrows=NROWS_VALID, 
                                             dtype=TRAIN_DTYPES, 
                                             usecols=TRAIN_DTYPES.keys(), 
                                             skiprows=range(1, NROWS_TRAIN)
                      )
df_valid = df_valid[df_valid.content_type_id == 0]

df_valid["prior_question_elapsed_time"].fillna(FILLNA_VAL, inplace=True)
df_valid["prior_question_elapsed_time"] = df_valid["prior_question_elapsed_time"] // 1000

part_ids_map = dict(zip(df_questions.question_id, df_questions.part))
df_valid['part_id'] = df_valid['content_id'].map(part_ids_map)
d, user_id_to_idx = get_feats_tranformer(df_valid)
# %%


model.eval()
dataset = Riiid(d=d)
scores = []
dataset_val = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=VAL_BATCH_SIZE, 
                                          collate_fn=collate_fn, drop_last=True)
len_dataset = len(dataset_val)

with tqdm(total=len_dataset) as pbar:
    for idx,batch in enumerate(dataset_val):
        content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
        content_id = Variable(content_id.cuda())
        part_id = Variable(part_id.cuda())
        prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
        mask = Variable(mask.cuda())
        labels = Variable(labels.cuda())
        with torch.set_grad_enabled(mode=False):
                output = model(content_id, part_id, prior_question_elapsed_time, mask)

                # crossEntropy loss
                output_prob = torch.softmax(output, dim=2)
                # pred = (output_prob >= 0.50)
                predicted_classes = torch.argmax(output, dim=2)

                # BCE loss
                # output_prob = output[:,:,1]
                # pred = (output_prob >= 0.50)
                # print(output.shape, labels.shape) # torch.Size([N, S, 2]) torch.Size([N, S])
                # _, predicted_classes = torch.max(output[:,:,].data, 1)

                score = roc_auc_compute_fn(labels, predicted_classes)
                scores.append(score)
        if idx % TQDM_INT == 0:
            pbar.update(TQDM_INT)

pd.Series(losses).astype(np.float32).plot(kind="line")
print(f"\n\nThe mean auc score is {np.mean(scores)}")

# %%
