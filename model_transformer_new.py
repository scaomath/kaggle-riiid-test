#%%
import os
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
from torch.utils.data import Dataset, DataLoader
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
MODEL_DIR = '/home/scao/Documents/kaggle-riiid-test/model/'
FOLD = 1
LAST_N = 100 # this parameter denotes how many last seen content_ids I am going to consider <aka the max_seq_len or the window size>.
TAIL_N = 100 # used for validation set per user_id
FILLNA_VAL = 100 # fillers for the values (a unique value)
TQDM_INT = 15 # tqdm update interval
PAD = 0
BATCH_SIZE = 128
VAL_BATCH_SIZE = 1024
DEBUG = False
NROWS_TRAIN = 5_000_000
NROWS_VALID = 2_000_000
EPOCHS = 20



def roc_auc_compute_fn(y_targets, y_preds):
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred)


def get_valid(train_df, n_tail=50):

    valid_df = train_df.groupby(['user_id']).tail(n_tail)
    print("valid:", valid_df.shape, "users:", valid_df['user_id'].nunique())
    # Train
    train_df.drop(valid_df.index, inplace = True)
    print("train:", train_df.shape, "users:", train_df['user_id'].nunique())
    return train_df, valid_df


def preprocess(data_df, question_df, train=True):
    if train:
        data_df['prior_question_had_explanation'] = \
            data_df['prior_question_had_explanation'].astype(np.float16).fillna(-1).astype(np.int8)
    data_df = data_df[data_df.content_type_id == 0]

    part_ids_map = dict(zip(question_df.question_id, question_df.part))
    data_df['part_id'] = data_df['content_id'].map(part_ids_map)

    data_df["prior_question_elapsed_time"].fillna(FILLNA_VAL, inplace=True) 
    # FILLNA_VAL different than all current values
    data_df["prior_question_elapsed_time"] = data_df["prior_question_elapsed_time"] // 1000
    return data_df


def get_feats(data_df):
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

def get_feats_test(data_df):
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
            "content_id":list, "task_container_id":list, 
            "part_id":list, "prior_question_elapsed_time":list
            }).reset_index().iterrows():
            # here we make a split whether a user has more than equal to 100 entries or less than that
            # if it's less than LAST_N, then we need to PAD it using the PAD token defined as 0 by me in this cell block
            # also, padded will be True where we have done padding obviously, rest places it's False.
            if len(row["content_id"]) >= 100:
                df[idx] = {
                    "user_id": row["user_id"],
                    "content_id" : deque(row["content_id"], maxlen=LAST_N),
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

def train_epoch(model, train_iterator, optimizer, criterion):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    label_all = []
    pred_all = []
    len_dataset = len(train_iterator)

    with tqdm(total=len_dataset) as pbar:
        for idx,batch in enumerate(train_iterator):
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
                optimizer.step()

                train_loss.append(loss.cpu().detach().data.numpy())
            pred_probs = torch.softmax(output, dim=2)
            pred = torch.argmax(pred_probs, dim=2)
            num_corrects += (pred == labels).sum().item()
            num_total += len(labels)

            label_all.extend(labels.reshape(-1).data.cpu().numpy())
            # pred_all.extend(pred.reshape(-1).data.cpu().numpy())
            pred_all.extend(pred_probs[:,:,1].reshape(-1).data.cpu().numpy()) # use probability to do auc

            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)

    acc = num_corrects / num_total
    auc = roc_auc_score(label_all, pred_all)
    loss = np.mean(train_loss)

    return loss, acc, auc


def valid_epoch(model, valid_iterator, criterion):
    model.eval()
    valid_loss = []
    num_corrects = 0
    num_total = 0
    label_all = []
    pred_all = []
    len_dataset = len(valid_iterator)

    with tqdm(total=len_dataset) as pbar:
        for idx, batch in enumerate(valid_iterator):
            content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
            content_id = Variable(content_id.cuda())
            part_id = Variable(part_id.cuda())
            prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
            mask = Variable(mask.cuda())
            labels = Variable(labels.cuda().long())
            with torch.set_grad_enabled(mode=False):
                output = model(content_id, part_id, prior_question_elapsed_time, mask)
                loss = criterion(output.reshape(-1, 2), labels.reshape(-1)) # Flatten and use crossEntropy
            # crossEntropy loss
            valid_loss.append(loss.cpu().detach().data.numpy())
            pred_probs = torch.softmax(output, dim=2)
            # pred = (output_prob >= 0.50)
            pred = torch.argmax(pred_probs, dim=2)
            
            # BCE loss
            # output_prob = output[:,:,1]
            # pred = (output_prob >= 0.50)
            # print(output.shape, labels.shape) # torch.Size([N, S, 2]) torch.Size([N, S])
            # _, predicted_classes = torch.max(output[:,:,].data, 1)

            num_corrects += (pred == labels).sum().item()
            num_total += len(labels)
            label_all.extend(labels.reshape(-1).data.cpu().numpy())
            # pred_all.extend(pred.reshape(-1).data.cpu().numpy())
            pred_all.extend(pred_probs[:,:,1].reshape(-1).data.cpu().numpy()) # use probability to do auc
            
            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)

    acc = num_corrects / num_total
    auc = roc_auc_score(label_all, pred_all)
    loss = np.mean(valid_loss)

    return loss, acc, auc


# %% Preparing train and validation set
start = time()
if DEBUG: 
    train_df = pd.read_csv(DATA_DIR+'train.csv', 
                        nrows=NROWS_TRAIN, 
                        dtype=TRAIN_DTYPES, 
                        usecols=TRAIN_DTYPES.keys())
else:
    train_df = pd.read_csv(DATA_DIR+'train.csv', 
                    dtype=TRAIN_DTYPES, 
                    usecols=TRAIN_DTYPES.keys())
print(f"Readding train.csv in {time()-start} seconds\n\n")
df_questions = pd.read_csv(DATA_DIR+'questions.csv')

train_df, valid_df = get_valid(train_df, n_tail=TAIL_N)
train_df = preprocess(train_df, df_questions, train=True)
valid_df = preprocess(valid_df, df_questions, train=False)
d, user_id_to_idx = get_feats(train_df)
d_val, user_id_to_idx = get_feats(valid_df)
print(f"\nProcessing train and valid in {time()-start} seconds\n\n")
print(user_id_to_idx[115], d[0]) # user_id 115; I encourage you to match the same with the dataframes.


# %%
dataset_train = Riiid(d=d)
dataset_val = Riiid(d=d_val)
# print(dataset[0]) # sample dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
sample = next(iter(DataLoader(dataset=dataset_train, 
                batch_size=1, collate_fn=collate_fn))) # dummy check
# createing the mdoel
model = TransformerModel(ninp=LAST_N, nhead=4, nhid=128, nlayers=3, dropout=0.3)
model = model.to(device)
# %% training and validation
losses = []
history = []
auc_max = -np.inf

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
lr = 1e-3 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, collate_fn=collate_fn)

dataset_val = DataLoader(dataset=dataset_val, batch_size=VAL_BATCH_SIZE, collate_fn=collate_fn, drop_last=True)

snapshot_path = "%s/fold%d/snapshots" % (MODEL_DIR, FOLD)
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

for epoch in range(1, EPOCHS+1):
    train_loss, train_acc, train_auc = train_epoch(model, dataset_train, optimizer, criterion)
    print(f"\n\n[Epoch {epoch}/{EPOCHS}]")
    print(f"Train: loss - {train_loss:.2f} acc - {train_acc:.4f} auc - {train_auc:.4f}")
    valid_loss, valid_acc, valid_auc = valid_epoch(model, dataset_val, criterion)
    print(f"\nValid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")
    lr = optimizer.param_groups[0]['lr']
    history.append({"epoch":epoch, "lr": lr, 
                    **{"train_auc": train_auc, "train_acc": train_acc}, 
                    **{"valid_auc": valid_auc, "valid_acc": valid_acc}})
    if valid_auc > auc_max:
        print(f"[Epoch {epoch}/{EPOCHS}], valid loss {valid_loss:.4f}")
        print(f"Metric loss improved from {auc_max:.4f} to {valid_auc:.4f}, saving model ...")
        auc_max = valid_auc
        torch.save(model.state_dict(), os.path.join(snapshot_path, "model_best_epoch.pt"))
# %%
