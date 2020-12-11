from collections import Counter, deque
from time import time
from typing import List

import datatable as dt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

'''
Download Kaggle data using kaggle API:
kaggle competitions download riiid-test-answer-prediction --path ./data
This file contains the datatset class and feature engineering functions
'''

FILLNA_VAL = 100
LAST_N = 100
TQDM_INT = 8
PAD = 0
FOLD = 1
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/fold{FOLD}/snapshots/'
N_EXERCISES = 13523 #  train_df['content_id'].unique()
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
DATA_TABLE = False
LAST_N = 100

class Riiid(Dataset):
    def __init__(self, d):
        super(Riiid, self).__init__()
        self.d = d
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        # you can return a dict of these as well etc etc...
        # remember the order
        return idx, self.d[idx]["content_id"], self.d[idx]["task_container_id"], \
    self.d[idx]["part_id"], self.d[idx]["prior_question_elapsed_time"], self.d[idx]["padded"], \
    self.d[idx]["answered_correctly"]


class RiiidTest(Dataset):
    
    def __init__(self, d):
        super(RiiidTest, self).__init__()
        self.d = d
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        # you can return a dict of these as well etc etc...
        # remember the order
        return idx, self.d[idx]["content_id"], self.d[idx]["task_container_id"], \
    self.d[idx]["part_id"], self.d[idx]["prior_question_elapsed_time"], self.d[idx]["padded"]

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

def collate_fn_test(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded = zip(*batch)
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    # remember the order
    return content_id, task_id, part_id, prior_question_elapsed_time, padded

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
    grp = data_df.groupby("user_id", sort=False).tail(LAST_N) # Select last_n rows of each user.
    grp_user = grp.groupby("user_id", sort=False)
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
    grp = data_df.groupby("user_id", sort=False).tail(LAST_N) # Select last_n rows of each user.
    grp_user = grp.groupby("user_id", sort=False)
    
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

    return df, user_id_to_idx


def get_feats_train(data_df):
    '''
    Using a deque as it automatically limits the max_size as per the Data Strucutre's defination itself
    so we don't need to manage that...
    '''
    df = {}
    user_id_to_idx = {}
    grp = data_df.groupby("user_id", sort=False).tail(LAST_N) # Select last_n rows of each user.
    grp_user = grp.groupby("user_id", sort=False)

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

        # if in future a new user comes, we will just increase the counts as of now... <WIP>
    return df, user_id_to_idx

class TransformerModel(nn.Module):

    def __init__(self, ninp:int=32, nhead:int=4, nhid:int=64, nlayers:int=2, dropout:float=0.3):
        '''
        nhead -> number of heads in the transformer multi attention thing.
        nhid -> the number of hidden dimension neurons in the model.
        nlayers -> how many layers we want to stack.
        '''
        super(TransformerModel, self).__init__()
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.exercise_embeddings = nn.Embedding(num_embeddings=N_EXERCISES, embedding_dim=ninp) # exercise_id
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

            pred_probs = torch.softmax(output[~mask], dim=1)
            pred = torch.argmax(pred_probs, dim=1)
            labels = labels[~mask]
            num_corrects += (pred == labels).sum().item()
            num_total += len(labels)

            label_all.extend(labels.reshape(-1).data.cpu().numpy())
            # pred_all.extend(pred.reshape(-1).data.cpu().numpy())
            pred_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy()) # use probability to do auc

            if idx % TQDM_INT == 0:
                pbar.set_description(f'loss - {train_loss[-1]:.4f}')
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
            pred_probs = torch.softmax(output[~mask], dim=1)
            # pred = (output_prob >= 0.50)
            pred = torch.argmax(pred_probs, dim=1)
            
            # BCE loss
            # output_prob = output[:,:,1]
            # pred = (output_prob >= 0.50)
            # print(output.shape, labels.shape) # torch.Size([N, S, 2]) torch.Size([N, S])
            # _, predicted_classes = torch.max(output[:,:,].data, 1)
            
            labels = labels[~mask]
            num_corrects += (pred == labels).sum().item()
            num_total += len(labels)
            label_all.extend(labels.reshape(-1).data.cpu().numpy())
            # pred_all.extend(pred.reshape(-1).data.cpu().numpy())
            pred_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy()) # use probability to do auc
            
            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)

    acc = num_corrects / num_total
    auc = roc_auc_score(label_all, pred_all)
    loss = np.mean(valid_loss)

    return loss, acc, auc


if __name__ == "__main__":
    start = time()
    data = dt.fread(DATA_DIR+"train.csv")
    print(f"Readding train.csv in {time()-start} seconds")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(ninp=LAST_N, nhead=4, nhid=128, nlayers=3, dropout=0.3)
    model = model.to(device)
    # print("Loading state_dict...")
    # model.load_state_dict(torch.load(MODEL_DIR+'model_best_epoch.pt', map_location=device))
    # model.eval()
    print(model)
