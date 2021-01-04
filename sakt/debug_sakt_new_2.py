#%%
import os
import gc
import sys

import pickle
from time import time

import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR,
                                      ReduceLROnPlateau)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sns.set()
DEFAULT_FIG_WIDTH = 20
sns.set_context("paper", font_scale=1.2) 
# WORKSPACE_FOLDER=/home/scao/Documents/kaggle-riiid-test
# PYTHONPATH=${WORKSPACE_FOLDER}:${WORKSPACE_FOLDER}/sakt:${WORKSPACE_FOLDER}/transformer

from sakt import *

HOME = os.path.abspath(os.path.join('.', os.pardir))
print(HOME, '\n\n')
# HOME = "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
get_system()

# from transformer_optimizers import *
# %%

'''
TO-DO:
features encoding:
1 previous answers correctly 
2 question tags 
'''
DEBUG = True
TRAIN = False
PREPROCESS = False

TEST_SIZE = 0.05

NUM_SKILLS = 13523 # number of problems
MAX_SEQ = 180
ACCEPTED_USER_CONTENT_SIZE = 4
EMBED_SIZE = 128
NUM_HEADS = 8
BATCH_SIZE = 64
VAL_BATCH_SIZE = 2048
DROPOUT = 0.1
SEED = 1127

'''
Columns placeholder and preprocessing params
'''
CONTENT_TYPE_ID = "content_type_id"
CONTENT_ID = "content_id"
TARGET = "answered_correctly"
USER_ID = "user_id"
PRIOR_QUESTION_TIME = 'prior_question_elapsed_time'

PRIOR_QUESTION_EXPLAIN = 'prior_question_had_explanation'
TASK_CONTAINER_ID = "task_container_id"
TIMESTAMP = "timestamp" 
ROW_ID = 'row_id'
FILLNA_VAL = 14_000 # for prior question elapsed time, rounded average in train
TIME_SCALING = 1000 # scaling down the prior question elapsed time

TRAIN_COLS = [TIMESTAMP, USER_ID, CONTENT_ID, CONTENT_TYPE_ID, TARGET]
TRAIN_COLS_NEW = [TIMESTAMP, USER_ID, CONTENT_ID, CONTENT_TYPE_ID, 
             TARGET, PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN]

TRAIN_DTYPES = {TIMESTAMP: 'int64', 
         USER_ID: 'int32', 
         CONTENT_ID: 'int16',
         CONTENT_TYPE_ID: 'bool',
         TARGET:'int8',
         PRIOR_QUESTION_TIME: np.float32,
         PRIOR_QUESTION_EXPLAIN: 'boolean'}


if DEBUG:
    NROWS_TEST = 25_000
    NROWS_TRAIN = 5_000_000
    NROWS_VAL = 500_000
else:
    NROWS_TEST = 250_000
    NROWS_TRAIN = 50_000_000
    NROWS_VAL = 2_000_000
# %%
if PREPROCESS:
    with timer("Loading train from parquet"):
        train_df = pd.read_parquet(os.path.join(DATA_DIR, 'cv5_train.parquet'),
                                columns=list(TRAIN_DTYPES.keys())).astype(TRAIN_DTYPES)
        valid_df = pd.read_parquet(os.path.join(DATA_DIR, 'cv5_valid.parquet'),
                                columns=list(TRAIN_DTYPES.keys())).astype(TRAIN_DTYPES)

    if DEBUG:
        train_df = train_df[:NROWS_TRAIN]
        valid_df = valid_df[:NROWS_VAL]

    with timer("Processing train"):
        train_group = preprocess(train_df)
        valid_group = preprocess(valid_df, train_flag=2)
else:
    with open(os.path.join(DATA_DIR, 'sakt_group.pickle'), 'rb') as f:
        group = pickle.load(f)
    train_group, valid_group = train_test_split(group, test_size = TEST_SIZE, random_state=SEED)


print(f"valid users: {len(valid_group.keys())}")
print(f"train users: {len(train_group.keys())}")
# %%

class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=100):
        super(SAKTDataset, self).__init__()
        self.samples, self.n_skill, self.max_seq = {}, n_skill, max_seq
        
        self.user_ids = []
        for i, user_id in enumerate(group.index):
            content_id, answered_correctly = group[user_id]
            if len(content_id) >= ACCEPTED_USER_CONTENT_SIZE:
                if len(content_id) > self.max_seq:
                    total_questions = len(content_id)
                    last_pos = total_questions // self.max_seq
                    for seq in range(last_pos):
                        index = f"{user_id}_{seq}"
                        self.user_ids.append(index)
                        start = seq * self.max_seq
                        end = (seq + 1) * self.max_seq
                        self.samples[index] = (content_id[start:end], 
                                               answered_correctly[start:end])
                    if len(content_id[end:]) >= ACCEPTED_USER_CONTENT_SIZE:
                        index = f"{user_id}_{last_pos + 1}"
                        self.user_ids.append(index)
                        self.samples[index] = (content_id[end:], 
                                               answered_correctly[end:])
                else:
                    index = f'{user_id}'
                    self.user_ids.append(index)
                    self.samples[index] = (content_id, answered_correctly)
                
                
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        content_id, answered_correctly = self.samples[user_id]
        seq_len = len(content_id)
        
        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            content_id_seq[:] = content_id[-self.max_seq:]
            answered_correctly_seq[:] = answered_correctly[-self.max_seq:]
        else:
            content_id_seq[-seq_len:] = content_id
            answered_correctly_seq[-seq_len:] = answered_correctly
            
        target_id = content_id_seq[1:] # question till the current one
        label = answered_correctly_seq[1:]
        
        x = content_id_seq[:-1].copy() # question till the previous one
        # encoded answers till the previous one
        x += (answered_correctly_seq[:-1] == 1) * self.n_skill
        
        return x, target_id, label
# %%
train_dataset = SAKTDataset(train_group, n_skill=NUM_SKILLS, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                drop_last=True)


val_dataset = SAKTDataset(valid_group, n_skill=NUM_SKILLS, max_seq=MAX_SEQ)
val_dataloader = DataLoader(val_dataset, 
                            batch_size=VAL_BATCH_SIZE, 
                            shuffle=False)

sample_batch = next(iter(val_dataloader))
sample_batch[0].shape, sample_batch[1].shape, sample_batch[2].shape
# %%
'''
Debugging
'''
content_id, answered_correctly = valid_group[valid_group.keys()[0]]
seq_len = len(content_id)
content_id_seq = np.zeros(MAX_SEQ, dtype=int)
answered_correctly_seq = np.zeros(MAX_SEQ, dtype=int)
if seq_len >= MAX_SEQ:
    content_id_seq[:] = content_id[-MAX_SEQ:]
    answered_correctly_seq[:] = answered_correctly[-MAX_SEQ:]
else:
    content_id_seq[-seq_len:] = content_id
    answered_correctly_seq[-seq_len:] = answered_correctly

# question till the current one, should be the same with sample_batch[1][0]
target_id = content_id_seq[1:] 
#  whether answered correctly, same with sample_batch[2][0]
label = answered_correctly_seq[1:] #
x = content_id_seq[:-1].copy() # question till the previous one
# encoded answers till the previous question
# if a user answered correctly it is added 13523
x += (answered_correctly_seq[:-1] == 1) * NUM_SKILLS
# %%
questions_df = pd.read_csv(os.path.join(DATA_DIR, 'questions.csv'))
questions_df['part'] = questions_df['part'].astype(np.int32)
questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)

train_debug = pd.merge(train_df, questions_df[['question_id', 'part']], 
                 left_on = 'content_id', right_on = 'question_id', how = 'left')
# %% model

class FFN(nn.Module):
    def __init__(self, state_size = 200, 
                    forward_expansion = 1, 
                    bn_size=MAX_SEQ - 1, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size
        
        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, 
                    heads = 8, 
                    dropout = DROPOUT, 
                    forward_expansion = 1):
        super(TransformerBlock, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, 
                        num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, 
                    forward_expansion = forward_expansion, 
                    dropout=dropout)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)
        

    def forward(self, value, key, query, att_mask):
        att_output, att_weight = self.multi_att(value, key, query, attn_mask=att_mask)
        att_output = self.dropout(self.layer_normal(att_output + value))
        att_output = att_output.permute(1, 0, 2) 
        # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        x = self.ffn(att_output)
        x = self.dropout(self.layer_normal_2(x + att_output))
        return x.squeeze(-1), att_weight
    
class Encoder(nn.Module):
    def __init__(self, n_skill, max_seq=100, 
                 embed_dim=128, 
                 dropout = DROPOUT, 
                 forward_expansion = 1, 
                 num_layers=1, 
                 heads = 8):
        super(Encoder, self).__init__()
        self.n_skill, self.embed_dim = n_skill, embed_dim
        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, heads=heads,
                forward_expansion = forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, question_ids):
        device = x.device
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(x + pos_x)
        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = self.e_embedding(question_ids)
        e = e.permute(1, 0, 2)
        for layer in self.layers:
            att_mask = future_mask(e.size(0)).to(device)
            x, att_weight = layer(e, x, x, att_mask=att_mask)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x, att_weight

class SAKTModel(nn.Module):
    def __init__(self, 
                n_skill, 
                max_seq=100, 
                embed_dim=128, 
                dropout = DROPOUT, 
                forward_expansion = 1, 
                enc_layers=1, 
                heads = 8):
        super(SAKTModel, self).__init__()
        self.encoder = Encoder(n_skill, 
                               max_seq, 
                               embed_dim, 
                               dropout, 
                               forward_expansion, 
                               num_layers=enc_layers,
                               heads=heads)
        self.pred = nn.Linear(embed_dim, 1)
        
    def forward(self, x, question_ids):
        x, att_weight = self.encoder(x, question_ids)
        x = self.pred(x)
        return x.squeeze(-1), att_weight

# %%
model = SAKTModel(n_skill=NUM_SKILLS, 
                  max_seq=MAX_SEQ, 
                  embed_dim=EMBED_SIZE, 
                  forward_expansion=1, 
                  enc_layers=1, 
                  heads=NUM_HEADS, dropout=DROPOUT)

n_params = get_num_params(model)
print(f"Current model has {n_params} parameters.")
# %%
