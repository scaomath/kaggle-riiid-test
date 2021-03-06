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
HOME = os.path.abspath(os.path.join('.', os.pardir))
print(HOME, '\n\n')
# HOME = "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
get_system()
from sakt import *
from iter_env import *

# from transformer_optimizers import *
# %%

'''
TO-DO:
features encoding:
1 how to address the problem with previous answers correctly not uniformly predicted
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
DEBUG_TEST_SIZE = 2500
DROPOUT = 0.1
SEED = 1127

get_seed(SEED)

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
        train_df = pd.read_parquet(os.path.join(DATA_DIR, 'cv2_train.parquet'),
                                columns=list(TRAIN_DTYPES.keys())).astype(TRAIN_DTYPES)
        valid_df = pd.read_parquet(os.path.join(DATA_DIR, 'cv2_valid.parquet'),
                                columns=list(TRAIN_DTYPES.keys())).astype(TRAIN_DTYPES)

    if DEBUG:
        train_df = train_df[:NROWS_TRAIN]
        valid_df = valid_df[:NROWS_VAL]

    with timer("Processing train"):
        train_group = preprocess(train_df)
        valid_group = preprocess(valid_df, train_flag=2)
else:
    with open(os.path.join(DATA_DIR, 'sakt_group_cv2.pickle'), 'rb') as f:
        group = pickle.load(f)
    train_group, valid_group = train_test_split(group, test_size = TEST_SIZE, random_state=SEED)


print(f"valid users: {len(valid_group.keys())}")
print(f"train users: {len(train_group.keys())}")
# %%

class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):
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
print(f"Length of the train loader is {len(train_dataloader)}")
#%%
sample_batch = next(iter(train_dataloader))
sample_batch[0].shape, sample_batch[1].shape, sample_batch[2].shape
# %%
'''
Debugging
'''
content_id, answered_correctly = train_group[train_group.keys()[0]]
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
# %% Merging questions

# questions_df = pd.read_csv(os.path.join(DATA_DIR, 'questions.csv'))
# questions_df['part'] = questions_df['part'].astype(np.int32)
# questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)

# train_debug = pd.merge(train_df, questions_df[['question_id', 'part']], 
#                  left_on = 'content_id', right_on = 'question_id', how = 'left')
# %% model

class FFN(nn.Module):
    def __init__(self, state_size = MAX_SEQ, 
                    forward_expansion = 1, 
                    bn_size=MAX_SEQ - 1, 
                    dropout=0.2):
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
                max_seq=MAX_SEQ, 
                embed_dim=EMBED_SIZE, 
                dropout = DROPOUT, 
                forward_expansion = 1, 
                enc_layers=1, 
                heads = NUM_HEADS):
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


class TestDataset(Dataset):
    def __init__(self, samples, test_df, n_skill, max_seq=100):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.n_skill, self.max_seq = n_skill, max_seq

    def __len__(self):
        return self.test_df.shape[0]
    
    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]
        
        user_id = test_info['user_id']
        target_id = test_info['content_id']
        
        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        
        if user_id in self.samples.index:
            content_id, answered_correctly = self.samples[user_id]
            
            seq_len = len(content_id)
            
            if seq_len >= self.max_seq:
                content_id_seq = content_id[-self.max_seq:]
                answered_correctly_seq = answered_correctly[-self.max_seq:]
            else:
                content_id_seq[-seq_len:] = content_id
                answered_correctly_seq[-seq_len:] = answered_correctly
                
        x = content_id_seq[1:].copy()
        x += (answered_correctly_seq[1:] == 1) * self.n_skill
        
        questions = np.append(content_id_seq[2:], [target_id])
        
        return x, questions
# %% Loading models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
model_file = MODEL_DIR+'sakt_seq_180_auc_0.7689.pth'
model = SAKTModel(n_skill=NUM_SKILLS, 
                  max_seq=MAX_SEQ, 
                  embed_dim=EMBED_SIZE, 
                  forward_expansion=1, 
                  enc_layers=1, 
                  heads=NUM_HEADS, 
                  dropout=DROPOUT)

n_params = get_num_params(model)
print(f"Current model has {n_params} parameters.")
        
model = model.to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
#%% Loading mock test set
with timer("Loading private simulated test set"):
    all_test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    all_test_df = all_test_df[:DEBUG_TEST_SIZE]

all_test_df['answer_correctly_true'] = all_test_df[TARGET]
# %% mock test
predicted = []
def set_predict(df):
    predicted.append(df)

# reload all user group for cv2
with timer('loading cv2'):
    with open(os.path.join(DATA_DIR, 'sakt_group_cv2.pickle'), 'rb') as f:
        group = pickle.load(f)

#%%
def iter_env_run(all_test_df, n_iter=1):
    '''
    Running mock test for n_iter iterations using tito's iter_env simulator and cv2_train user group.
    '''
    iter_test = Iter_Valid(all_test_df, max_user=1000)
    prev_test_df = None
    prev_group = None
    batch_user_ids = []

    # reload all user group for cv2
    with open(os.path.join(DATA_DIR, 'sakt_group_cv2.pickle'), 'rb') as f:
        group = pickle.load(f)

    for _ in range(n_iter):
    
        test_df, sample_prediction_df = next(iter_test)

        if prev_test_df is not None:
            prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
            prev_test_df = prev_test_df[prev_test_df.content_type_id == False]
            prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly']]\
                                    .groupby('user_id').apply(lambda r: (
                                        r['content_id'].values,
                                        r['answered_correctly'].values))
            for prev_user_id in prev_group.index:
                prev_group_content = prev_group[prev_user_id][0]
                prev_group_answered_correctly = prev_group[prev_user_id][1]
                if prev_user_id in group.index:
                    group[prev_user_id] = (np.append(group[prev_user_id][0], prev_group_content), 
                                        np.append(group[prev_user_id][1], prev_group_answered_correctly))
                else:
                    group[prev_user_id] = (prev_group_content, prev_group_answered_correctly)

                if len(group[prev_user_id][0]) > MAX_SEQ:
                    new_group_content = group[prev_user_id][0][-MAX_SEQ:]
                    new_group_answered_correctly = group[prev_user_id][1][-MAX_SEQ:]
                    group[prev_user_id] = (new_group_content, new_group_answered_correctly)

        prev_test_df = test_df.copy()
        test_df = test_df[test_df.content_type_id == False]

        batch_user_ids.append(test_df.user_id.unique())

        test_dataset = TestDataset(group, test_df, NUM_SKILLS, max_seq=MAX_SEQ)
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_df), shuffle=False)

        item = next(iter(test_dataloader))
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()

        with torch.no_grad():
            output, _ = model(x, target_id)

        output = torch.sigmoid(output)
        preds = output[:, -1]
        test_df['answered_correctly'] = preds.cpu().numpy()
        set_predict(test_df.loc[test_df['content_type_id'] == 0, 
                                ['row_id', 'answered_correctly']])

    return test_df, output, item, group, prev_group, batch_user_ids
# %%
# user_common = set(batch_user_ids[0])
# for k in range(1, len(batch_user_ids)):
#     user_common = user_common.intersection(set(batch_user_ids[k]))
# %% 
'''
Current set up, cv2_valid first 25k rows
first 4 batches common user_id: 143316232, 1089397940, 1140583044 (placeholder user?)
'''
print(group[1089397940])
#%% iter number 1
test_df, output, item, group_updated, _, _ = iter_env_run(all_test_df, n_iter=1)
u_idx_loc = test_df.index.get_loc(test_df[test_df.user_id==1089397940].index[0])
print(f"local index of user 1089397940: {u_idx_loc}", '\n')
print(test_df.iloc[u_idx_loc], '\n')
print(item[1][u_idx_loc, -12:]) # user 1089397940 first batch in example_test (question sequence)
print(item[0][u_idx_loc, -12:]) # user 1089397940 first batch in example_test: skill sequence = prev_content_id * (correct or not) + 13523
print(output[u_idx_loc, -12:].cpu().numpy(),'\n') # user 1089397940 probability prediction

print(group_updated[1089397940][0][:12]) # in the first iteration the length is only 11
print(group_updated[1089397940][1][:12])
#%% iter number 2
test_df, output, item, group_updated, _, _ = iter_env_run(all_test_df, n_iter=2)
u_idx_loc = test_df.index.get_loc(test_df[test_df.user_id==1089397940].index[0])
print(f"local index of user 1089397940: {u_idx_loc}", '\n')
print(test_df.iloc[u_idx_loc], '\n')
print(item[1][u_idx_loc, -12:]) # user 1089397940 2nd batch in example_test (question sequence)
print(item[0][u_idx_loc, -12:]) # user 1089397940 2nd batch in example_test: skill sequence = prev_content_id * (correct or not) + 13523
print(output[u_idx_loc, -12:].cpu().numpy(),'\n') # user 1089397940 probability prediction

print(group_updated[1089397940][0][:12]) # in the 2nd iteration the length is only 11
print(group_updated[1089397940][1][:12])
# %%
test_df, output, item, group_updated, _, _ = iter_env_run(all_test_df, n_iter=3)
u_idx_loc = test_df.index.get_loc(test_df[test_df.user_id==1089397940].index[0])
print(f"local index of user 1089397940: {u_idx_loc}", '\n')
print(test_df.iloc[u_idx_loc], '\n')
print(item[1][u_idx_loc, -12:]) # user 1089397940 3rd batch in example_test (question sequence)
print(item[0][u_idx_loc, -12:]) # user 1089397940 3rd batch in example_test: skill sequence = prev_content_id * (correct or not) + 13523
print(output[u_idx_loc, -12:].cpu().numpy(),'\n') # user 1089397940 probability prediction

print(group_updated[1089397940][0][:12]) # in the 3rd iteration the length is only 11
print(group_updated[1089397940][1][:12])
# %%
