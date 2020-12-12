#%%
CUDA_LAUNCH_BLOCKING="1"
import gc, sys, os
sys.path.append("..") 
from tqdm import tqdm
from time import time
import pickle
import numpy as np
import pandas as pd
import datatable as dt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
DEFAULT_FIG_WIDTH = 20
sns.set_context("paper", font_scale=1.2) 

from sakt import *
from utils import *

print('Python     : ' + sys.version.split('\n')[0])
print('Numpy      : ' + np.__version__)
print('Pandas     : ' + pd.__version__)
print('PyTorch    : ' + torch.__version__)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device     : {DEVICE}')
'''
https://www.kaggle.com/mpware/sakt-fork
Self-Attentive model for Knowledge Tracing model (SAKT)
This is a fork of: https://www.kaggle.com/wangsg/a-self-attentive-model-for-knowledge-tracing from @wangsg

Which is an implementation of this paper: https://arxiv.org/pdf/1907.06837.pdf

With the following improvements:

Pytorch random fixed to be reproductible
Random sequence added during training
torch.sigmoid added/fixed to train loop
Training plot
Train/Valid simple split to save best model
'''

HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'


STAGE = "stage1"
TQDM_INT = 8 
FOLD = 1

TRAIN = True
PREPROCESS = True
EPOCHS = 60
LEARNING_RATE = 1e-3
NROWS_TRAIN = 10_000_000


#%%

print("Loading training...")
start = time()
data_df = pd.read_csv(DATA_DIR+'train.csv', 
                          usecols=[1, 2, 3, 4, 7, 8, 9], 
                          nrows=NROWS_TRAIN,
                          dtype=TRAIN_DTYPES)
print(f"Loaded train.csv in {time()-start} seconds.\n\n")

#%%
print("Processing training...")
train_df = data_df.copy()
start = time()
train_df[PRIOR_QUESTION_TIME].fillna(conf.FILLNA_VAL, inplace=True) 
    # FILLNA_VAL different than all current values
train_df[PRIOR_QUESTION_TIME] = round(train_df[PRIOR_QUESTION_TIME] / TIME_SCALING)
# train_df[PRIOR_QUESTION_TIME] = train_df[PRIOR_QUESTION_TIME].replace(np.inf, 301).astype(np.int16) 
train_df[PRIOR_QUESTION_EXPLAIN] = train_df[PRIOR_QUESTION_EXPLAIN].astype(np.float16).fillna(2).astype(np.int8)

train_df = train_df[train_df[CONTENT_TYPE_ID] == False]
train_df = train_df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)

group = train_df[[USER_ID, CONTENT_ID, PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN, TARGET]]\
    .groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, 
                      r[PRIOR_QUESTION_TIME].values,
                      r[PRIOR_QUESTION_EXPLAIN].values,
                      r[TARGET].values))

# group = train_df[[USER_ID, CONTENT_ID, TARGET]]\
#     .groupby(USER_ID)\
#     .apply(lambda r: (r[CONTENT_ID].values, 
#                       r[TARGET].values))
train_group, valid_group = train_test_split(group, test_size=0.1)
print(f"Prcocessed train.csv in {time()-start} seconds.\n\n")

# skills = train_df[CONTENT_ID].unique()
n_skill = conf.NUM_SKILLS #len(skills) # len(skills) might not have all
print("Number of skills", n_skill)


print(f"Valid by user:  {len(valid_group)}")
print(f"Train by user:  {len(train_group)}\n\n")

# %%
train_dataset = SAKTDatasetNew(train_group, n_skill, subset="train")
valid_dataset = SAKTDatasetNew(valid_group, n_skill, subset="valid")
# train_dataset = SAKTDataset(train_group, n_skill, subset="train")
# valid_dataset = SAKTDataset(valid_group, n_skill, subset="valid")



train_loader = DataLoader(train_dataset, 
                              batch_size=conf.BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=conf.WORKERS)
val_loader = DataLoader(valid_dataset, 
                              batch_size=conf.VAL_BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=conf.WORKERS)

item = train_dataset.__getitem__(10)

print("x", len(item[0]), item[0], '\n\n')
print("target_id", len(item[1]), item[1] , '\n\n')
print("label", len(item[2]), item[2], '\n\n')
print("prior question time", len(item[3]), item[3],  '\n\n')
print("prior question explained", len(item[4]), item[4])

# %%

class SAKTModelNew(nn.Module):
    def __init__(self, n_skill, max_seq=conf.MAX_SEQ, embed_dim=conf.NUM_EMBED, num_heads=conf.NUM_HEADS):
        super(SAKTModelNew, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.pqt_embedding = nn.Embedding(conf.NUM_TIME+1, embed_dim) # embedding of prior question time
        self.pa_embedding = nn.Embedding(3+1, embed_dim) # embedding of priori question answered

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.scaling = conf.SCALING
    
    def forward(self, x, question_ids, prior_question_time=None, prior_question_explain=None):
        device = x.device        
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        pq_x = self.pqt_embedding(prior_question_time)
        pa_x = self.pa_embedding(prior_question_explain)

        x = x + pos_x + pq_x + pa_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output) # original
        # x = self.layer_normal(x) + att_output # modified, seems not changing much
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)

        return x.squeeze(-1), att_weight


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAKTModelNew(n_skill, embed_dim=conf.NUM_EMBED, num_heads=conf.NUM_HEADS)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=0.0001)
# optimizer = HNAGOptimizer(model.parameters(), lr=1e-3) 
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)
num_params = get_num_params(model)
print('\n\n')
print(f"# heads  : {conf.NUM_HEADS}")
print(f"# embed  : {conf.NUM_EMBED}")
print(f"# params : {num_params}")
# %%
train_loss = []
num_corrects = 0
num_total = 0
labels = []
outs = []

len_dataset = len(train_loader)

for idx, item in enumerate(train_loader): 
    x = item[0].to(device).long()
    target_id = item[1].to(device).long()
    prior_q_time = item[3].to(device).long()
    priori_q_explain = item[4].to(device).long()
    label = item[2].to(device).float()
    break


# %% 
'''
Bug: np.float16 and negative values in tensor
will trigger CUDA error 59: Device-side assert triggered
'''
max_seq = conf.MAX_SEQ
embed_dim = conf.NUM_EMBED
embedding = nn.Embedding(2*n_skill+1, embed_dim).to(device)
pos_embedding = nn.Embedding(max_seq-1, embed_dim).to(device)
e_embedding = nn.Embedding(n_skill+1, embed_dim).to(device)
pqt_embedding = nn.Embedding(conf.NUM_TIME+1, embed_dim).to(device) # embedding of prior question time
pa_embedding = nn.Embedding(3+1, embed_dim).to(device) # embedding of priori question answered
# %%
device = x.device        
x = embedding(x)
pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
pos_x = pos_embedding(pos_id)

pq_x = pqt_embedding(prior_q_time)
pa_x = pa_embedding(priori_q_explain)
print(x.size(), pos_x.size(), pq_x.size(), pa_x.size())
# %%