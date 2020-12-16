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
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts, CyclicLR,
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
HOME = "/home/scao/Documents/kaggle-riiid-test/"

MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
get_system()

from sakt import *
from transformer_optimizers import *

# print('Python     : ' + sys.version.split('\n')[0])
# print('Numpy      : ' + np.__version__)
# print('Pandas     : ' + pd.__version__)
# print('PyTorch    : ' + torch.__version__)
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'Device     : {DEVICE}')
'''
https://www.kaggle.com/mpware/sakt-fork
Self-Attentive model for Knowledge Tracing model (SAKT)
This is a fork of: https://www.kaggle.com/wangsg/a-self-attentive-model-for-knowledge-tracing from @wangsg
an implementation of this paper: https://arxiv.org/pdf/1907.06837.pdf

Version notes:

- Increasing seq_len for sakt new model does not work well

- Testing performance of the following configs for both attention layers and after concat att outputs:
1. bn(relu(f(x))) + x, epoch 1 auc 0.7372, epoch 3 auc 0.7422, epoch 5 auc 0.7445 
2. bn(relu(f(x)) + x), epoch 1 auc 0.7379, epoch 3 auc 0.7413, epoch 5 auc 0.7443
3. bn(f(x)) + x: epoch 0 auc 0.7369, epoch 2 auc 0.7415, epoch 4 auc 0.7448
4. bn(f(x) + x): epoch 0 auc 0.7380, epoch 2 auc 0.7418, epoch 4 auc 0.7445

- Testing a new model: two attention layers stacked using question id as key
epoch 0 auc 0.7256, epoch 2 auc 0.7399, epoch 4 auc 0.7424
Later epoch does not perform well

- Testing a warm-up scheduler with 10 warm-up epochs for a model with two attention layers


'''

TQDM_INT = 8 

STAGE = "stage1"
FOLD = 1
DATE_STR = get_date()
TRAIN = True
DEBUG = False
ROLLOUT = True
EPOCHS = 60 if ROLLOUT else 13
LEARNING_RATE = 1e-3
NROWS_TRAIN = 5_000_000
PREPROCESS = 2


class conf:
    METRIC_ = "max"
    FILLNA_VAL = 14_000 # for prior question elapsed time, rounded average in train
    TQDM_INT = 8
    WORKERS = 8 # 0
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    VAL_BATCH_SIZE = 4096
    NUM_EMBED = 512
    NUM_HEADS = 8
    NUM_SKILLS = 13523 # len(skills)
    NUM_TIME = 300 # when scaled by 1000 and round, priori question time's unique values
    MAX_SEQ = 150
    SCALING = 1 # scaling before sigmoid
    PATIENCE = 8 # overfit patience
    SAVING_THRESHOLD = 0.754 # the threshold for auc to save a model

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
#%%
if PREPROCESS == 1:
    print("\nLoading training csv...")
    start = time()
    if DEBUG:
        data_df = pd.read_csv(DATA_DIR+'train.csv', 
                            usecols=[1, 2, 3, 4, 7, 8, 9], 
                            nrows=NROWS_TRAIN,
                            dtype=TRAIN_DTYPES)
    else:
        data_df = pd.read_csv(DATA_DIR+'train.csv', usecols=TRAIN_COLS_NEW, dtype=TRAIN_DTYPES)
    print(f"Loaded train.csv in {time()-start} seconds.\n\n")

    print("Processing training...")
    train_df = data_df.copy()
    start = time()
    train_df[PRIOR_QUESTION_TIME].fillna(conf.FILLNA_VAL, inplace=True) 
        # FILLNA_VAL different than all current values
    train_df[PRIOR_QUESTION_TIME] = round(train_df[PRIOR_QUESTION_TIME] / TIME_SCALING).astype(np.int16)
    train_df[PRIOR_QUESTION_EXPLAIN] = train_df[PRIOR_QUESTION_EXPLAIN].astype(np.float16).fillna(0).astype(np.int8)

    train_df = train_df[train_df[CONTENT_TYPE_ID] == False]
    train_df = train_df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)

    group = train_df[[USER_ID, CONTENT_ID, PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN, TARGET]]\
        .groupby(USER_ID)\
        .apply(lambda r: (r[CONTENT_ID].values, 
                        r[PRIOR_QUESTION_TIME].values,
                        r[PRIOR_QUESTION_EXPLAIN].values,
                        r[TARGET].values))
    with open(DATA_DIR+'sakt_data_new.pickle', 'wb') as f:
        pickle.dump(group, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Prcocessed train.csv in {time()-start} seconds.\n\n")
    train_group, valid_group = train_test_split(group, test_size=0.1)
elif PREPROCESS == 2:
    print("Loading training parquet...")
    start = time()
    train_df = pd.read_parquet(DATA_DIR+'cv3_train.parquet')
    valid_df = pd.read_parquet(DATA_DIR+'cv3_valid.parquet')
    train_df = train_df[TRAIN_COLS_NEW]
    valid_df = valid_df[TRAIN_COLS_NEW]
    train_group = preprocess_sakt(train_df)
    valid_group = preprocess_sakt(valid_df)
    print(f"Prcocessed train.parquet in {time()-start} seconds.\n\n")
else:
    print("\nLoading preprocessed file...")
    with open(DATA_DIR+'sakt_data_new.pickle', 'rb') as f:
        group = pickle.load(f)
    train_group, valid_group = train_test_split(group, test_size=0.1)

# skills = train_df[CONTENT_ID].unique()
n_skill = conf.NUM_SKILLS #len(skills) # len(skills) might not have all
print("Number of skills", n_skill)

print(f"Valid by user:  {len(valid_group)}")
print(f"Train by user:  {len(train_group)}\n\n")

# %%
train_dataset = SAKTDatasetNew(train_group, n_skill, subset="train")
train_loader = DataLoader(train_dataset, 
                              batch_size=conf.BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=conf.WORKERS)

valid_dataset = SAKTDatasetNew(valid_group, n_skill, subset="valid")
val_loader = DataLoader(valid_dataset, 
                              batch_size=conf.VAL_BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=conf.WORKERS)

item = train_dataset.__getitem__(5)

print("x", len(item[0]), item[0], '\n\n')
print("target_id", len(item[1]), item[1] , '\n\n')
print("label", len(item[2]), item[2], '\n\n')
print("prior question time", len(item[3]), item[3],  '\n\n')
print("prior question explained", len(item[4]), item[4])

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAKTModelNew(n_skill, embed_dim=conf.NUM_EMBED, num_heads=conf.NUM_HEADS)
# model = SAKTMulti(n_skill, embed_dim=conf.NUM_EMBED, num_heads=conf.NUM_HEADS)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=conf.PATIENCE-1, threshold=1e-4,verbose=1)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=LEARNING_RATE*1e-2,verbose=1)
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, t_total=EPOCHS)
# scheduler = CyclicLR(optimizer, base_lr=1e-1*LEARNING_RATE, max_lr=LEARNING_RATE, step_size_up=5,mode="triangular2", verbose=1)
# optimizer = HNAGOptimizer(model.parameters(), lr=1e-3) 
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)
num_params = get_num_params(model)
print(f'\n\nModel: {model.__name__}')
print(f"# heads  : {conf.NUM_HEADS}")
print(f"# embed  : {conf.NUM_EMBED}")
print(f"seq len  : {conf.MAX_SEQ}")
print(f"# params : {num_params}")
# %%
if TRAIN:
    model, history = run_train_new(model, train_loader, val_loader, 
                               optimizer, criterion, scheduler=scheduler,
                               epochs=EPOCHS, device="cuda", conf=conf)
    with open(DATA_DIR+f'history_{DATE_STR}.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    tqdm.write("\nLoading state_dict...\n")
    model_file = find_sakt_model(model_type='saktnew')
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    valid_loss, valid_acc, valid_auc = valid_epoch(model, val_loader, criterion)
    print(f"\nValid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")

'''
Mock test in iter_env_sakt_new
'''
