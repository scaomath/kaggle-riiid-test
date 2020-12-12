#%%
import gc
import os
import sys
sys.path.append("..") 
import pickle
from time import time

import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts, CyclicLR,
                                      ReduceLROnPlateau)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sns.set()
DEFAULT_FIG_WIDTH = 20
sns.set_context("paper", font_scale=1.2) 

from utils import *

from sakt import *

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

TQDM_INT = 8 
HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
STAGE = "stage1"
FOLD = 1

TRAIN = True
PREPROCESS = False
DEBUG = False
EPOCHS = 80
LEARNING_RATE = 1e-3
NROWS_TRAIN = 10_000_000

#%%
if PREPROCESS:
    print("\nLoading training...")
    start = time()
    if DEBUG:
        data_df = pd.read_csv(DATA_DIR+'train.csv', 
                            usecols=[1, 2, 3, 4, 7, 8, 9], 
                            nrows=NROWS_TRAIN,
                            dtype=TRAIN_DTYPES)
    else:
        data_df = pd.read_csv(DATA_DIR+'train.csv', usecols=[1, 2, 3, 4, 7, 8, 9], dtype=TRAIN_DTYPES)
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
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-5)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3)
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
if TRAIN:
    model, history = run_train_new(model, train_loader, val_loader, 
                               optimizer, scheduler, criterion, 
                               epochs=EPOCHS, device="cuda")
    with open(DATA_DIR+f'history.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    tqdm.write("\nLoading state_dict...\n")
    model_file = find_sakt_model()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    valid_loss, valid_acc, valid_auc = valid_epoch(model, val_loader, criterion)
    print(f"\nValid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")

'''
Mock test in iter_env_sakt_new
'''
