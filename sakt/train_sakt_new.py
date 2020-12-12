#%%
import gc, sys, os
sys.path.append("..") 

# import os.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

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

TQDM_INT = 8 
HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
STAGE = "stage1"
FOLD = 1

TRAIN = True
PREPROCESS = True
DEBUG = False
EPOCHS = 60
LEARNING_RATE = 1e-3
NROWS_TRAIN = 10_000_000


#%%

print("Loading training...")
start = time()
if DEBUG:
    data_df = pd.read_csv(DATA_DIR+'train.csv', 
                          usecols=[1, 2, 3, 4, 7, 8, 9], 
                          nrows=NROWS_TRAIN,
                          dtype=TRAIN_DTYPES)
else:
    data_df = pd.read_csv(DATA_DIR+'train.csv', usecols=[1, 2, 3, 4, 7, 8, 9], dtype=TRAIN_DTYPES)
print(f"Loaded train.csv in {time()-start} seconds.\n\n")

#%%
print("Processing training...")
train_df = data_df.copy()
start = time()
train_df[PRIOR_QUESTION_TIME].fillna(conf.FILLNA_VAL, inplace=True) 
    # FILLNA_VAL different than all current values
train_df[PRIOR_QUESTION_TIME] = round(train_df[PRIOR_QUESTION_TIME] / TIME_SCALING)
train_df[PRIOR_QUESTION_TIME] = train_df[PRIOR_QUESTION_TIME].replace(np.inf, conf.FILLNA_VAL).astype(np.float16) 
train_df[PRIOR_QUESTION_EXPLAIN] = train_df[PRIOR_QUESTION_EXPLAIN].astype(np.float16).fillna(-1).astype(np.int8)

train_df = train_df[train_df[CONTENT_TYPE_ID] == False]
train_df = train_df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)

group = train_df[[USER_ID, CONTENT_ID, PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN, TARGET]]\
    .groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, 
                      r[PRIOR_QUESTION_TIME].values,
                      r[PRIOR_QUESTION_EXPLAIN].values,
                      r[TARGET].values))
train_group, valid_group = train_test_split(group, test_size=0.1)
print(f"Prcocessed train.csv in {time()-start} seconds.\n\n")

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

if TRAIN:
    losses = []
    history = []
    auc_max = -np.inf
    over_fit = 0

    print("\n\nTraining...:")
    for epoch in range(1, EPOCHS+1):

        if epoch == 10:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1*LEARNING_RATE)
        elif epoch == 20:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.5*LEARNING_RATE)
        elif epoch == 25:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1*LEARNING_RATE)
        elif epoch == 35:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.5*LEARNING_RATE)
        elif epoch == 40:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1*LEARNING_RATE)

        train_loss, train_acc, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_auc = valid_epoch(model, val_loader, criterion)

        print(f"\n\n[Epoch {epoch}/{EPOCHS}]")
        print(f"Train: loss - {train_loss:.2f} acc - {train_acc:.4f} auc - {train_auc:.4f}")
        print(f"Valid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")
        lr = optimizer.param_groups[0]['lr']
        history.append({"epoch":epoch, "lr": lr, 
                        **{"train_auc": train_auc, "train_acc": train_acc}, 
                        **{"valid_auc": valid_auc, "valid_acc": valid_acc}})
        
        if valid_auc > auc_max:
            print(f"Epoch {epoch}: auc improved from {auc_max:.4f} to {valid_auc:.4f}") 
            auc_max = valid_auc
            over_fit = 0
            if valid_auc > 0.75:
                torch.save(model.state_dict(), 
                os.path.join(MODEL_DIR, f"sakt_head_{conf.NUM_HEADS}_embed_{conf.NUM_EMBED}_auc_{valid_auc:.4f}.pt"))
                print("Saving model ...\n\n")
        else:
            over_fit += 1
        
        if over_fit >= 5:
            print(f"Early stop epoch at {epoch}")
            break

        # if epoch % 10 == 0:
        #     conf.BATCH_SIZE *= 2
        #     train_loader = DataLoader(train_dataset, 
        #                               batch_size=conf.BATCH_SIZE, 
        #                               shuffle=True, 
        #                               num_workers=conf.WORKERS)
    # model, history = run_train(model, train_loader, val_loader, 
    #                            optimizer, scheduler, criterion, 
    #                            epochs=EPOCHS, device="cuda")
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
Mock test in iter_env_sakt
'''
# %%
