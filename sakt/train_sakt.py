#%%
import gc
import os
import sys
sys.path.append("..") 
import pickle

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
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts,
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

which is an implementation of this paper: https://arxiv.org/pdf/1907.06837.pdf

To-do:

1. check the bias of the prediction

'''

TQDM_INT = 8 
HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
STAGE = "stage1"
FOLD = 1

TRAIN = True
PREPROCESS = False
EPOCHS = 60
LEARNING_RATE = 1e-3

#%%

if PREPROCESS:
    print("Preprocessing training...")
    train_df = pd.read_csv(DATA_DIR+'train.csv', usecols=[1, 2, 3, 4, 7], dtype=TRAIN_DTYPES)
    print(f'Loaded train.')
    print(TRAIN_DTYPES)
    train_df = train_df[train_df[CONTENT_TYPE_ID] == False]
    train_df = train_df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)
    group = train_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
        .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))
    train_group, valid_group = train_test_split(group, test_size=0.1)
else: # using preprocessed files
    print("Loading training...")
    train_df = pd.read_parquet(DATA_DIR+'cv2_train.parquet')
    valid_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    train_df = train_df[TRAIN_DTYPES.keys()]
    valid_df = valid_df[TRAIN_DTYPES.keys()]
    train_df = train_df[train_df[CONTENT_TYPE_ID] == False].reset_index(drop = True)
    valid_df = valid_df[valid_df[CONTENT_TYPE_ID] == False].reset_index(drop = True)

    print("valid:", valid_df.shape, "users:", valid_df[USER_ID].nunique())
    print("train:", train_df.shape, "users:", train_df[USER_ID].nunique())
    # Index by user_id
    valid_group = valid_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
        .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

    # Index by user_id
    train_group = train_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
        .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))



# skills = train_df[CONTENT_ID].unique()
n_skill = conf.NUM_SKILLS #len(skills) # len(skills) might not have all
print("Number of skills", n_skill)


print("valid by user:", len(valid_group))
print("train by user:", len(train_group))

# %%
train_dataset = SAKTDataset(train_group, n_skill, subset="train")
train_loader = DataLoader(train_dataset, 
                              batch_size=conf.BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=conf.WORKERS)

valid_dataset = SAKTDataset(valid_group, n_skill, subset="valid")
val_loader = DataLoader(valid_dataset, 
                              batch_size=conf.VAL_BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=conf.WORKERS)

item = train_dataset.__getitem__(5)

print("x", len(item[0]), item[0], '\n\n')
print("target_id", len(item[1]), item[1] , '\n\n')
print("label", len(item[2]), item[2])

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAKTModel(n_skill, embed_dim=conf.NUM_EMBED, num_heads=conf.NUM_HEADS)
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

        train_loss, train_acc, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_auc = valid_epoch(model, val_loader, criterion)
        scheduler.step(valid_loss)
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
