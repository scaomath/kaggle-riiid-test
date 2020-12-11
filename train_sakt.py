#%%
import gc, sys, os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

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
NUM_HEADS = 10
NUM_EMBED = 120
MAX_SEQ = 100
N_TAIL = 100

TRAIN = True
EPOCHS = 40


#%%
train_df = pd.read_parquet(DATA_DIR+'cv2_train.parquet')
valid_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
train_df = train_df[TRAIN_DTYPES.keys()]
valid_df = valid_df[TRAIN_DTYPES.keys()]

train_df = train_df[train_df[CONTENT_TYPE_ID] == False].reset_index(drop = True)
valid_df = valid_df[valid_df[CONTENT_TYPE_ID] == False].reset_index(drop = True)

print("valid:", valid_df.shape, "users:", valid_df[USER_ID].nunique())
print("train:", train_df.shape, "users:", train_df[USER_ID].nunique())
# print(train_df[train_df[USER_ID] == 2147482216].head(10))
# print(valid_df[valid_df[USER_ID] == 115].head(10))

skills = train_df[CONTENT_ID].unique()
n_skill = conf.NUM_SKILLS #len(skills) # len(skills) might not have all
print("Number of skills", n_skill)

# Index by user_id
valid_group = valid_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

# Index by user_id
train_group = train_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)
num_params = get_num_params(model)
print(f"Num of params: {num_params}")
# %%

if TRAIN:
    losses = []
    history = []
    auc_max = -np.inf

    print("\n\nTraining...:")
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_auc = valid_epoch(model, val_loader, criterion)

        print(f"\n\n[Epoch {epoch}/{EPOCHS}]")
        print(f"\nTrain: loss - {train_loss:.2f} acc - {train_acc:.4f} auc - {train_auc:.4f}")
        print(f"\nValid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")
        lr = optimizer.param_groups[0]['lr']
        history.append({"epoch":epoch, "lr": lr, 
                        **{"train_auc": train_auc, "train_acc": train_acc}, 
                        **{"valid_auc": valid_auc, "valid_acc": valid_acc}})
        
        if valid_auc > auc_max:
            print(f"[Epoch {epoch}/{EPOCHS}] auc improved from {auc_max:.4f} to {valid_auc:.4f}") 
            print("saving model ...")
            auc_max = valid_auc
            torch.save(model.state_dict(), 
            os.path.join(MODEL_DIR, f"sakt_head_{conf.NUM_HEADS}_embed_{conf.NUM_EMBED}_auc_{valid_auc:.4f}.pt"))
        
        with open(DATA_DIR+f'history.pickle', 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if epoch % 10 == 0:
            conf.BATCH_SIZE *= 2
            train_loader = DataLoader(train_dataset, 
                                      batch_size=conf.BATCH_SIZE, 
                                      shuffle=True, 
                                      num_workers=conf.WORKERS)
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