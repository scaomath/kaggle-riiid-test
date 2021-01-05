#%% final model of sakt
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
# %%
DEBUG = True
TRAIN = True
PREPROCESS = False

TEST_SIZE = 0.05 # un-used

NUM_SKILLS = 13523 # number of problems
MAX_SEQ = 180
ACCEPTED_USER_CONTENT_SIZE = 4
EMBED_SIZE = 128
RECENT_SIZE = 20 # recent data in the training loop
NUM_HEADS = 8
BATCH_SIZE = 64
VAL_BATCH_SIZE = 2048
DEBUG_TEST_SIZE = 10_000
TEST_SIZE = 25_000
DROPOUT = 0.1
SEED = 1127

get_seed(SEED)

#%%
with timer("Loading train and valid"):
    # with open(os.path.join(DATA_DIR, 'sakt_group_cv2.pickle'), 'rb') as f:
    #         train_group = pickle.load(f)
    # with open(os.path.join(DATA_DIR, 'sakt_group_cv2_valid.pickle'), 'rb') as f:
    #         valid_group = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'sakt_group.pickle'), 'rb') as f:
        group = pickle.load(f)
    train_group, valid_group = train_test_split(group, test_size = TEST_SIZE, random_state=SEED)

# %%
train_dataset = SAKTDataset(train_group, n_skill=NUM_SKILLS, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, 
                        batch_size=BATCH_SIZE, 
                        # shuffle=True, 
                        shuffle=False,
                        drop_last=True)


val_dataset = SAKTDataset(valid_group, n_skill=NUM_SKILLS, max_seq=MAX_SEQ)
val_dataloader = DataLoader(val_dataset, 
                        batch_size=VAL_BATCH_SIZE, 
                        shuffle=False)
print(len(train_dataloader))
# %%
sample_batch = next(iter(train_dataloader))
print(sample_batch[0].shape, sample_batch[1].shape, sample_batch[2].shape)

#%%
model = SAKTModel(NUM_SKILLS, 
                  max_seq=MAX_SEQ, 
                  embed_dim=EMBED_SIZE, 
                  forward_expansion=1, 
                  enc_layers=1, 
                  heads=NUM_HEADS, 
                  dropout=DROPOUT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_params = get_num_params(model)
print(f"Current model has {n_params} parameters running on {device}.")

# %%
def run_train(lr=1e-3, epochs=10, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                                    steps_per_epoch=len(train_dataloader), 
                                                    epochs=epochs)
    model.to(device)
    criterion.to(device)
    best_auc = 0.0
    for epoch in range(epochs):
        train_epoch(model, train_dataloader, optimizer, criterion, scheduler, device)
        # train_epoch_weighted(model, train_dataloader, optimizer, criterion, scheduler, device)
        val_loss, val_acc, val_auc = valid_epoch(model, val_dataloader, criterion, device)
        tqdm.write(f"\nEpoch - {epoch+1} val_loss - {val_loss:.3f} acc - {val_acc:.3f} auc - {val_auc:.3f}")
        if best_auc < val_auc:
            print(f'Epoch - {epoch+1} best model with val auc: {val_auc}')
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 
            f'sakt_seq_{MAX_SEQ}_auc_{val_auc:.4f}.pt'))
# %%
LR = 1e-3
EPOCHS = 10
run_train(lr=LR, epochs=EPOCHS)
# %%
with timer("Loading private simulated test set"):
    all_test_df = pd.read_parquet(os.path.join(DATA_DIR,'cv5_valid.parquet'))
    all_test_df = all_test_df[:DEBUG_TEST_SIZE]

# %%

iter_test = Iter_Valid(all_test_df, max_user=1000)
prev_test_df = None
prev_group = None
predicted = []

def set_predict(df):
    predicted.append(df)
# %%
with tqdm(total=len(all_test_df)) as pbar:
    for idx, (test_df, sample_prediction_df) in enumerate(iter_test):
        
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
        pbar.set_description(f'Current batch test length: {len(test_df)}')
        pbar.update(len(test_df))

#%%
y_true = all_test_df[all_test_df.content_type_id == 0].answered_correctly
y_pred = pd.concat(predicted).answered_correctly
print(f'\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))
# %%
