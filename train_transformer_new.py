#%%
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import datatable as dt
from time import time
from tqdm import tqdm
from typing import List
from collections import deque, Counter

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchsummary import summary

from sklearn.metrics import roc_auc_score
from utils import *
from dataset import *
from models import *
# %%
'''
To-do:
- Fix the same user_id prediction problem
- Check how the predicted probability relates to the original target in val set
'''
TRAIN_DTYPES = {
    # 'row_id': np.uint32,
    'timestamp': np.int64,
    'user_id': np.int32,
    'content_id': np.int16,
    'content_type_id': np.int8,
    'task_container_id': np.int16,
    'user_answer': np.int8,
    'answered_correctly': np.int8,
    'prior_question_elapsed_time': np.float32,
    'prior_question_had_explanation': 'boolean'
}

TEST_DTYPES = {
    # 'row_id': np.uint32,
    'timestamp': np.int64,
    'user_id': np.int32,
    'content_id': np.int16,
    'content_type_id': np.int8,
    'task_container_id': np.int16,
    'prior_question_elapsed_time': np.float32,
    'prior_question_had_explanation': 'boolean'
}


DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
FOLD = 1
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
LAST_N = 100 # this parameter denotes how many last seen content_ids I am going to consider <aka the max_seq_len or the window size>.
TAIL_N = 100 # used for validation set per user_id
FILLNA_VAL = 100 # fillers for the values (a unique value)
TQDM_INT = 15 # tqdm update interval
PAD = 0
BATCH_SIZE = 512
VAL_BATCH_SIZE = 2048

NROWS_TRAIN = 5_000_000
NROWS_VALID = 2_000_000
NROWS_TEST = 60

EPOCHS = 40

DEBUG = False
TRAIN = False
PREPROCESS = False


# %% Preparing train and validation set
df_questions = pd.read_csv(DATA_DIR+'questions.csv')
start = time()
if PREPROCESS:
    if DEBUG: 
        train_df = pd.read_csv(DATA_DIR+'train.csv', 
                            nrows=NROWS_TRAIN, 
                            dtype=TRAIN_DTYPES, 
                            usecols=TRAIN_DTYPES.keys())
    else:
        train_df = pd.read_csv(DATA_DIR+'train.csv', 
                            dtype=TRAIN_DTYPES, 
                            usecols=TRAIN_DTYPES.keys())
    train_df, valid_df = get_valid(train_df, n_tail=TAIL_N)
    train_df = preprocess(train_df, df_questions, train=True)
    valid_df = preprocess(valid_df, df_questions, train=False)
    d, user_id_to_idx = get_feats(train_df)
    d_val, user_id_to_idx = get_feats(valid_df)
    print(f"\nProcessing train and valid in {time()-start} seconds\n\n")
    skills = train_df['content_id'].unique()
    n_skill = 13523 # len(skills)
    print("Number of skills", n_skill, '\n\n')
    print(user_id_to_idx[115], d[0]) # user_id 115; I encourage you to match the same with the dataframes.

    with open(DATA_DIR+'d_train.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(DATA_DIR+'d_val.pickle', 'wb') as handle:
        pickle.dump(d_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

else: # to-do: save variable 
    with open(DATA_DIR+'d_train.pickle', 'rb') as handle:
        d = pickle.load(handle)
    with open(DATA_DIR+'d_val.pickle', 'rb') as handle:
        d_val = pickle.load(handle)


print(f"Readding train and validation data in {time()-start} seconds\n\n")





# %% data, model
dataset_train = Riiid(d=d)
dataset_val = Riiid(d=d_val)
# print(dataset[0]) # sample dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
sample = next(iter(DataLoader(dataset=dataset_train, 
                batch_size=1, collate_fn=collate_fn))) # dummy check
# createing the mdoel
model = TransformerModel(ninp=LAST_N, nhead=4, nhid=128, nlayers=3, dropout=0.3)
model = model.to(device)

losses = []
history = []
auc_max = -np.inf

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
lr = 1e-3 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, collate_fn=collate_fn)

dataset_val = DataLoader(dataset=dataset_val, batch_size=VAL_BATCH_SIZE, collate_fn=collate_fn, drop_last=True)

snapshot_path = "%s/fold%d/snapshots" % (MODEL_DIR, FOLD)
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)


#%%
if TRAIN:
    print("\n\nTraining...:")
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc, train_auc = train_epoch(model, dataset_train, optimizer, criterion)
        print(f"\n\n[Epoch {epoch}/{EPOCHS}]")
        print(f"Train: loss - {train_loss:.2f} acc - {train_acc:.4f} auc - {train_auc:.4f}")
        valid_loss, valid_acc, valid_auc = valid_epoch(model, dataset_val, criterion)
        print(f"\nValid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")
        lr = optimizer.param_groups[0]['lr']
        history.append({"epoch":epoch, "lr": lr, 
                        **{"train_auc": train_auc, "train_acc": train_acc}, 
                        **{"valid_auc": valid_auc, "valid_acc": valid_acc}})
        if valid_auc > auc_max:
            print(f"[Epoch {epoch}/{EPOCHS}] auc improved from {auc_max:.4f} to {valid_auc:.4f}") 
            print("saving model ...")
            auc_max = valid_auc
            torch.save(model.state_dict(), os.path.join(snapshot_path, f"model_CV_auc_{valid_auc:.4f}.pt"))
else:
    print("Loading state_dict...")
    model.load_state_dict(torch.load(os.path.join(snapshot_path, "model_best_epoch.pt"), map_location=device))
    model.eval()
    valid_loss, valid_acc, valid_auc = valid_epoch(model, dataset_val, criterion)
    print(f"\nValid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")

# %%
if not TRAIN and DEBUG:
    test_df = pd.read_csv(DATA_DIR+'train.csv', 
                        nrows=NROWS_TEST, 
                        dtype=TRAIN_DTYPES, 
                        usecols=TRAIN_DTYPES.keys())

    # test_df = pd.read_csv(DATA_DIR+'example_test.csv', 
    #                     nrows=NROWS_TEST, 
    #                     dtype=TEST_DTYPES, 
    #                     usecols=TEST_DTYPES.keys())

    test_df = preprocess(test_df, df_questions, train=False)
    d_test = {}
    user_id_to_idx = {}
    grp = test_df.groupby("user_id").tail(100)
    grp_user = grp.groupby("user_id")
    num_user_id_grp = len(grp_user)

    for idx, row in grp_user.agg({
        "content_id":list, "task_container_id":list, 
        "part_id":list, "prior_question_elapsed_time":list
        }).reset_index().iterrows():
        
        print('\n\n',idx, row)
        print(row["content_id"])
        
        # here we make a split whether a user has more than equal to 100 entries or less than that
        # if it's less than LAST_N, then we need to PAD it using the PAD token defined as 0 by me in this cell block
        # also, padded will be True where we have done padding obviously, rest places it's False.
        if len(row["content_id"]) >= 100:
            d_test[idx] = {
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
            d_test[idx] = {
                "user_id": row["user_id"],
                "content_id" : deque(row["content_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                "task_container_id" : deque(row["task_container_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                "part_id": deque(row["part_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                "padded" : deque([False]*len(row["content_id"]) + [True]*(100-len(row["content_id"])), maxlen=LAST_N)
            }
        user_id_to_idx[row["user_id"]] = idx

    dataset_test = RiiidTest(d=d_test)
    # dataset_test = Riiid(d=d_test)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=VAL_BATCH_SIZE, 
                                 collate_fn=collate_fn_test, shuffle=False, drop_last=False)
    
    output_all = []
    for idx, batch in enumerate(test_dataloader):
        content_id, _, part_id, prior_question_elapsed_time, mask = batch
        target_id = batch[1].to(device).long()

        content_id = Variable(content_id.cuda())
        part_id = Variable(part_id.cuda())
        prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
        mask = Variable(mask.cuda())

        with torch.no_grad():
            output = model(content_id, part_id, prior_question_elapsed_time, mask)
        pred_probs = torch.softmax(output[~mask], dim=1)
        # pred = (output_prob >= 0.50)
        pred = torch.argmax(pred_probs, dim=1)
        output_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy())
    test_df['answered_correctly'] = output_all
# %%
