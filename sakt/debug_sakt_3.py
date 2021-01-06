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

#%%
'''
TO-DO:
Adding the train_loader split by the average timestamp gap to mimic the test scenario
'''

DEBUG = True
TRAIN = False
PREPROCESS = True

TEST_SIZE = 0.05

NUM_SKILLS = 13523 # number of problems
MAX_SEQ = 180
ACCEPTED_USER_CONTENT_SIZE = 5
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
TIMESTAMP_GAP = 20_000

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
    NROWS_TRAIN = 1_000_000
    NROWS_VAL = 100_000
else:
    NROWS_TEST = 250_000
    NROWS_TRAIN = 50_000_000
    NROWS_VAL = 2_000_000
# %%
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
        train_group = preprocess_final(train_df)
        valid_group = preprocess_final(valid_df, train_flag=2)
else:
    with open(os.path.join(DATA_DIR, 'sakt_group_cv2.pickle'), 'rb') as f:
        group = pickle.load(f)
    train_group, valid_group = train_test_split(group, test_size = TEST_SIZE, random_state=SEED)


print(f"valid users: {len(valid_group.keys())}")
print(f"train users: {len(train_group.keys())}")
# %%
class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, 
                        max_seq=MAX_SEQ, 
                        min_seq=ACCEPTED_USER_CONTENT_SIZE,
                        recent_seq=RECENT_SIZE,
                        gap=TIMESTAMP_GAP):
        super(SAKTDataset, self).__init__()
        self.samples = {}
        self.n_skill = n_skill
        self.max_seq = max_seq
        self.min_seq = min_seq
        self.recent_seq = recent_seq
        self.gap = gap
        
        self.user_ids = []
        for i, user_id in enumerate(group.index):
            content_id, answered_correctly, timestamp = group[user_id]

            if len(content_id) >= self.min_seq:

                if len(content_id) > self.max_seq:
                    total_questions = len(content_id)
                    num_seq_user = total_questions // self.max_seq
                    for seq in range(num_seq_user):
                        index = f"{user_id}_{seq}"
                        self.user_ids.append(index)
                        start = seq * self.max_seq
                        end = (seq + 1) * self.max_seq
                        # timestamp[end-1] the original last entry's timestamp
                        # the difference with previous entry should be bigger than a threshold
                        # otherwise they may be in the same test_df batch
                        idx_same_bundle = []
                        while timestamp[end-1] - timestamp[end-2] < self.gap and end-start >= self.min_seq:
                            if timestamp[end-1] == timestamp[end-2]:
                                idx_same_bundle.append(end-1)
                            end -= 1     
                        self.samples[index] = (content_id[start:end], answered_correctly[start:end])
                        
                        if idx_same_bundle and end-1-start >= self.min_seq: 
                            # seeing multiple questions at a time, this list is not empty
                            for i, idx in enumerate(idx_same_bundle):
                                index = f"{user_id}_{seq}_{i}"
                                self.user_ids.append(index)
                                self.samples[index] = (np.r_[content_id[start:end-1], content_id[idx]], 
                                                       np.r_[answered_correctly[start:end-1], content_id[idx]])

                    content_id_last = content_id[end:]
                    timestamp_last = timestamp[end:]
                    end = len(content_id_last)
                    if end >= self.min_seq:
                        while timestamp_last[end-1] - timestamp_last[end-2] < self.gap and end >= self.min_seq:
                            end -= 1
                        index = f"{user_id}_{num_seq_user + 1}"
                        self.user_ids.append(index)
                        self.samples[index] = (content_id_last[:end], content_id_last[:end])
                else:
                    index = f'{user_id}'
                    end = len(timestamp)
                    # last time stamp diff should be bigger than a threshold
                    while timestamp[end-1] - timestamp[end-2] < self.gap and end >= self.min_seq:
                        end -= 1
                    self.user_ids.append(index)
                    self.samples[index] = (content_id[:end], answered_correctly[:end])
            '''
            New: adding a shifted sequence
            '''
            if len(content_id) >= self.recent_seq: #
                for i in range(1, self.recent_seq//2): # adding a shifted sequence
                    '''
                    generating much much more sequences by truncating
                    '''
                    content_id_truncated_end = content_id[:-i]
                    answered_correctly_truncated_end = answered_correctly[:-i]
                    if len(content_id_truncated_end) >= self.min_seq:
                        if len(content_id_truncated_end) > self.max_seq:
                            total_questions_2 = len(content_id_truncated_end)
                            num_seq_user = total_questions_2 // self.max_seq
                            for seq in range(num_seq_user):
                                index = f"{user_id}_{seq}_{i}_2"
                                self.user_ids.append(index)
                                start = seq * self.max_seq
                                end = (seq + 1) * self.max_seq
                                self.samples[index] = (content_id_truncated_end[start:end], 
                                                    answered_correctly_truncated_end[start:end])
                            if len(content_id_truncated_end[end:]) >= self.min_seq:
                                index = f"{user_id}_{num_seq_user + 1}_{i}_2"
                                self.user_ids.append(index)
                                self.samples[index] = (content_id_truncated_end[end:], 
                                                    answered_correctly_truncated_end[end:])
                        else:
                            index = f'{user_id}_{i}_2'
                            self.user_ids.append(index)
                            self.samples[index] = (content_id_truncated_end, 
                                                   answered_correctly_truncated_end)
                
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
            
        target_id = content_id_seq[1:] # question including the current one
        label = answered_correctly_seq[1:]
        
        x = content_id_seq[:-1].copy() # question till the previous one
        # encoded answers till the previous one
        x += (answered_correctly_seq[:-1] == 1) * self.n_skill
        
        return x, target_id, label