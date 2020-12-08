import datatable as dt
from time import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from collections import deque, Counter

'''
Download Kaggle data using kaggle API:
kaggle competitions download riiid-test-answer-prediction --path ./data
This file contains the datatset class and feature engineering functions
'''


DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
DATA_TABLE = False
FILLNA_VAL = 100
LAST_N = 100
TQDM_INT = 8
PAD = 0

class Riiid(Dataset):
    def __init__(self, d):
        self.d = d
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        # you can return a dict of these as well etc etc...
        # remember the order
        return idx, self.d[idx]["content_id"], self.d[idx]["task_container_id"], \
    self.d[idx]["part_id"], self.d[idx]["prior_question_elapsed_time"], self.d[idx]["padded"], \
    self.d[idx]["answered_correctly"]


class RiiidTest(Dataset):
    
    def __init__(self, d):
        self.d = d
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        # you can return a dict of these as well etc etc...
        # remember the order
        return idx, self.d[idx]["content_id"], self.d[idx]["task_container_id"], \
    self.d[idx]["part_id"], self.d[idx]["prior_question_elapsed_time"], self.d[idx]["padded"]

def collate_fn(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded, labels = zip(*batch)
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    labels = torch.Tensor(labels)
    # remember the order
    return content_id, task_id, part_id, prior_question_elapsed_time, padded, labels

def collate_fn_test(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded = zip(*batch)
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    # remember the order
    return content_id, task_id, part_id, prior_question_elapsed_time, padded

def get_valid(train_df, n_tail=50):

    valid_df = train_df.groupby(['user_id']).tail(n_tail)
    print("valid:", valid_df.shape, "users:", valid_df['user_id'].nunique())
    # Train
    train_df.drop(valid_df.index, inplace = True)
    print("train:", train_df.shape, "users:", train_df['user_id'].nunique())
    return train_df, valid_df


def preprocess(data_df, question_df, train=True):
    if train:
        data_df['prior_question_had_explanation'] = \
            data_df['prior_question_had_explanation'].astype(np.float16).fillna(-1).astype(np.int8)
    data_df = data_df[data_df.content_type_id == 0]

    part_ids_map = dict(zip(question_df.question_id, question_df.part))
    data_df['part_id'] = data_df['content_id'].map(part_ids_map)

    data_df["prior_question_elapsed_time"].fillna(FILLNA_VAL, inplace=True) 
    # FILLNA_VAL different than all current values
    data_df["prior_question_elapsed_time"] = data_df["prior_question_elapsed_time"] // 1000
    return data_df


def get_feats(data_df):
    '''
    Using a deque as it automatically limits the max_size as per the Data Strucutre's defination itself
    so we don't need to manage that...
    '''
    df = {}
    user_id_to_idx = {}
    grp = data_df.groupby("user_id").tail(LAST_N) # Select last_n rows of each user.
    grp_user = grp.groupby("user_id")
    num_user_id_grp = len(grp_user)

    with tqdm(total=num_user_id_grp) as pbar:
        for idx, row in grp_user.agg({
            "content_id":list, "answered_correctly":list, "task_container_id":list, 
            "part_id":list, "prior_question_elapsed_time":list
            }).reset_index().iterrows():
            # here we make a split whether a user has more than equal to 100 entries or less than that
            # if it's less than LAST_N, then we need to PAD it using the PAD token defined as 0 by me in this cell block
            # also, padded will be True where we have done padding obviously, rest places it's False.
            if len(row["content_id"]) >= 100:
                df[idx] = {
                    "user_id": row["user_id"],
                    "content_id" : deque(row["content_id"], maxlen=LAST_N),
                    "answered_correctly" : deque(row["answered_correctly"], maxlen=LAST_N),
                    "task_container_id" : deque(row["task_container_id"], maxlen=LAST_N),
                    "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"], maxlen=LAST_N),
                    "part_id": deque(row["part_id"], maxlen=LAST_N),
                    "padded" : deque([False]*100, maxlen=LAST_N)
                }
            else:
                # we have to pad...
                # (max_batch_len - len(seq))
                df[idx] = {
                    "user_id": row["user_id"],
                    "content_id" : deque(row["content_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "answered_correctly" : deque(row["answered_correctly"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "task_container_id" : deque(row["task_container_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "part_id": deque(row["part_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "padded" : deque([False]*len(row["content_id"]) + [True]*(100-len(row["content_id"])), maxlen=LAST_N)
                }
            user_id_to_idx[row["user_id"]] = idx
            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)
        # if in future a new user comes, we will just increase the counts as of now... <WIP>
    return df, user_id_to_idx

def get_feats_test(data_df):
    '''
    Using a deque as it automatically limits the max_size as per the Data Strucutre's defination itself
    so we don't need to manage that...
    '''
    df = {}
    user_id_to_idx = {}
    grp = data_df.groupby("user_id").tail(LAST_N) # Select last_n rows of each user.
    grp_user = grp.groupby("user_id")
    num_user_id_grp = len(grp_user)

    with tqdm(total=num_user_id_grp) as pbar:
        for idx, row in grp_user.agg({
            "content_id":list, "task_container_id":list, 
            "part_id":list, "prior_question_elapsed_time":list
            }).reset_index().iterrows():
            # here we make a split whether a user has more than equal to 100 entries or less than that
            # if it's less than LAST_N, then we need to PAD it using the PAD token defined as 0 by me in this cell block
            # also, padded will be True where we have done padding obviously, rest places it's False.
            if len(row["content_id"]) >= 100:
                df[idx] = {
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
                df[idx] = {
                    "user_id": row["user_id"],
                    "content_id" : deque(row["content_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "task_container_id" : deque(row["task_container_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "part_id": deque(row["part_id"] + [PAD]*(100-len(row["content_id"])), maxlen=LAST_N),
                    "padded" : deque([False]*len(row["content_id"]) + [True]*(100-len(row["content_id"])), maxlen=LAST_N)
                }
            user_id_to_idx[row["user_id"]] = idx
            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)
        # if in future a new user comes, we will just increase the counts as of now... <WIP>
    return df, user_id_to_idx

if __name__ == "__main__":
    if DATA_TABLE:
        start = time()
        data = dt.fread(DATA_DIR+"train.csv")
        print(f"Readding train.csv in {time()-start} seconds")