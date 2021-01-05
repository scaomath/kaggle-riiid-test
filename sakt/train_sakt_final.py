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
RECENT_SIZE = 12 # recent data in the training loop
NUM_HEADS = 8
BATCH_SIZE = 128
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
#%%
class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):
        super(SAKTDataset, self).__init__()
        self.samples, self.n_skill, self.max_seq = {}, n_skill, max_seq
        
        self.user_ids = []
        for i, user_id in enumerate(group.index):
            content_id, answered_correctly = group[user_id]
            if len(content_id) >= ACCEPTED_USER_CONTENT_SIZE:
                if len(content_id) > self.max_seq:
                    total_questions = len(content_id)
                    last_pos = total_questions // self.max_seq
                    for seq in range(last_pos):
                        index = f"{user_id}_{seq}"
                        self.user_ids.append(index)
                        start = seq * self.max_seq
                        end = (seq + 1) * self.max_seq
                        self.samples[index] = (content_id[start:end], 
                                               answered_correctly[start:end])
                    if len(content_id[end:]) >= ACCEPTED_USER_CONTENT_SIZE:
                        index = f"{user_id}_{last_pos + 1}"
                        self.user_ids.append(index)
                        self.samples[index] = (content_id[end:], 
                                               answered_correctly[end:])
                else:
                    index = f'{user_id}'
                    self.user_ids.append(index)
                    self.samples[index] = (content_id, answered_correctly)
            '''
            New: adding a shifted sequence
            '''
            if len(content_id) >= RECENT_SIZE: #
                for i in range(RECENT_SIZE//2):
                    '''
                    generating much much more sequences by truncating
                    '''
                    content_id_truncated_end = content_id[:-i]
                    answered_correctly_truncated_end = answered_correctly[:-i]
                    if len(content_id_truncated_end) >= ACCEPTED_USER_CONTENT_SIZE:
                        if len(content_id_truncated_end) > self.max_seq:
                            total_questions_2 = len(content_id_truncated_end)
                            last_pos = total_questions_2 // self.max_seq
                            for seq in range(last_pos):
                                index = f"{user_id}_{seq}_{i}_2"
                                self.user_ids.append(index)
                                start = seq * self.max_seq
                                end = (seq + 1) * self.max_seq
                                self.samples[index] = (content_id_truncated_end[start:end], 
                                                    answered_correctly_truncated_end[start:end])
                            if len(content_id_truncated_end[end:]) >= ACCEPTED_USER_CONTENT_SIZE:
                                index = f"{user_id}_{last_pos + 1}_{i}_2"
                                self.user_ids.append(index)
                                self.samples[index] = (content_id_truncated_end[end:], 
                                                    answered_correctly_truncated_end[end:])
                        else:
                            index = f'{user_id}_{i}_2'
                            self.user_ids.append(index)
                            self.samples[index] = (content_id_truncated_end, 
                                                   answered_correctly_truncated_end)


                    '''
                    Truncating the started
                    '''
                    content_id_truncated_start = content_id[i:]
                    answered_correctly_truncated_start = answered_correctly[i:]

                    if len(content_id_truncated_start) >= ACCEPTED_USER_CONTENT_SIZE:
                        if len(content_id_truncated_start) > self.max_seq:
                            total_questions_1 = len(content_id_truncated_start)
                            last_pos = total_questions_1 // self.max_seq
                            for seq in range(last_pos):
                                index = f"{user_id}_{seq}_{i}_1"
                                self.user_ids.append(index)
                                start = seq * self.max_seq
                                end = (seq + 1) * self.max_seq
                                self.samples[index] = (content_id_truncated_start[start:end], 
                                                    answered_correctly_truncated_start[start:end])
                            if len(content_id_truncated_start[end:]) >= ACCEPTED_USER_CONTENT_SIZE:
                                index = f"{user_id}_{last_pos + 1}_{i}_1"
                                self.user_ids.append(index)
                                self.samples[index] = (content_id_truncated_start[end:], 
                                                    answered_correctly_truncated_start[end:])
                        else:
                            index = f'{user_id}_{i}_1'
                            self.user_ids.append(index)
                            self.samples[index] = (content_id_truncated_start, 
                                                   answered_correctly_truncated_start)

                
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


class TestDataset(Dataset):
    def __init__(self, samples, test_df, n_skill, max_seq=100):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.n_skill, self.max_seq = n_skill, max_seq

    def __len__(self):
        return self.test_df.shape[0]
    
    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]
        
        user_id = test_info['user_id']
        target_id = test_info['content_id']
        
        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        
        if user_id in self.samples.index:
            content_id, answered_correctly = self.samples[user_id]
            
            seq_len = len(content_id)
            
            if seq_len >= self.max_seq:
                content_id_seq = content_id[-self.max_seq:]
                answered_correctly_seq = answered_correctly[-self.max_seq:]
            else:
                content_id_seq[-seq_len:] = content_id
                answered_correctly_seq[-seq_len:] = answered_correctly
                
        x = content_id_seq[1:].copy()
        x += (answered_correctly_seq[1:] == 1) * self.n_skill
        
        questions = np.append(content_id_seq[2:], [target_id])
        
        return x, questions
# %%
train_dataset = SAKTDataset(train_group, n_skill=NUM_SKILLS, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        drop_last=True)


val_dataset = SAKTDataset(valid_group, n_skill=NUM_SKILLS, max_seq=MAX_SEQ)
val_dataloader = DataLoader(val_dataset, 
                        batch_size=VAL_BATCH_SIZE, 
                        shuffle=False)
# %%
sample_batch = next(iter(train_dataloader))
print(sample_batch[0].shape, sample_batch[1].shape, sample_batch[2].shape)
# %%
class FFN(nn.Module):
    def __init__(self, state_size = MAX_SEQ, 
                    forward_expansion = 1, 
                    bn_size=MAX_SEQ - 1, 
                    dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size
        
        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, 
                    heads = 8, 
                    dropout = DROPOUT, 
                    forward_expansion = 1):
        super(TransformerBlock, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, 
                        num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, 
                    forward_expansion = forward_expansion, 
                    dropout=dropout)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)
        

    def forward(self, value, key, query, att_mask):
        att_output, att_weight = self.multi_att(value, key, query, attn_mask=att_mask)
        att_output = self.dropout(self.layer_normal(att_output + value))
        att_output = att_output.permute(1, 0, 2) 
        # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        x = self.ffn(att_output)
        x = self.dropout(self.layer_normal_2(x + att_output))
        return x.squeeze(-1), att_weight
    
class Encoder(nn.Module):
    def __init__(self, n_skill, max_seq=100, 
                 embed_dim=128, 
                 dropout = DROPOUT, 
                 forward_expansion = 1, 
                 num_layers=1, 
                 heads = 8):
        super(Encoder, self).__init__()
        self.n_skill, self.embed_dim = n_skill, embed_dim
        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, heads=heads,
                forward_expansion = forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, question_ids):
        device = x.device
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(x + pos_x)
        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = self.e_embedding(question_ids)
        e = e.permute(1, 0, 2)
        for layer in self.layers:
            att_mask = future_mask(e.size(0)).to(device)
            x, att_weight = layer(e, x, x, att_mask=att_mask)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x, att_weight

class SAKTModel(nn.Module):
    def __init__(self, 
                n_skill, 
                max_seq=MAX_SEQ, 
                embed_dim=EMBED_SIZE, 
                dropout = DROPOUT, 
                forward_expansion = 1, 
                enc_layers=1, 
                heads = NUM_HEADS):
        super(SAKTModel, self).__init__()
        self.encoder = Encoder(n_skill, 
                               max_seq, 
                               embed_dim, 
                               dropout, 
                               forward_expansion, 
                               num_layers=enc_layers,
                               heads=heads)
        self.pred = nn.Linear(embed_dim, 1)
        
    def forward(self, x, question_ids):
        x, att_weight = self.encoder(x, question_ids)
        x = self.pred(x)
        return x.squeeze(-1), att_weight

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
print(f"Current model has {n_params} parameters.")

# %%
def load_from_item(item):
    x = item[0].to(device).long()
    target_id = item[1].to(device).long()
    label = item[2].to(device).float()
    target_mask = (target_id != 0)
    return x, target_id, label, target_mask

def update_stats(tbar, train_loss, loss, output, label, num_corrects, num_total, labels, outs):
    train_loss.append(loss.item())
    pred = (torch.sigmoid(output) >= 0.5).long()
    num_corrects += (pred == label).sum().item()
    num_total += len(label)
    labels.extend(label.view(-1).data.cpu().numpy())
    outs.extend(output.view(-1).data.cpu().numpy())
    tbar.set_description('loss - {:.4f}'.format(loss))
    return num_corrects, num_total

def train_epoch(model, dataloader, optim, criterion, scheduler, device="cpu"):
    model.train()
    
    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []
    
    tbar = tqdm(dataloader)
    for item in tbar:
        x, target_id, label, target_mask = load_from_item(item)
        
        optim.zero_grad()
        output, _ = model(x, target_id)
        
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        scheduler.step()
        
        tbar.set_description('loss - {:.4f}'.format(loss))

def val_epoch(model, val_iterator, criterion, device="cpu"):
    model.eval()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(val_iterator)
    for item in tbar:
        x, target_id, label, target_mask = load_from_item(item)

        with torch.no_grad():
            output, atten_weight = model(x, target_id)
        
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)
        
        num_corrects, num_total = update_stats(tbar, 
                                               train_loss, 
                                               loss, 
                                               output, 
                                               label, 
                                               num_corrects, 
                                               num_total, 
                                               labels, 
                                               outs)

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.average(train_loss)

    return loss, acc, auc

def do_train(lr=1e-3, epochs=10):
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
        val_loss, val_acc, val_auc = val_epoch(model, val_dataloader, criterion, device)
        print(f"\nepoch - {epoch + 1} val_loss - {val_loss:.3f} acc - {val_acc:.3f} auc - {val_auc:.3f}")
        if best_auc < val_auc:
            print(f'epoch - {epoch + 1} best model with val auc: {val_auc}')
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'sakt_seq_{MAX_SEQ}_auc_{val_auc:.4f}.pt'))
# %%
LR = 1e-3
EPOCHS = 10
do_train(lr=LR, epochs=EPOCHS)
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
