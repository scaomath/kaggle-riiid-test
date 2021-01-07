#%% saint
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
# %%

PREPROCESS = False

MAX_SEQ = 100
NUM_EMBED = 256
NUM_LAYERS = 4
NUM_HEADS = 8
BATCH_SIZE = 512
NUM_SKILLS = 13523
VAL_BATCH_SIZE = 512
DROPOUT = 0.1
NUM_PARTS = 7
ACCEPTED_USER_CONTENT_SIZE = 4

CONTENT_TYPE_ID = "content_type_id"
CONTENT_ID = "content_id"
TARGET = "answered_correctly"
USER_ID = "user_id"
PRIOR_QUESTION_TIME = 'prior_question_elapsed_time'
PRIOR_QUESTION_EXPLAIN = 'prior_question_had_explanation'
TASK_CONTAINER_ID = "task_container_id"
TIMESTAMP = "timestamp" 
ROW_ID = 'row_id'
PART = 'part'
TRAIN_DTYPES = {
    TIMESTAMP: 'int64',
    USER_ID: 'int32', 
    CONTENT_ID: 'int16', 
    CONTENT_TYPE_ID:'int8', 
    TASK_CONTAINER_ID: 'int16',
    TARGET: 'int8', 
    PRIOR_QUESTION_TIME: 'float32', 
    PRIOR_QUESTION_EXPLAIN: 'bool'
}
# %%
class FFN(nn.Module):
    def __init__(self, state_size = MAX_SEQ, 
                    bn_size=MAX_SEQ - 1, 
                    dropout=DROPOUT):
        super(FFN, self).__init__()
        self.state_size = state_size
        
        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)

class SAINTModel(nn.Module):
    def __init__(self, n_skill, n_part, 
                        max_seq=MAX_SEQ, 
                        embed_dim=NUM_EMBED,
                        nhead=NUM_HEADS,
                        nlayer=NUM_LAYERS):
        super(SAINTModel, self).__init__()

        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.n_cat = n_part

        self.e_embedding = nn.Embedding(self.n_skill+1, embed_dim) ##exercise
        self.c_embedding = nn.Embedding(self.n_cat+1, embed_dim) ##category
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim) ## position
        self.res_embedding = nn.Embedding(2, embed_dim) ## response

        self.transformer = nn.Transformer(nhead=nhead, 
                                          d_model = embed_dim, 
                                          num_encoder_layers= nlayer, 
                                          num_decoder_layers= nlayer, 
                                          dropout = 0.1)

        self.dropout = nn.Dropout(0.1)
        self.layer_normal = nn.LayerNorm(embed_dim) 
        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, question, part, response):

        device = question.device  

        question = self.e_embedding(question)
        part = self.c_embedding(part)
        pos_id = torch.arange(question.size(1)).unsqueeze(0).to(device)
        pos_id = self.pos_embedding(pos_id)
        response = self.res_embedding(response)

        enc = question + part + pos_id
        dec = pos_id + response

        enc = enc.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        dec = dec.permute(1, 0, 2)
        mask = future_mask(enc.size(0)).to(device)

        att_output = self.transformer(enc, dec, src_mask=mask, tgt_mask=mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)
# %%

def preprocess_saint(df):
    df = df[df[CONTENT_TYPE_ID] == False].reset_index(drop = True)    

    df = df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)

    df = df[df[CONTENT_TYPE_ID] == False]
    
    group = df[[USER_ID, CONTENT_ID, TARGET, PART]]\
        .groupby(USER_ID)\
        .apply(lambda r: (r[CONTENT_ID].values, 
                            r[TARGET].values,
                            r[PART].values))
    return group

if PREPROCESS:
    train_df = dt.fread(os.path.join(DATA_DIR, 'train.csv'), 
                        columns=set(TRAIN_DTYPES.keys())).to_pandas().astype(TRAIN_DTYPES)
    questions_df = pd.read_csv(os.path.join(DATA_DIR, 'questions.csv'))
    questions_df['part'] = questions_df['part'].astype(np.int8)
    questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)
    train_df = pd.merge(train_df, questions_df[['question_id', 'part']], 
                    left_on = 'content_id',  right_on = 'question_id',  how = 'left')
    group = preprocess_saint(train_df)
    with open(os.path.join(DATA_DIR,'group_saint.pickle'), 'wb') as f:
        pickle.dump(group, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(DATA_DIR+'/group_saint.pickle', 'rb') as f:
        group = pickle.load(f)

train_group, valid_group = train_test_split(group, test_size=0.1)

# %%
class SAINTDataset(Dataset):
    '''
    Only for validation
    '''
    def __init__(self, group, n_skill, 
                        max_seq=MAX_SEQ, 
                        min_seq=ACCEPTED_USER_CONTENT_SIZE,
                        recent_seq=None):
        super(SAINTDataset, self).__init__()
        self.samples = {}
        self.n_skill = n_skill
        self.max_seq = max_seq
        self.min_seq = min_seq
        self.recent_seq = recent_seq
        
        self.user_ids = []
        for i, user_id in enumerate(group.index):
            content_id, answered_correctly, part = group[user_id]
            if len(content_id) >= self.min_seq:
                if len(content_id) > self.max_seq:
                    total_questions = len(content_id)
                    last_pos = total_questions // self.max_seq
                    for seq in range(last_pos):
                        index = f"{user_id}_{seq}"
                        self.user_ids.append(index)
                        start = seq * self.max_seq
                        end = (seq + 1) * self.max_seq
                        self.samples[index] = (content_id[start:end], 
                                               answered_correctly[start:end],
                                               part[start:end])
                    if len(content_id[end:]) >= self.min_seq:
                        index = f"{user_id}_{last_pos + 1}"
                        self.user_ids.append(index)
                        self.samples[index] = (content_id[end:], 
                                               answered_correctly[end:],
                                               part[end:])
                else:
                    index = f'{user_id}'
                    self.user_ids.append(index)
                    self.samples[index] = (content_id, answered_correctly, part)
            '''
            New: adding a shifted sequence
            '''
            if self.recent_seq is None: self.recent_seq = 10**6
            if len(content_id) >= 2*self.recent_seq: #
                for i in range(1, self.recent_seq): # adding a shifted sequence
                    '''
                    generating much much more sequences by truncating
                    '''
                    content_id_shift = content_id[:-i]
                    answered_correctly_shift = answered_correctly[:-i]
                    part_shift = part[:-i]
                    if len(content_id_shift) >= self.min_seq:
                        if len(content_id_shift) > self.max_seq:
                            total_questions_2 = len(content_id_shift)
                            last_pos = total_questions_2 // self.max_seq
                            for seq in range(last_pos):
                                index = f"{user_id}_{seq}_{i}_2"
                                self.user_ids.append(index)
                                start = seq * self.max_seq
                                end = (seq + 1) * self.max_seq
                                self.samples[index] = (content_id_shift[start:end], 
                                                      answered_correctly_shift[start:end],
                                                      part_shift[start:end])
                            if len(content_id_shift[end:]) >= self.min_seq:
                                index = f"{user_id}_{last_pos + 1}_{i}_2"
                                self.user_ids.append(index)
                                self.samples[index] = (content_id_shift[end:], 
                                                     answered_correctly_shift[end:],
                                                     part_shift[end:])
                        else:
                            index = f'{user_id}_{i}_2'
                            self.user_ids.append(index)
                            self.samples[index] = (content_id_shift, 
                                                   answered_correctly_shift,
                                                   part_shift)

                
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        content_id, answered_correctly, part = self.samples[user_id]
        seq_len = len(content_id)
        
        content_id_seq = np.zeros(self.max_seq, dtype=int)
        label_seq = np.zeros(self.max_seq, dtype=int)
        part_seq = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            content_id_seq[:] = content_id[-self.max_seq:]
            label_seq[:] = answered_correctly[-self.max_seq:]
            part_seq[:] = part[-self.max_seq:]
        else:
            content_id_seq[-seq_len:] = content_id
            label_seq[-seq_len:] = answered_correctly
            part_seq[-seq_len:] = part

        question_id = content_id_seq[1:] # question including the current one
        label = label_seq[1:] # answers including the current
        part = part_seq[1:] # timestamp including the current
        response = label_seq[:-1] # previous answers

        return question_id, part, response, label
# %%
train_dataset = SAINTDataset(train_group, n_skill=NUM_SKILLS, recent_seq=None)
train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=8)

valid_dataset = SAINTDataset(valid_group, n_skill=NUM_SKILLS)
valid_loader = DataLoader(valid_dataset, 
                        batch_size=VAL_BATCH_SIZE, 
                        shuffle=False, drop_last=False,
                        num_workers=8)
print(len(train_loader), len(valid_loader))
# %%
sample = next(iter(train_loader))

print("\n\nq size", (sample[0]).size(),'\n', sample[0], '\n\n')
print("\n\npart size", (sample[1]).size(),'\n', sample[1] , '\n\n')
print("\n\nreponse size", (sample[2]).size(), '\n', sample[2], '\n\n')
print("\n\nlabel size", (sample[3]).size(), '\n', sample[3])
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAINTModel(n_skill=NUM_SKILLS, 
                    n_part=NUM_PARTS,
                    max_seq=MAX_SEQ, 
                    embed_dim=NUM_EMBED,
                    nhead=NUM_HEADS,
                    nlayer=NUM_LAYERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_params = get_num_params(model)
print(f"Current model has {n_params} parameters running on {device}.")
# %%

def load_from_item(item, device='cuda'):
    question_id = item[0].to(device).long()
    part = item[1].to(device).long()
    response = item[2].to(device).long()
    label = item[3].to(device).float()
    target_mask = (question_id != 0)
    return question_id, part, response, label, target_mask


def update_stats(train_loss, loss, output, label, num_corrects, num_total, labels, outs):
    train_loss.append(loss.item())
    pred = (torch.sigmoid(output) >= 0.5).long()
    num_corrects += (pred == label).sum().item()
    num_total += len(label)
    labels.extend(label.view(-1).data.cpu().numpy())
    outs.extend(output.view(-1).data.cpu().numpy())
    return num_corrects, num_total

def train_epoch(model, dataloader, optim, criterion, scheduler, device='cuda'):
    model.train()
    train_loss = []

    tbar = tqdm(dataloader)
    for idx, item in enumerate(tbar):
        question_id, part, response, label, target_mask = load_from_item(item, device)
        
        optim.zero_grad()
        output = model(question_id, part, response)
        
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        scheduler.step()
        
        train_loss.append(loss.item())
        lr = optim.param_groups[0]['lr']
        train_loss_mean = np.mean(train_loss)
        tbar.set_description(f'Train loss - {train_loss_mean:.6f} - LR: {lr:.3e}')

def valid_epoch(model, val_iterator, criterion):
    model.eval()

    val_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(val_iterator)
    for item in tbar:
        question_id, part, response, label, target_mask = load_from_item(item, device)

        with torch.no_grad():
            output = model(question_id, part, response)
        
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)
        
        num_corrects, num_total = update_stats(train_loss, loss, 
                                               output, label, 
                                               num_corrects,num_total, 
                                               labels, outs)

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.average(val_loss)

    return loss, acc, auc

def run_train(lr=1e-3, epochs=10, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                                    steps_per_epoch=len(train_loader), 
                                                    epochs=epochs)
    model.to(device)
    criterion.to(device)
    best_auc = 0.0
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        # train_epoch_weighted(model, train_dataloader, optimizer, criterion, scheduler, device)
        val_loss, val_acc, val_auc = valid_epoch(model, valid_loader, criterion, device)
        val_info = f"\nEpoch - {epoch+1}"
        val_info += f" val_loss - {val_loss:.3f} acc - {val_acc:.3f} auc - {val_auc:.3f}"
        tqdm.write(val_info)
        if best_auc < val_auc:
            print(f'Epoch - {epoch+1} best model with val auc: {val_auc}')
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 
            f'saint_seq_{MAX_SEQ}_auc_{val_auc:.4f}.pt'))
# %%

LR = 1e-3
EPOCHS = 2
run_train(lr=LR, epochs=EPOCHS)
# %%
