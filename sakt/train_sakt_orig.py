#%% 
import gc, os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from utils import *
#%%
MAX_SEQ = 100
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
dtype = {'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16','content_type_id': 'int8','answered_correctly':'int8'}

train_df = pd.read_csv(DATA_DIR+'train.csv', usecols=[1, 2, 3,4,7], dtype=dtype)
train_df.head()
# %%
train_df = train_df[train_df.content_type_id == False]

#arrange by timestamp
train_df = train_df.sort_values(['timestamp'], ascending=True).reset_index(drop = True)

skills = train_df["content_id"].unique()
n_skill = len(skills)
print("number skills", len(skills))
# %%
group = train_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id')\
        .apply(lambda r: (r['content_id'].values,
                          r['answered_correctly'].values))

# %%
class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ): #HDKIM 100
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group
        
#         self.user_ids = [x for x in group.index]
        self.user_ids = []
        for user_id in group.index:
            q, qa = group[user_id]
            if len(q) < 5: #HDKIM 10
                continue
            self.user_ids.append(user_id)
            
            #HDKIM Memory reduction
            if len(q)>self.max_seq:
                group[user_id] = (q[-self.max_seq:],qa[-self.max_seq:])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
        
        target_id = q[1:]
        label = qa[1:]

        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x, target_id, label
# %%
dataset = SAKTDataset(group, n_skill)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=8)

item = dataset.__getitem__(5)
# print(item[0])
# print(item[1])
# print(item[2])
# %%
class FFN(nn.Module):
    def __init__(self, state_size=256):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=MAX_SEQ, embed_dim=128): #HDKIM 100
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, x, question_ids):
        device = x.device        
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAKTModel(n_skill, embed_dim=128)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)
# %%
def train_epoch(model, train_iterator, optim, criterion, device="cuda"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()

        optim.zero_grad()
        output, atten_weight = model(x, target_id)
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        output = output[:, -1]
        label = label[:, -1] 
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc
# %%
epochs = 20
for epoch in range(epochs):
    loss, acc, auc = train_epoch(model, dataloader, optimizer, criterion, device)
    print("epoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, loss, acc, auc))
