#%% 
import gc, sys, os
from tqdm.notebook import tqdm

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

from utils import *

print('Python     : ' + sys.version.split('\n')[0])
print('Numpy      : ' + np.__version__)
print('Pandas     : ' + pd.__version__)
print('PyTorch    : ' + torch.__version__)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # if IS_TPU == False else xm.xla_device()
print('Device     : {}'.format(DEVICE))
# %%
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
HOME =  "/home/scao/Documents/kaggle-riiid-test"
DATA_HOME = HOME+'/data/'
MODEL_NAME = "SAKT-v1"
MODEL_PATH = HOME + MODEL_NAME
STAGE = "stage1"
MODEL_BEST = 'model_best.pt'
FOLD = 1

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    
CONTENT_TYPE_ID = "content_type_id"
CONTENT_ID = "content_id"
TARGET = "answered_correctly"
USER_ID = "user_id"
TASK_CONTAINER_ID = "task_container_id"
TIMESTAMP = "timestamp" 
# %%
dtype = {TIMESTAMP: 'int64', 
         USER_ID: 'int32', 
         CONTENT_ID: 'int16',
         CONTENT_TYPE_ID: 'bool',
         TARGET:'int8'}
train_df = pd.read_csv(DATA_HOME + 'train.csv', usecols=[1, 2, 3, 4, 7], dtype=dtype)
train_df = train_df[train_df[CONTENT_TYPE_ID] == False].reset_index(drop = True)
print(len(train_df))
# %%
# Valid with last 100 interactions (must be improved to balance new users and be around 2.5M rows only)
# However, valid_df includes all users' history needed for testing.
valid_df = train_df.groupby([USER_ID]).tail(100)
print("valid:", valid_df.shape, "users:", valid_df[USER_ID].nunique())
# Train
train_df.drop(valid_df.index, inplace = True)
print("train:", train_df.shape, "users:", train_df[USER_ID].nunique())
# %%
print(train_df[train_df[USER_ID] == 2147482216].head(10))
print(valid_df[valid_df[USER_ID] == 115].head(10))
# %% preprocess
skills = train_df[CONTENT_ID].unique()
n_skill = 13523 # len(skills)
print("Number of skills", n_skill)

# Index by user_id
valid_df = valid_df.reset_index(drop = True)
valid_group = valid_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID).apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

# Index by user_id
train_df = train_df.reset_index(drop = True)
train_group = train_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID).apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))
# %%
class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, subset="train", max_seq=100):
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill # 13523
        self.samples = group
        self.subset = subset
        
        # self.user_ids = [x for x in group.index]
        self.user_ids = []
        for user_id in group.index:
            q, qa = group[user_id]
            if len(q) < 10: # 10 interactions minimum
                continue
            self.user_ids.append(user_id) # user_ids indexes

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index] # Pick a user
        q_, qa_ = self.samples[user_id] # Pick full sequence for user
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            if self.subset == "train":
                if seq_len > self.max_seq:
                    random_start_index = np.random.randint(seq_len - self.max_seq)
                    q[:] = q_[random_start_index:random_start_index + self.max_seq] # Pick 100 questions from a random index
                    qa[:] = qa_[random_start_index:random_start_index + self.max_seq] # Pick 100 answers from a random index
                else:
                    q[:] = q_[-self.max_seq:]
                    qa[:] = qa_[-self.max_seq:]
            else:
                q[:] = q_[-self.max_seq:] # Pick last 100 questions
                qa[:] = qa_[-self.max_seq:] # Pick last 100 answers
        else:
            q[-seq_len:] = q_ # Pick last N question with zero padding
            qa[-seq_len:] = qa_ # Pick last N answers with zero padding        
                
        target_id = q[1:] # Ignore first item 1 to 99
        label = qa[1:] # Ignore first item 1 to 99

        # x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy() # 0 to 98
        x += (qa[:-1] == 1) * self.n_skill # y = et + rt x E

        return x, target_id, label

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128):
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
def train_epoch(model, train_iterator, optim, criterion, device="cuda"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    # tbar = tqdm(train_iterator)
    for item in train_iterator: #tbar:
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
        #outs.extend(output.view(-1).data.cpu().numpy())
        outs.extend(torch.sigmoid(output).view(-1).data.cpu().numpy())

        # tbar.set_description('loss - {:.4f}'.format(loss))
    
    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc

def valid_epoch(model, valid_iterator, criterion, device="cuda"):
    model.eval()

    valid_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    #tbar = tqdm(valid_iterator)
    for item in valid_iterator: # tbar:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()

        with torch.no_grad():
            output, atten_weight = model(x, target_id)
        loss = criterion(output, label)
        valid_loss.append(loss.item())

        output = output[:, -1] # (BS, 1)
        label = label[:, -1] 
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        #outs.extend(output.view(-1).data.cpu().numpy())
        outs.extend(torch.sigmoid(output).view(-1).data.cpu().numpy())

        #tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(valid_loss)

    return loss, acc, auc

class conf:
    METRIC_ = "max"
    WORKERS = 10 # 0
    BATCH_SIZE = 2048
    lr = 1e-3
    D_MODEL = 128

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
# %%
train_dataset = SAKTDataset(train_group, n_skill, subset="train")
train_dataloader = DataLoader(train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=conf.WORKERS)

valid_dataset = SAKTDataset(valid_group, n_skill, subset="valid")
valid_dataloader = DataLoader(valid_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=conf.WORKERS)

item = train_dataset.__getitem__(5)

print("x", len(item[0]), item[0])
print("target_id", len(item[1]), item[1])
print("label", len(item[2]), item[2])
# %%
device = DEVICE

model = SAKTModel(n_skill, embed_dim=conf.D_MODEL)
optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)
# %%
epochs = 48
auc_max = -np.inf
history = []

snapshot_path = "%s/fold%d/%s/snapshots" % (MODEL_PATH, FOLD, STAGE)
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

print("Stage:", STAGE, "fold:", FOLD, "on:", DEVICE, "workers:", conf.WORKERS, "batch size:", conf.BATCH_SIZE, "metric_:", conf.METRIC_, 
      "train dataset:", len(train_dataset), "valid dataset:", len(valid_dataset))

for epoch in range(1, epochs+1):
    train_loss, train_acc, train_auc = train_epoch(model, train_dataloader, optimizer, criterion, device)
    print("\nEpoch#{}, train_loss - {:.2f} acc - {:.4f} auc - {:.4f}".format(epoch, train_loss, train_acc, train_auc))
    valid_loss, valid_acc, valid_auc = valid_epoch(model, valid_dataloader, criterion, device)
    print("Epoch#{}, valid_loss - {:.2f} acc - {:.4f} auc - {:.4f}".format(epoch, valid_loss, valid_acc, valid_auc))
    lr = optimizer.param_groups[0]['lr']
    history.append({"epoch":epoch, "lr": lr, **{"train_auc": train_auc, "train_acc": train_acc}, **{"valid_auc": valid_auc, "valid_acc": valid_acc}})
    if valid_auc > auc_max:
        print("Epoch#%s, valid loss %.4f, Metric loss improved from %.4f to %.4f, saving model ..." % (epoch, valid_loss, auc_max, valid_auc))
        auc_max = valid_auc
        torch.save(model.state_dict(), os.path.join(snapshot_path, MODEL_BEST))

if history:
    metric = "auc"
    # Plot training history
    history_pd = pd.DataFrame(history[1:]).set_index("epoch")
    train_history_pd = history_pd[[c for c in history_pd.columns if "train_" in c]]
    valid_history_pd = history_pd[[c for c in history_pd.columns if "valid_" in c]]
    lr_history_pd = history_pd[[c for c in history_pd.columns if "lr" in c]]
    fig, ax = plt.subplots(1,2, figsize=(DEFAULT_FIG_WIDTH, 6))
    t_epoch = train_history_pd["train_%s" % metric].argmin() if conf.METRIC_ == "min" else train_history_pd["train_%s" % metric].argmax()
    v_epoch = valid_history_pd["valid_%s" % metric].argmin() if conf.METRIC_ == "min" else valid_history_pd["valid_%s" % metric].argmax()
    d = train_history_pd.plot(kind="line", ax=ax[0], title="Epoch: %d, Train: %.3f" % (t_epoch, train_history_pd.iloc[t_epoch,:]["train_%s" % metric]))
    d = lr_history_pd.plot(kind="line", ax=ax[0], secondary_y=True)
    d = valid_history_pd.plot(kind="line", ax=ax[1], title="Epoch: %d, Valid: %.3f" % (v_epoch, valid_history_pd.iloc[v_epoch,:]["valid_%s" % metric]))
    d = lr_history_pd.plot(kind="line", ax=ax[1], secondary_y=True)
    plt.savefig("%s/train.png" % snapshot_path, bbox_inches='tight')
    plt.show()
# %% Test

class TestDataset(Dataset):
    def __init__(self, samples, test_df, skills, max_seq=100):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.skills = skills
        self.n_skill = len(skills)
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_ = self.samples[user_id]
            
            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_          
        
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill
        
        questions = np.append(q[2:], [target_id])
        
        return x, questions
#%% Reload model with best weights
model = SAKTModel(n_skill, embed_dim=conf.D_MODEL)
resume_path = os.path.join(snapshot_path, MODEL_BEST)
if os.path.exists(resume_path):
    model.load_state_dict(torch.load(resume_path, map_location=conf.map_location))
    print("Resuming, model weights loaded: %s" % resume_path)
model.to(device)
_ = model.eval()