#%% 
import gc, sys, os
from tqdm import tqdm

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

CONTENT_TYPE_ID = "content_type_id"
CONTENT_ID = "content_id"
TARGET = "answered_correctly"
USER_ID = "user_id"
TASK_CONTAINER_ID = "task_container_id"
TIMESTAMP = "timestamp" 

TRAIN_DTYPES = {TIMESTAMP: 'int64', 
         USER_ID: 'int32', 
         CONTENT_ID: 'int16',
         CONTENT_TYPE_ID: 'bool',
         TARGET:'int8'}

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

TQDM_INT = 8 
HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = HOME+'/data/'
MODEL_PATH = HOME + 'model'
STAGE = "stage1"
FOLD = 1
NUM_HEADS = 10
NUM_EMBED = 128
NUM_SKILLS = 13523
MAX_SEQ = 100
DEBUG = True
NROWS_TRAIN = 5_000_000
N_TAIL = 100
# if not os.path.exists(MODEL_PATH):
    # os.makedirs(MODEL_PATH)
    
# %%

# train_df = pd.read_pickle(DATA_DIR+'cv2_train.pickle')
# valid_df = pd.read_pickle(DATA_DIR+'cv2_valid.pickle')
# train_df = train_df[TRAIN_DTYPES.keys()]
# valid_df = valid_df[TRAIN_DTYPES.keys()]
if DEBUG:
    train_df = pd.read_csv(DATA_DIR + 'train.csv', 
                        nrows=NROWS_TRAIN, 
                        usecols=[1, 2, 3, 4, 7], 
                        dtype=TRAIN_DTYPES)
else:
    train_df = pd.read_csv(DATA_DIR + 'train.csv', 
                        usecols=[1, 2, 3, 4, 7], 
                        dtype=TRAIN_DTYPES)
train_df = train_df[train_df[CONTENT_TYPE_ID] == False]
#arrange by timestamp
train_df = train_df.sort_values(['timestamp'], ascending=True).reset_index(drop = True)
print(len(train_df))

valid_df = train_df.groupby([USER_ID]).tail(N_TAIL)
train_df.drop(valid_df.index, inplace = True)

print("valid:", valid_df.shape, "users:", valid_df[USER_ID].nunique())
print("train:", train_df.shape, "users:", train_df[USER_ID].nunique())
# print(train_df[train_df[USER_ID] == 2147482216].head(10))
# print(valid_df[valid_df[USER_ID] == 115].head(10))

skills = train_df[CONTENT_ID].unique()
n_skill = NUM_SKILLS #len(skills) # len(skills) might not have all
print("Number of skills", n_skill)

# Index by user_id
valid_df = valid_df.reset_index(drop = True)
valid_group = valid_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

# Index by user_id
train_df = train_df.reset_index(drop = True)
train_group = train_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

# %%
class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, subset="train", max_seq = MAX_SEQ):
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


# class SAKTDataset(Dataset):
#     def __init__(self, group, n_skill, max_seq=MAX_SEQ): #HDKIM 100
#         super(SAKTDataset, self).__init__()
#         self.max_seq = max_seq
#         self.n_skill = n_skill
#         self.samples = group
        
# #         self.user_ids = [x for x in group.index]
#         self.user_ids = []
#         for user_id in group.index:
#             q, qa = group[user_id]
#             if len(q) < 5: #HDKIM 10
#                 continue
#             self.user_ids.append(user_id)
            
#             #HDKIM Memory reduction
#             if len(q)>self.max_seq:
#                 group[user_id] = (q[-self.max_seq:],qa[-self.max_seq:])

#     def __len__(self):
#         return len(self.user_ids)

#     def __getitem__(self, index):
#         user_id = self.user_ids[index]
#         q_, qa_ = self.samples[user_id]
#         seq_len = len(q_)

#         q = np.zeros(self.max_seq, dtype=int)
#         qa = np.zeros(self.max_seq, dtype=int)
#         if seq_len >= self.max_seq:
#             q[:] = q_[-self.max_seq:]
#             qa[:] = qa_[-self.max_seq:]
#         else:
#             q[-seq_len:] = q_
#             qa[-seq_len:] = qa_
        
#         target_id = q[1:]
#         label = qa[1:]

#         x = np.zeros(self.max_seq-1, dtype=int)
#         x = q[:-1].copy()
#         x += (qa[:-1] == 1) * self.n_skill

#         return x, target_id, label

class FFN(nn.Module):
    def __init__(self, state_size=256):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.3)
    
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
    def __init__(self, n_skill, 
                       max_seq=MAX_SEQ, 
                       embed_dim=NUM_EMBED, 
                       num_heads=NUM_HEADS): #HDKIM 100
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, 
                                               num_heads=num_heads, 
                                               dropout=0.2)

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
    
    len_dataset = len(train_iterator)

    with tqdm(total=len_dataset) as pbar:
        for idx, item in enumerate(train_iterator): 
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

            if idx % TQDM_INT == 0:
                pbar.set_description(f'loss - {train_loss[-1]:.4f}')
                pbar.update(TQDM_INT)
    
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

    for item in valid_iterator: 
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


    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(valid_loss)

    return loss, acc, auc

class conf:
    METRIC_ = "max"
    WORKERS = 8 # 0
    BATCH_SIZE = 512
    VAL_BATCH_SIZE = 2048
    lr = 1e-3
    D_MODEL = 128
    NUM_HEADS = 8

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
# %%
train_dataset = SAKTDataset(train_group, n_skill, subset="train")
# train_dataset = SAKTDataset(train_group, n_skill)
train_dataloader = DataLoader(train_dataset, 
                              batch_size=conf.BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=conf.WORKERS)

valid_dataset = SAKTDataset(valid_group, n_skill, subset="valid")
# valid_dataset = SAKTDataset(valid_group, n_skill)
valid_dataloader = DataLoader(valid_dataset, 
                              batch_size=conf.VAL_BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=conf.WORKERS)

item = train_dataset.__getitem__(5)

print("x", len(item[0]), item[0], '\n\n')
print("target_id", len(item[1]), item[1] , '\n\n')
print("label", len(item[2]), item[2])
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAKTModel(n_skill, embed_dim=conf.D_MODEL, num_heads=conf.NUM_HEADS)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)
num_params = get_num_params(model)
print("\nFold:   ", FOLD, 
      "\non:     ", DEVICE, 
      "\nworkers:", conf.WORKERS, 
      "\ntraining batch size:", conf.BATCH_SIZE, 
      "\nmetric_:", conf.METRIC_) 
print(f"# of params in model: {num_params}")
print( "train dataset:", len(train_dataset), "valid dataset:", len(valid_dataset))


#%%
model.train()

train_loss = []
num_corrects = 0
num_total = 0
labels = []
outs = []

len_dataset = len(train_dataloader)


for idx, item in enumerate(train_dataloader): 
    print(f'\n\nBatch {idx+1}')
    x = item[0].to(device).long()
    target_id = item[1].to(device).long()
    label = item[2].to(device).float()
    
    optimizer.zero_grad()
    output, atten_weight = model(x, target_id)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())

    output = output[:, -1]
    label = label[:, -1] 
    pred = (torch.sigmoid(output) >= 0.5).long()
    
    num_corrects += (pred == label).sum().item()
    num_total += len(label)

    labels.extend(label.view(-1).data.cpu().numpy())
    #outs.extend(output.view(-1).data.cpu().numpy())
    outs.extend(torch.sigmoid(output).view(-1).data.cpu().numpy())
        

# acc = num_corrects / num_total
# auc = roc_auc_score(labels, outs)
# loss = np.mean(train_loss)
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
model_files = find_files('sakt', MODEL_DIR)
print(f"Loading {model_files[-1]}")
model.load_state_dict(torch.load(model_files[-1], map_location=conf.map_location))
model.to(device)
_ = model.eval()

#%% 
test_df = pd.read_csv(DATA_DIR+'example_test.csv', 
                        dtype=TEST_DTYPES, 
                        usecols=TEST_DTYPES.keys())
test_df = test_df[test_df.content_type_id == False]
    
test_dataset = TestDataset(valid_group, test_df, skills)
test_dataloader = DataLoader(test_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, drop_last=False)

outs = []

for item in test_dataloader:
    x = item[0].to(device).long()
    target_id = item[1].to(device).long()

    with torch.no_grad():
        output, _ = model(x, target_id)
            
    output = torch.sigmoid(output)
    output = output[:, -1]

    outs.extend(output.view(-1).data.cpu().numpy())
    
test_df['answered_correctly'] = outs

test_df['answered_correctly'].hist()