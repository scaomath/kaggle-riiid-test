import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys

import numpy as np
from random import random, randint
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm

HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'

sys.path.append(HOME)
from utils import *


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

'''
Model and training hyperparams
'''
MAX_SEQ = 150
NUM_EMBED = 128
NUM_HEADS = 8
NUM_SKILLS = 13523 # len(skills)
NUM_HIDDEN = 128
NUM_LAYERS = 2
NUM_TIME = 300 # when scaled by 1000 and round, priori question time's unique values
LEARNING_RATE = 1e-3
PATIENCE = 8 # overfit patience

'''
System setup
'''
TQDM_INT = 8
WORKERS = 8 # 0
METRIC_ = "max"
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location='cpu'


def preprocess_sakt(df,train_flag=1):
    '''
    Train: train_flag=1
    Valid: train_flag=2
    Test: train_flag=0
    '''
    df = df[df[CONTENT_TYPE_ID] == False].reset_index(drop = True)

    if train_flag == 1:
        df = df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)

    df[PRIOR_QUESTION_TIME].fillna(FILLNA_VAL, inplace=True) 
        # FILLNA_VAL different than all current values
    df[PRIOR_QUESTION_TIME] = round(df[PRIOR_QUESTION_TIME] / TIME_SCALING).astype(np.int16)
    df[PRIOR_QUESTION_EXPLAIN] = df[PRIOR_QUESTION_EXPLAIN].astype(np.float16).fillna(0).astype(np.int8)

    df = df[df[CONTENT_TYPE_ID] == False]
    

    if train_flag:
        group = df[[USER_ID, CONTENT_ID, PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN, TARGET]]\
            .groupby(USER_ID)\
            .apply(lambda r: (r[CONTENT_ID].values, 
                            r[PRIOR_QUESTION_TIME].values,
                            r[PRIOR_QUESTION_EXPLAIN].values,
                            r[TARGET].values))
    else:
        group = df[[USER_ID, CONTENT_ID, PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN]]\
            .groupby(USER_ID)\
            .apply(lambda r: (r[CONTENT_ID].values, 
                            r[PRIOR_QUESTION_TIME].values,
                            r[PRIOR_QUESTION_EXPLAIN].values))
    return group

class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill # 13523
        self.samples = group
        
        # self.user_ids = [x for x in group.index]
        self.user_ids = []
        for user_id in group.index:
            q, qa = group[user_id]
            if len(q) < 2: # 5 interactions minimum
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
            if random() > 0.1:
                start_index = randint(0, seq_len - self.max_seq)
                end_index = start_index + self.max_seq
                q[:] = q_[start_index:end_index] # Pick 100 questions from a random index
                qa[:] = qa_[start_index:end_index] # Pick 100 answers from a random index
            else:
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
        else:
            if random() > 0.1:
                seq_len = randint(2,seq_len)
                q[-seq_len:] = q_[:seq_len]
                qa[-seq_len:] = qa_[:seq_len]
            else:
                q[-seq_len:] = q_ # Pick last N question with zero padding
                qa[-seq_len:] = qa_ # Pick last N answers with zero padding        
                
        target_id = q[1:] # Ignore first item 1 to 99
        label = qa[1:] # Ignore first item 1 to 99

        # x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy() # 0 to 98
        x += (qa[:-1] == 1) * self.n_skill # y = et + rt x E

        return x, target_id, label


class SAKTDatasetNew(Dataset):
    def __init__(self, group, n_skill, subset="train", max_seq=MAX_SEQ):
        super(SAKTDatasetNew, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill # 13523
        self.samples = group
        self.subset = subset
        
        # self.user_ids = [x for x in group.index]
        self.user_ids = []
        for user_id in group.index:
            '''
            q: question_id
            pqt: previous question time
            pqe: previous question explain or not
            qa: question answer correct or not
            '''
            q, pqt, pqe, qa = group[user_id] 
            if len(q) < 5: # 5 interactions minimum
                continue
            self.user_ids.append(user_id) # user_ids indexes

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index] # Pick a user
        q_, pqt_, pqe_, qa_ = self.samples[user_id] # Pick full sequence for user
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        pqt = np.zeros(self.max_seq, dtype=int)
        pqe = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            if self.subset == "train":
                # if seq_len > self.max_seq:
                if random() > 0.1:
                    random_start_index = randint(0, seq_len - self.max_seq)
                    '''
                    Pick 100 questions, answers, prior question time, 
                    priori question explain from a random index
                    '''
                    q[:] = q_[random_start_index:random_start_index + self.max_seq] 
                    qa[:] = qa_[random_start_index:random_start_index + self.max_seq] 
                    pqt[:] = pqt_[random_start_index:random_start_index + self.max_seq] 
                    pqe[:] = pqe_[random_start_index:random_start_index + self.max_seq] 
                else:
                    q[:] = q_[-self.max_seq:]
                    qa[:] = qa_[-self.max_seq:]
                    pqt[:] = pqt_[-self.max_seq:] 
                    pqe[:] = pqe_[-self.max_seq:]
            else:
                q[:] = q_[-self.max_seq:] # Pick last 100 questions
                qa[:] = qa_[-self.max_seq:] # Pick last 100 answers
                pqt[:] = pqt_[-self.max_seq:] 
                pqe[:] = pqe_[-self.max_seq:]
        else:
            if random()>0.1:
                seq_len = randint(2,seq_len)
                q[-seq_len:] = q_[:seq_len]
                qa[-seq_len:] = qa_[:seq_len]
                pqt[-seq_len:] = pqt_[:seq_len]
                pqe[-seq_len:] = pqe_[:seq_len]
            else:
                q[-seq_len:] = q_ # Pick last N question with zero padding
                qa[-seq_len:] = qa_ # Pick last N answers with zero padding
                pqt[-seq_len:] = pqt_
                pqe[-seq_len:] = pqe_      
                
        target_id = q[1:] # Ignore first item 1 to 99
        label = qa[1:] # Ignore first item 1 to 99
        prior_q_time = pqt[1:]
        prior_q_explain = pqe[1:]


        # x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy() # 0 to 98
        x += (qa[:-1] == 1) * self.n_skill # y = et + rt x E

        return x, target_id,  label,  prior_q_time,  prior_q_explain


class TestDataset(Dataset):
    def __init__(self, samples, test_df, n_skills, max_seq=MAX_SEQ):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.n_skill = n_skills
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

class TestDatasetNew(Dataset):
    def __init__(self, samples, test_df, n_skills, max_seq=MAX_SEQ):
        super(TestDatasetNew, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.n_skill = n_skills
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]
        prior_q_time = test_info["prior_question_elapsed_time"]
        prior_q_explain = test_info["prior_question_had_explanation"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        pqt = np.zeros(self.max_seq, dtype=int)
        pqe = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, pqt_, pqe_, qa_ = self.samples[user_id]
            
            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
                pqt = pqt_[-self.max_seq:]
                pqe = pqe_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_  
                pqt[-seq_len:] = pqt_
                pqe[-seq_len:] = pqe_                
        
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill
        
        questions = np.append(q[2:], [target_id])

        # prior_q_time = pqt[1:]  # this only stores the current one CV -0.002
        # prior_q_explain = pqe[1:] #  this only stores the current one
        prior_q_time = np.append(pqt[2:], [prior_q_time])
        prior_q_explain = np.append(pqe[2:], [prior_q_explain])
        
        return x, questions, prior_q_time, prior_q_explain

class FFN(nn.Module):
    def __init__(self, state_size=NUM_EMBED, nc=1):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size//nc)
        self.relu = nn.ReLU()
        # self.leakyrelu = nn.LeakyReLU()
        self.lr2 = nn.Linear(state_size//nc, state_size//nc)
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
    def __init__(self, n_skill, 
                       max_seq=MAX_SEQ, 
                       embed_dim=NUM_EMBED, 
                       num_heads=NUM_HEADS,
                       num_layers=NUM_LAYERS):
        super(SAKTModel, self).__init__()
        self.__name__ = 'sakt'
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim, nc=1)
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
        att_weight = None 

        for _ in range(self.num_layers):
            x, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
            x = self.layer_normal(x + e)
        x = x.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        output = self.ffn(x)
        # x = self.layer_normal(x + output) # original
        x = self.layer_normal(output) # if nc>=2 get rid of the skip connection
        x = self.pred(x)

        return x.squeeze(-1), att_weight

class SAKTModelNew(nn.Module):
    def __init__(self, n_skill, 
                       max_seq=MAX_SEQ, 
                       embed_dim=NUM_EMBED, 
                       num_heads=NUM_HEADS, 
                    #    num_hid=NUM_HIDDEN,
                       num_layers=NUM_LAYERS,
                       ):
        super(SAKTModelNew, self).__init__()
        self.__name__ = 'saktnew'
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        # embedding of prior question time
        self.pqt_embedding = nn.Embedding(NUM_TIME+1, embed_dim) 
        # embedding of priori question answered
        self.pa_embedding = nn.Embedding(2+1, embed_dim) 

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 
        self.num_layers = num_layers
        self.ffn = FFN(embed_dim, nc=1)
        # self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
        # self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x, question_ids, prior_question_time=None, prior_question_explain=None):
        '''
        x: encoded performance on all previous questions for a user
        pos_id: ???

        Attention:
        query: e
        key: x
        value: x

        To do: multiple attention layers
        '''
        device = x.device        
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)

        pq_x = self.pqt_embedding(prior_question_time)
        pa_x = self.pa_embedding(prior_question_explain)

        x += pos_x + pq_x + pa_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_weight = None

        for _ in range(self.num_layers):
            x, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
            x = self.layer_normal(x + e)

        x = x.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        output = self.ffn(x)
        x = self.layer_normal(x + output) # original
        # x = self.leakyrelu(x)
        x = self.pred(x)

        return x.squeeze(-1), att_weight


class SAKTMulti(nn.Module):
    def __init__(self, n_skill, max_seq=MAX_SEQ, embed_dim=NUM_EMBED, num_heads=NUM_HEADS):
        super(SAKTMulti, self).__init__()
        self.__name__ = 'saktmulti'
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        # embedding of prior question time
        self.pqt_embedding = nn.Embedding(NUM_TIME+1, embed_dim) 
        # embedding of priori question answered
        self.pa_embedding = nn.Embedding(2+1, embed_dim) 

        self.multi_att1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
        self.multi_att2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.2)
        self.multi_att3 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.4)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(3*embed_dim)
        self.layer_normal1 = nn.LayerNorm(3*embed_dim) 
        self.fc1 = nn.Linear(3*embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, question_ids, prior_question_time=None, prior_question_explain=None):
        '''
        x: encoded performance on all previous questions for a user
        pos_id: ???

        Attention:
        query: e
        key: x
        value: x

        To do: multiple attention layers
        '''
        device = x.device        
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x += pos_x

        e = self.e_embedding(question_ids)
        pq_x = self.pqt_embedding(prior_question_time)
        pa_x = self.pa_embedding(prior_question_explain)

        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        pq_x = pq_x.permute(1, 0, 2)
        pa_x = pa_x.permute(1, 0, 2)

        att_mask = future_mask(x.size(0)).to(device)

        att_output1, att_weight1 = self.multi_att1(e, x, x, attn_mask=att_mask)
        att_output1 = self.layer_normal(att_output1 + e)

        att_output2, att_weight2 = self.multi_att2(pq_x, x, x, attn_mask=att_mask)
        att_output2 = self.layer_normal(att_output2 + pq_x)

        att_output3, att_weight3 = self.multi_att3(pa_x, x, x, attn_mask=att_mask)
        att_output3 = self.layer_normal(att_output3 + pa_x)

        att_output1 = att_output1.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        att_output2 = att_output2.permute(1, 0, 2)
        att_output3 = att_output3.permute(1, 0, 2)

        att_output = torch.cat((att_output1, att_output2, att_output3), dim=2)
        x = self.ffn(att_output)
        x = self.dropout(x)
        # x = self.relu(x)
        # x = self.layer_normal1(x) + att_output # config 1: after normalization then residual
        x = self.layer_normal1(x + att_output) # config 2: first residual then normalization
        x = self.fc1(x)
        x = self.fc2(x)

        return x.squeeze(-1), [att_weight1, att_weight2, att_weight3]

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
            # print(f'X shape: {x.shape}, target_id shape: {target_id.shape}')
            loss = criterion(output, label)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())

            output = output[:, -1]
            label = label[:, -1] 
            output = torch.sigmoid(output)
            pred = (output >= 0.5).long()
            
            num_corrects += (pred == label).sum().item()
            num_total += len(label)

            labels.extend(label.view(-1).data.cpu().numpy())
            outs.extend(output.view(-1).data.cpu().numpy())

            if idx % TQDM_INT == 0:
                pbar.set_description(f'train loss - {train_loss[-1]:.4f}')
                pbar.update(TQDM_INT)
    
    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc

def valid_epoch(model, valid_iterator, criterion, device="cuda", conf=None):
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

        output = conf.SCALING*output[:, -1] # (BS, 1)
        label = label[:, -1] 
        output = torch.sigmoid(output)
        pred = ( output >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(valid_loss)

    return loss, acc, auc

def train_epoch_new(model, train_iterator, optim, criterion, device="cuda"):
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
            prior_q_time = item[3].to(device).long()
            priori_q_explain = item[4].to(device).long()
            label = item[2].to(device).float()

            optim.zero_grad()
            output, atten_weight = model(x, target_id, prior_q_time, priori_q_explain)
            # print(f'X shape: {x.shape}, target_id shape: {target_id.shape}')
            loss = criterion(output, label)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())

            output = output[:, -1]
            label = label[:, -1] 
            output = torch.sigmoid(output)
            pred = ( output >= 0.5).long()
            
            num_corrects += (pred == label).sum().item()
            num_total += len(label)

            labels.extend(label.view(-1).data.cpu().numpy())
            outs.extend(output.view(-1).data.cpu().numpy())

            if idx % TQDM_INT == 0:
                pbar.set_description(f'train loss - {train_loss[-1]:.4f}')
                pbar.update(TQDM_INT)
    
    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc

def valid_epoch_new(model, valid_iterator, criterion, device="cuda", conf=None):
    model.eval()

    valid_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for item in valid_iterator: 
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        prior_q_time = item[3].to(device).long()
        priori_q_explain = item[4].to(device).long()
        label = item[2].to(device).float()

        with torch.no_grad():
            output, atten_weight = model(x, target_id, prior_q_time, priori_q_explain)
        loss = criterion(output, label)
        valid_loss.append(loss.item())

        output = conf.SCALING*output[:, -1] # (BS, 1)
        label = label[:, -1] 
        output = torch.sigmoid(output)
        pred = (output >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(valid_loss)

    return loss, acc, auc



def run_train(model, train_iterator, valid_iterator, optim, criterion, scheduler=None,
              epochs=60, device="cuda", conf=None):
    
    scheduler_name = str(scheduler.__class__) if scheduler is not None else ''
    history = []
    auc_max = 0
    best_epoch = 0
    val_auc = 0.5

    for epoch in range(epochs):

        tqdm.write(f"\n\n")
        model.train()

        train_loss = []
        num_corrects = 0
        num_total = 0
        over_fit = 0
        labels = []
        outs = []

        len_dataset = len(train_iterator)

        with tqdm(total=len_dataset) as pbar:
            for idx, item in enumerate(train_iterator): 
                x = item[0].to(device).long()
                target_id = item[1].to(device).long()
                label = item[2].to(device).float()

                optim.zero_grad()
                output, _ = model(x, target_id)
                # print(f'X shape: {x.shape}, target_id shape: {target_id.shape}')
                loss = criterion(output, label)
                loss.backward()
                optim.step()
                train_loss.append(loss.item())

                output = output[:, -1]
                label = label[:, -1] 
                output = torch.sigmoid(output)
                pred = (output >= 0.5).long()
                
                num_corrects += (pred == label).sum().item()
                num_total += len(label)

                labels.extend(label.view(-1).data.cpu().numpy())
                outs.extend(output.view(-1).data.cpu().numpy())

                if idx % TQDM_INT == 0:
                    descr = f'[Epoch {epoch}/{epochs}]:  train loss - {train_loss[-1]:.6f}   '
                    descr += f"best val auc - {auc_max:.6f} at epoch {best_epoch}"
                    pbar.set_description(descr)
                    pbar.update(TQDM_INT)
        # end of one epoch
        if val_auc and scheduler is not None:
            if 'ReduceLROnPlateau' in scheduler_name:
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        train_acc = num_corrects / num_total
        train_auc = roc_auc_score(labels, outs)
        train_loss = np.mean(train_loss)

        tqdm.write(f"\nTrain: loss - {train_loss:.4f} acc - {train_acc:.4f} auc - {train_auc:.4f}")

        val_loss, val_acc, val_auc = valid_epoch(model, valid_iterator, criterion, 
                                                 device=device, conf=conf)
        tqdm.write(f"\nValid: loss - {val_loss:.4f} acc - {val_acc:.4f} auc - {val_auc:.4f}")

        lr = optim.param_groups[0]['lr']

        tqdm.write(f'\nCurrent learning rate: {lr:4e}')

        history.append({"epoch":epoch, "lr": lr, 
                        **{"train_auc": train_auc, "train_acc": train_acc}, 
                        **{"valid_auc": val_auc, "valid_acc": val_acc}})
        
        if val_auc > auc_max:
            print(f"\n[Epoch {epoch}/{epochs}] auc improved from {auc_max:.6f} to {val_auc:.6f}") 
            auc_max = val_auc
            best_epoch = epoch
            over_fit = 0
            if val_auc > conf.SAVING_THRESHOLD:
                model_name = f"{model.__name__}_head_{conf.NUM_HEADS}_embed_{conf.NUM_EMBED}_seq_{conf.MAX_SEQ}"
                model_name += f"_auc_{val_auc:.4f}.pt"
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))
                print("Best epoch model saved.\n\n")
        else:
            over_fit += 1

        if over_fit >= conf.PATIENCE:
            print(f"\nEarly stop epoch at {epoch}")
            break
    return model, history


def run_train_new(model, train_iterator, valid_iterator, optim, criterion, 
              scheduler=None, epochs=60, device="cuda", conf=None):
    history = []
    scheduler_name = str(scheduler.__class__) if scheduler is not None else ''
    auc_max = 0
    best_epoch = 0
    val_loss = 0 
    val_auc = 0.5

    for epoch in range(epochs):

        tqdm.write("\n\n")
        model.train()

        train_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outs = []
        over_fit = 0

        len_dataset = len(train_iterator)

        with tqdm(total=len_dataset) as pbar:
            for idx, item in enumerate(train_iterator): 
                x = item[0].to(device).long()
                target_id = item[1].to(device).long()
                prior_q_time = item[3].to(device).long()
                priori_q_explain = item[4].to(device).long()
                label = item[2].to(device).float()

                optim.zero_grad()
                output, _ = model(x, target_id, prior_q_time, priori_q_explain)
                # print(f'X shape: {x.shape}, target_id shape: {target_id.shape}')
                loss = criterion(output, label)
                loss.backward()
                optim.step()
                train_loss.append(loss.item())

                output = output[:, -1]
                label = label[:, -1] 
                output = torch.sigmoid(output)
                pred = (output >= 0.5).long()
                
                num_corrects += (pred == label).sum().item()
                num_total += len(label)

                labels.extend(label.view(-1).data.cpu().numpy())
                outs.extend(output.view(-1).data.cpu().numpy())

                if idx % TQDM_INT == 0:
                    descr = f'[Epoch {epoch}/{epochs}]:  train loss - {train_loss[-1]:.6f}   '
                    descr += f"best val auc - {auc_max:.6f} at epoch {best_epoch}"
                    pbar.set_description(descr)
                    pbar.update(TQDM_INT)
        
        # end of one epoch
        if val_auc and scheduler is not None:
            if 'ReduceLROnPlateau' in scheduler_name:
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        train_acc = num_corrects / num_total
        train_auc = roc_auc_score(labels, outs)
        train_loss = np.mean(train_loss)

        tqdm.write(f"\nTrain: loss - {train_loss:.4f} acc - {train_acc:.4f} auc - {train_auc:.4f}")

        val_loss, val_acc, val_auc = valid_epoch_new(model, valid_iterator, criterion, 
                                                     device=device, conf=conf)
        tqdm.write(f"\nValid: loss - {val_loss:.4f} acc - {val_acc:.4f} auc - {val_auc:.4f}")

        lr = optim.param_groups[0]['lr']

        tqdm.write(f'\nCurrent learning rate: {lr:4e}')

        history.append({"epoch":epoch, "lr": lr, 
                        **{"train_auc": train_auc, "train_acc": train_acc}, 
                        **{"valid_auc": val_auc, "valid_acc": val_acc}})
        
        if val_auc > auc_max:
            print(f"\n[Epoch {epoch}/{epochs}] auc improved from {auc_max:.6f} to {val_auc:.6f}") 
            auc_max = val_auc
            best_epoch = epoch
            over_fit = 0
            if val_auc > conf.SAVING_THRESHOLD:
                model_name = f"{model.__name__}_head_{conf.NUM_HEADS}_embed_{conf.NUM_EMBED}_seq_{conf.MAX_SEQ}"
                model_name += f"_auc_{val_auc:.4f}.pt"
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))
                print("\n\nBest epoch model saved.\n\n")
        else:
            over_fit += 1

        if over_fit >= conf.PATIENCE:
            print(f"\nEarly stop epoch at {epoch}")
            break
    return model, history


def load_sakt_model(model_file, device='cuda'):
    # creating the model and load the weights
    configs = []
    model_file_lst = model_file.split('_')
    for c in ['head', 'embed', 'seq']:
        idx = model_file_lst.index(c) + 1
        configs.append(int(model_file_lst[idx]))

    # configs.append(int(model_file[model_file.rfind('head')+5]))
    # configs.append(int(model_file[model_file.rfind('embed')+6:model_file.rfind('embed')+9]))
    # configs.append(int(model_file[model_file.rfind('seq')+4:model_file.rfind('seq')+7]))
    conf_dict = dict(n_skill=NUM_SKILLS,
                     num_heads=configs[0],
                     embed_dim=configs[1], 
                     max_seq=configs[2], 
                     )

    if 'saktmulti' in model_file:
        model = SAKTMulti(**conf_dict)
        conf_dict['model_name'] = 'SaktMulti'
    elif 'saktnew' in model_file:
        model = SAKTModelNew(**conf_dict)
        conf_dict['model_name'] = 'SaktNew'
    else:
        model = SAKTModel(**conf_dict)
        conf_dict['model_name'] = 'SaktOrig'
        
    model = model.to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))

    return model, conf_dict

def find_sakt_model(model_dir=MODEL_DIR, model_file=None, model_type=None):
    # find the best AUC model, or a given model
    if model_type is None: model_type = 'sakt'

    if model_file is None:
        try:
            model_files = find_files(model_type, model_dir)
            auc_scores = [s.rsplit('.')[-2] for s in model_files]
            model_file = model_files[argmax(auc_scores)]
        except:
            FileExistsError('Model file not found.')
    return model_file


class HNAGOptimizer(Optimizer):
    """
    Implements a Hessian-driven Nesterov Accelerated Gradient Algorithm.
    First order optimization methods based on Hessian-driven Nesterov accelerated gradient flow, 
    https://arxiv.org/abs/1912.09276
    only 1 gradient evaluation per iteration in this version

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        mu (float, optional): estimated convexity of the loss (default: 0.5)
        hessian(bool, optional): enables Hessian-driven momentum (default: True)
        
    Example:
        >>> optimizer = HNAGOptimizer(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=1e-3, mu=0.5,
                       weight_decay=0, hessian=True):
        defaults = dict(lr=lr, 
                        mu=mu,
                        weight_decay=weight_decay, 
                        hessian=hessian)
        if hessian and (lr is None):
            raise ValueError("Hessian-driven method requires a specific learning rate.")
        super(HNAGOptimizer, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): 
                A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        
        if closure is not None:
            # reserved for H-Nag to evaluate the model before next iteration
            loss = closure()
            # print(loss.item())

        for _, group in enumerate(self.param_groups):
            # print(self.param_groups[0])
            weight_decay = group['weight_decay']
            mu = group['mu']
            lr = group['lr']

            for _, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad_x = p.grad.data.detach()
                if group['weight_decay'] != 0:
                    grad_x = grad_x.add(weight_decay, p.data)
                
                param_state = self.state[p]

                x = p.data.detach()
                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0 # for debugging
                    param_state['v'] = torch.zeros_like(p.data)
                    param_state['y'] = torch.zeros_like(p.data)

                    alpha  = (2*lr)**(0.5) # approximated root
                    if not 0 < alpha < 1:
                        raise ValueError("alpha has to be in (0,1).")
                    param_state['alpha'] = alpha

                v = param_state['v']
                y = param_state['y']
                alpha = param_state['alpha']

                v_new = (alpha*v + 2*lr*x - 2*lr*grad_x/mu)/(alpha + 2*lr)

                y_new = x - 2*lr*grad_x/mu
                alpha_new = ((alpha**2 + 2*alpha*lr)/(1+alpha))**(0.5)
                x = (y_new + alpha_new*v_new)/(1+alpha_new)


                p.data = x
                param_state['alpha'] = alpha_new
                param_state['v'] = v_new
                param_state['y'] = y_new

        return loss

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAKTModel(NUM_SKILLS, embed_dim=NUM_EMBED)
    num_params = get_num_params(model)
    print(f"\nVanilla SAKT # params: {num_params}")
    # print(model)
    # model_file = find_sakt_model()
    # if model_file is not None: 
    #     model = load_sakt_model(model_file)
    #     print(f"Loaded {model_file}.")

    model = SAKTModelNew(NUM_SKILLS, embed_dim=NUM_EMBED)
    num_params = get_num_params(model)
    print(f"\nNew SAKT with more embeddings # params: {num_params}")
    # print(model)

    model = SAKTMulti(NUM_SKILLS, embed_dim=NUM_EMBED)
    num_params = get_num_params(model)
    print(f"\nMulti Attention layer SAKT # params: {num_params}")
    # print(model)

    model_file = find_sakt_model()
    print(f"\nFound model at {model_file}")
    model, conf_dict = load_sakt_model(model_file)
    print(f"\nLoaded model with {conf_dict}")

