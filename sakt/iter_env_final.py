#%%
import sys
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(HOME) 
MODEL_DIR = HOME+'/model/'
DATA_DIR = HOME+'/data/'
sys.path.append(HOME)

from utils import *
from iter_env import *

PREPROCESS = False
NUM_SKILLS = 13523 # number of problems
MAX_SEQ = 180
ACCEPTED_USER_CONTENT_SIZE = 4
EMBED_SIZE = 128
RECENT_SIZE = 12 # recent data in the training loop
NUM_HEADS = 8
BATCH_SIZE = 128
VAL_BATCH_SIZE = 2048
TEST_SIZE = 25_000
DROPOUT = 0.1
SEED = 1127

#%%

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

TRAIN_DTYPES = {TIMESTAMP: 'int64', 
         USER_ID: 'int32', 
         CONTENT_ID: 'int16',
         CONTENT_TYPE_ID: 'bool',
         TARGET:'int8',
         PRIOR_QUESTION_TIME: np.float32,
         PRIOR_QUESTION_EXPLAIN: 'boolean'}
#%%

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
# model_file = MODEL_DIR+'sakt_seq_180_auc_0.7689.pth' # current sakt in the pipeline, LB 0.776
model_file = MODEL_DIR+'sakt_seq_180_auc_0.7696.pt' # trained using the new trainer
model = SAKTModel(n_skill=NUM_SKILLS, 
                max_seq=MAX_SEQ, 
                embed_dim=EMBED_SIZE, 
                heads=NUM_HEADS,
                enc_layers=1)
        
model = model.to(device)
model.load_state_dict(torch.load(model_file, map_location=device))

model.eval()

#%%
with timer("Loading train and valid"):
    train_df = pd.read_parquet(DATA_DIR+'cv2_train.parquet',
                                        columns=list(TRAIN_DTYPES.keys()))
    train_df = train_df.astype(TRAIN_DTYPES)
    all_test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    all_test_df = all_test_df.astype(TRAIN_DTYPES)
    all_test_df = all_test_df[:TEST_SIZE]

if PREPROCESS:
    with timer("Processing cv2_train user group"):
        train_df = train_df[TRAIN_DTYPES.keys()]
        train_df = train_df[train_df[CONTENT_TYPE_ID] == False].reset_index(drop = True)
        group = train_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
            .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))
else:
    with timer('Loading cv2_train user group'):
        with open(os.path.join(DATA_DIR, 'sakt_group_cv2.pickle'), 'rb') as f:
            group = pickle.load(f)

iter_test = Iter_Valid(all_test_df, max_user=1000)
prev_test_df= None
predicted = []
def set_predict(df):
    predicted.append(df)
#%%
len_test = len(all_test_df)
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

y_true = all_test_df[all_test_df.content_type_id == 0].answered_correctly
y_pred = pd.concat(predicted).answered_correctly
print(f'\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))


# %%
