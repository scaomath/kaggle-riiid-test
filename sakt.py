from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from torchsummary import summary
from utils import *

MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'


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

class conf:
    METRIC_ = "max"
    TQDM_INT = 10
    WORKERS = 8 # 0
    BATCH_SIZE = 512
    VAL_BATCH_SIZE = 4096
    lr = 1e-3
    NUM_EMBED = 160
    NUM_HEADS = 10
    NUM_SKILLS = 13523 # len(skills)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

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
    def __init__(self, n_skill, max_seq=100, embed_dim=conf.NUM_EMBED, num_heads=conf.NUM_HEADS):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.3)

        self.dropout = nn.Dropout(0.3)
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
            pred = (torch.sigmoid(output) >= 0.5).long()
            
            num_corrects += (pred == label).sum().item()
            num_total += len(label)

            labels.extend(label.view(-1).data.cpu().numpy())
            #outs.extend(output.view(-1).data.cpu().numpy())
            outs.extend(torch.sigmoid(output).view(-1).data.cpu().numpy())

            if idx % conf.TQDM_INT == 0:
                pbar.set_description(f'loss - {train_loss[-1]:.4f}')
                pbar.update(conf.TQDM_INT)
    
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


def load_sakt_model(model_file, device='cuda'):
    # creating the model and load the weights
    model = SAKTModel(conf.NUM_SKILLS, embed_dim=conf.NUM_EMBED)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))

    return model

def find_sakt_model(model_dir=MODEL_DIR, model_file=None):
    # find the best AUC model, or a given model
    if model_file is None:
        try:
            model_files = find_files('sakt', model_dir)
            tmp = [s.rsplit('.')[-2] for s in model_files]
            model_file = model_files[argmax(tmp)]
        except:
            FileExistsError('model file not found')
    return model_file

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAKTModel(conf.NUM_SKILLS, embed_dim=conf.NUM_EMBED)
    num_params = get_num_params(model)
    print(f"Params number: {num_params}")
    model_file = find_sakt_model()
    if model_file is not None: model = load_sakt_model(model_file)