import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from typing import List
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

FILLNA_VAL = 100
LAST_N = 100
TQDM_INT = 8
PAD = 0
FOLD = 1
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/fold{FOLD}/snapshots/'
N_EXERCISES = 13523 #  train_df['content_id'].unique()


class TransformerModel(nn.Module):

    def __init__(self, ninp:int=32, nhead:int=2, nhid:int=64, nlayers:int=2, dropout:float=0.3):
        '''
        nhead -> number of heads in the transformer multi attention thing.
        nhid -> the number of hidden dimension neurons in the model.
        nlayers -> how many layers we want to stack.
        '''
        super(TransformerModel, self).__init__()
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.exercise_embeddings = nn.Embedding(num_embeddings=N_EXERCISES, embedding_dim=ninp) # exercise_id
        self.pos_embedding = nn.Embedding(ninp, ninp) # positional embeddings
        self.part_embeddings = nn.Embedding(num_embeddings=7+1, embedding_dim=ninp) # part_id_embeddings
        self.prior_question_elapsed_time = nn.Embedding(num_embeddings=301, embedding_dim=ninp) # prior_question_elapsed_time
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # init embeddings
        self.exercise_embeddings.weight.data.uniform_(-initrange, initrange)
        self.part_embeddings.weight.data.uniform_(-initrange, initrange)
        self.prior_question_elapsed_time.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, content_id, part_id, prior_question_elapsed_time=None, mask_src=None):
        '''
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        padding mask is (N, S) with boolean True/False.
        SRC_MASK is (S, S) with float(’-inf’) and float(0.0).
        '''

        embedded_src = self.exercise_embeddings(content_id) + \
        self.pos_embedding(torch.arange(0, content_id.shape[1]).to(self.device).unsqueeze(0).repeat(content_id.shape[0], 1)) + \
        self.part_embeddings(part_id) + self.prior_question_elapsed_time(prior_question_elapsed_time) # (N, S, E)
        embedded_src = embedded_src.transpose(0, 1) # (S, N, E)
        
        _src = embedded_src * np.sqrt(self.ninp)
        
        output = self.transformer_encoder(src=_src, src_key_padding_mask=mask_src)
        output = self.decoder(output)
        output = output.transpose(1, 0)
        return output

def pad_seq(seq: List[int], max_batch_len: int = LAST_N, pad_value: int = True) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]

def train_epoch(model, train_iterator, optimizer, criterion):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    label_all = []
    pred_all = []
    len_dataset = len(train_iterator)

    with tqdm(total=len_dataset) as pbar:
        for idx,batch in enumerate(train_iterator):
            content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
            content_id = Variable(content_id.cuda())
            part_id = Variable(part_id.cuda())
            prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
            mask = Variable(mask.cuda())
            labels = Variable(labels.cuda().long())
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(mode=True):
                output = model(content_id, part_id, prior_question_elapsed_time, mask)
                # output is (N,S,2) # i am working on it
                
                # loss = criterion(output[:,:,1], labels) # BCEWithLogitsLoss
                loss = criterion(output.reshape(-1, 2), labels.reshape(-1)) # Flatten and use crossEntropy
                loss.backward()
                optimizer.step()

                train_loss.append(loss.cpu().detach().data.numpy())

            pred_probs = torch.softmax(output[~mask], dim=1)
            pred = torch.argmax(pred_probs, dim=1)
            labels = labels[~mask]
            num_corrects += (pred == labels).sum().item()
            num_total += len(labels)

            label_all.extend(labels.reshape(-1).data.cpu().numpy())
            # pred_all.extend(pred.reshape(-1).data.cpu().numpy())
            pred_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy()) # use probability to do auc

            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)

    acc = num_corrects / num_total
    auc = roc_auc_score(label_all, pred_all)
    loss = np.mean(train_loss)

    return loss, acc, auc


def valid_epoch(model, valid_iterator, criterion):
    model.eval()
    valid_loss = []
    num_corrects = 0
    num_total = 0
    label_all = []
    pred_all = []
    len_dataset = len(valid_iterator)

    with tqdm(total=len_dataset) as pbar:
        for idx, batch in enumerate(valid_iterator):
            content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
            content_id = Variable(content_id.cuda())
            part_id = Variable(part_id.cuda())
            prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
            mask = Variable(mask.cuda())
            labels = Variable(labels.cuda().long())
            with torch.set_grad_enabled(mode=False):
                output = model(content_id, part_id, prior_question_elapsed_time, mask)
                loss = criterion(output.reshape(-1, 2), labels.reshape(-1)) # Flatten and use crossEntropy
            # crossEntropy loss
            valid_loss.append(loss.cpu().detach().data.numpy())
            pred_probs = torch.softmax(output[~mask], dim=1)
            # pred = (output_prob >= 0.50)
            pred = torch.argmax(pred_probs, dim=1)
            
            # BCE loss
            # output_prob = output[:,:,1]
            # pred = (output_prob >= 0.50)
            # print(output.shape, labels.shape) # torch.Size([N, S, 2]) torch.Size([N, S])
            # _, predicted_classes = torch.max(output[:,:,].data, 1)
            
            labels = labels[~mask]
            num_corrects += (pred == labels).sum().item()
            num_total += len(labels)
            label_all.extend(labels.reshape(-1).data.cpu().numpy())
            # pred_all.extend(pred.reshape(-1).data.cpu().numpy())
            pred_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy()) # use probability to do auc
            
            if idx % TQDM_INT == 0:
                pbar.update(TQDM_INT)

    acc = num_corrects / num_total
    auc = roc_auc_score(label_all, pred_all)
    loss = np.mean(valid_loss)

    return loss, acc, auc


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(ninp=LAST_N, nhead=4, nhid=128, nlayers=3, dropout=0.3)
    model = model.to(device)
    print("Loading state_dict...")
    model.load_state_dict(torch.load(MODEL_DIR+'model_best_epoch.pt', map_location=device))
    model.eval()
    print(model)