#%%
import gc
import sys

import pickle

import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau, CyclicLR)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sns.set()
DEFAULT_FIG_WIDTH = 20
sns.set_context("paper", font_scale=1.2) 
HOME = "/home/scao/Documents/kaggle-riiid-test/"
sys.path.append(HOME) 
from utils import *
from sakt import *
from transformer_optimizers import *

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

which is an implementation of this paper: https://arxiv.org/pdf/1907.06837.pdf

To-do and 
Version Notes

1. check the bias of the prediction

2. replicate the no cv result on kaggle with LB 0.771


3. A two layer attention with shared weights
Change the last several layers to embed_dim->embed_dim//2 ->1, deleted the skip connection.
seq len=150, embed_dim = 256 with 8 heads, CV 0.7577; iter_env CV 0.7270

4. One layer attention, no skip connection with layer norm, seq len 150, embed_dim 128, 8 heads, CV scaling = 2, LR = 1e-3, CV 0.7604; iter_env CV 0.7291

5. One layer attention, no skip connection with layer norm, seq len 150, embed_dim 128, 8 heads, CV scaling = 3.5, LR = 1e-4 with a scheduler, CV 0.7552, (deleted)

6. One layer attention, seq len 150, embed_dim 128, 8 heads, CV scaling = 2, LR = 1e-3, CV 0.7605; iter_env CV 0.7318
If there is layer normalization, multiplying with a scaling factor does not matter much for AUC

7. Same with 6, embed_dim 160, head 10, CV scaling 2, CV 0.7579

8. Same with 6 Label smoothing with a factor of 0.2, CV scaling = 4, CV < 0.73....

9. Testing adding a "User growth" feature to the embedding...
'''

class conf:
    METRIC_ = "max"
    FILLNA_VAL = 14_000 # for prior question elapsed time, rounded average in train
    TQDM_INT = 8
    WORKERS = 8 # 0
    SCALING = 4 # scaling before sigmoid for valid
    LABEL_SMOOTHING = True # whether to do label smoothing
    SMOOTHING_FACTOR = 0.2 # the label smoothing factor for train
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1024
    VAL_BATCH_SIZE = 4096
    MAX_SEQ = 150
    NUM_EMBED = 128
    NUM_LAYERS = 1
    NUM_HEADS = 8
    NUM_SKILLS = 13523 # len(skills)
    NUM_TIME = 300 # when scaled by 1000 and round, priori question time's unique values
    PATIENCE = 6 # overfit patience
    SAVING_THRESHOLD = 0.754  # the threshold for auc to save a model

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

TQDM_INT = 8 
HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
FOLD = 1

TRAIN = True
PREPROCESS = 2
EPOCHS = 70
LEARNING_RATE = conf.LEARNING_RATE
DATE_STR = get_date()

#%%
if PREPROCESS == 1:
    print("\nLoading train...")
    train_df = pd.read_csv(DATA_DIR+'train.csv', usecols=[1, 2, 3, 4, 7], dtype=TRAIN_DTYPES)
    print(f'Loaded train.')
    print(TRAIN_DTYPES)
    train_df = train_df[train_df[CONTENT_TYPE_ID] == False]
    train_df = train_df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)
    group = train_df[[USER_ID, CONTENT_ID, TARGET]]\
                .groupby(USER_ID)\
                .apply(lambda r: (r[CONTENT_ID].values, 
                                  r[TARGET].values))
    train_group, valid_group = train_test_split(group, test_size=0.1)
elif PREPROCESS == 2: # using preprocessed files
    print("\nLoading train from parquet...")
    train_df = pd.read_parquet(DATA_DIR+'cv3_train.parquet')
    valid_df = pd.read_parquet(DATA_DIR+'cv3_valid.parquet')
    print(f'Loaded train.')
    train_df = train_df[TRAIN_COLS]
    valid_df = valid_df[TRAIN_COLS]
    train_group = preprocess(train_df, 
                  label_smoothing=conf.LABEL_SMOOTHING, smoothing_factor=conf.SMOOTHING_FACTOR)
    valid_group = preprocess(valid_df, train_flag=2, label_smoothing=False)

    print("valid:", valid_df.shape, "users:", valid_df[USER_ID].nunique())
    print("train:", train_df.shape, "users:", train_df[USER_ID].nunique())
else:
    print("\nLoading preprocessed file...")
    with open(DATA_DIR+'sakt_data_new.pickle', 'rb') as f:
        group = pickle.load(f)
    train_group, valid_group = train_test_split(group, test_size=0.1)

# skills = train_df[CONTENT_ID].unique()
n_skill = conf.NUM_SKILLS #len(skills) # len(skills) might not have all
print("\nNumber of skills", n_skill)


print("\nvalid by user:", len(valid_group))
print("train by user:", len(train_group))

# %%
train_dataset = SAKTDataset(train_group, n_skill)
train_loader = DataLoader(train_dataset, 
                              batch_size=conf.BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=conf.WORKERS)

valid_dataset = SAKTDataset(valid_group, n_skill)
val_loader = DataLoader(valid_dataset, 
                        batch_size=conf.VAL_BATCH_SIZE, 
                        shuffle=False, drop_last=False,
                        num_workers=conf.WORKERS)

item = train_dataset.__getitem__(1)
sample = next(iter(train_loader))

print("\n\nx len", len(item[0]),'\n', item[0], '\n\n')
print("\n\ntarget_id len", len(item[1]),'\n', item[1] , '\n\n')
print("\n\nlabel len", len(item[2]), '\n', item[2])

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAKTModel(n_skill, embed_dim=conf.NUM_EMBED, 
                           max_seq=conf.MAX_SEQ,
                           num_heads=conf.NUM_HEADS,
                           num_layers=conf.NUM_LAYERS)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2,
#                               patience=conf.PATIENCE,
#                               cooldown=conf.PATIENCE//2, 
#                               threshold=1e-5, verbose=1)
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, t_total=EPOCHS)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=LEARNING_RATE*1e-2, verbose=1)
# scheduler = CyclicLR(optimizer, base_lr=1e-1*LEARNING_RATE, max_lr=LEARNING_RATE, step_size_up=5,mode="triangular2", verbose=1)
# optimizer = HNAGOptimizer(model.parameters(), lr=1e-3) 
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)
num_params = get_num_params(model)
print(f"\n\n# heads      : {conf.NUM_HEADS}")
print(f"# embed      : {conf.NUM_EMBED}")
print(f"# att layers : {conf.NUM_LAYERS}")
print(f"# seqlen     : {conf.MAX_SEQ}")
print(f"# params     : {num_params}")
# %%

if TRAIN:
    losses = []
    history = []
    auc_max = -np.inf
    over_fit = 0

    print("\nTraining...:\n")
    model, history = run_train(model, train_loader, val_loader, optimizer, criterion, 
              scheduler=scheduler,epochs=EPOCHS, device="cuda", conf=conf)
    with open(DATA_DIR+f'history_{model.__name__}_{DATE_STR}.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    tqdm.write("\nLoading state_dict...\n")
    # model_file = find_sakt_model()
    model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_layer_1_head_8_embed_128_seq_150_auc_0.7604.pt'
    # model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_layer_2_head_8_embed_256_seq_150_auc_0.7577.pt'
    conf.SCALING = 3
    model, conf_dict = load_sakt_model(model_file)
    print(f"\nLoaded model at \n {model_file} \n with {conf_dict}")
    model.eval()
    valid_loss, valid_acc, valid_auc, results = valid_epoch(model, val_loader, criterion, conf=conf)
    print(f"\nValid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")

'''
Mock test in iter_env_sakt
'''
# %%