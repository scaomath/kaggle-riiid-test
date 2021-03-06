#%%
import sys
import os
from collections import Counter, deque
from time import time
from typing import List

import dask.dataframe as dd
import datatable as dt
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm

HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'

sys.path.append(HOME)
from utils import *
from iter_env import *
get_system()
from transformer import *
from transformer_optimizers import *
# %%
'''
To-do:
- Fix the same user_id prediction problem
- Check how the predicted probability relates to the original target in val set
- Add user when doing inference

Debugging notes:
- when updating past users, the padded needs to be updated;
The bugged version predicts the first few question answers to each user, in reality it should predicted the last few unmasked entry to the test data.
'''
TRAIN_DTYPES = {
    # 'row_id': np.uint32,
    'timestamp': np.int64,
    'user_id': np.int32,
    'content_id': np.int16,
    'content_type_id': np.int8,
    'task_container_id': np.int16,
    'user_answer': np.int8,
    'answered_correctly': np.int8,
    'prior_question_elapsed_time': np.float32,
    'prior_question_had_explanation': 'boolean'
}
TRAIN_COLS = list(TRAIN_DTYPES.keys())

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

MAX_SEQ = 150 # this parameter denotes how many last seen content_ids I am going to consider <aka the max_seq_len or the window size>.
NUM_HEADS = 8
NUM_EMBED = 512
NUM_HIDDEN = 128 # the FC size in the transformer block
NUM_LAYERS = 2 # number of transformer blocks

# TAIL_N = 50 # used for validation set per user_id
FILLNA_VAL = 100 # fillers for the values (a unique value)
TQDM_INT = 15 # tqdm update interval
PAD = 0
BATCH_SIZE = 512
VAL_BATCH_SIZE = 4096
TEST_BATCH_SIZE = 51200

NROWS_TRAIN = 5_000_000
NROWS_VALID = 200_000
NROWS_TEST = 250_000
SIMU_PUB_SIZE = 25_000
DEBUG_PUB_SIZE = 1000

EPOCHS = 40

DEBUG = False
TRAIN = False


# %% Preparing train and validation set
start = time()
if DEBUG: 
    print("\nLoading train from parquet...")
    train_df = pd.read_parquet(DATA_DIR+'cv3_valid.parquet')
    valid_df = train_df[:NROWS_VALID].copy()
else:
    print("\nLoading train from parquet...")
    train_df = pd.read_parquet(DATA_DIR+'cv3_train.parquet')
    valid_df = pd.read_parquet(DATA_DIR+'cv3_valid.parquet')

#%%
train_df = train_df[TRAIN_COLS]
valid_df = valid_df[TRAIN_COLS]
train_df = train_df[train_df['content_type_id'] == False].reset_index(drop = True)
valid_df = valid_df[valid_df['content_type_id'] == False].reset_index(drop = True)

print("valid:", valid_df.shape, "users:", valid_df['user_id'].nunique())
print("train:", train_df.shape, "users:", train_df['user_id'].nunique())

print(f"Loaded train in {time()-start} seconds\n\n")
df_questions = pd.read_csv(DATA_DIR+'questions.csv')
print(train_df.head(10))

#%%
train_df = preprocess(train_df, df_questions)
valid_df = preprocess(valid_df, df_questions)
d, user_id_to_idx_train = get_feats(train_df)
d_val, user_id_to_idx_val = get_feats(valid_df)
print(f"\nProcessed train and valid in {time()-start} seconds\n\n")
uid_train = list(user_id_to_idx_train.keys())
print(uid_train[0]) 
print(d[0]['user_id'])

#%% loading dataset
dataset_train = Riiid(d=d)
dataset_val = Riiid(d=d_val)
# print(dataset[0]) # sample dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
sample = next(iter(DataLoader(dataset=dataset_train, 
                batch_size=1, collate_fn=collate_fn))) # dummy check
train_loader = DataLoader(dataset=dataset_train, 
                          batch_size=BATCH_SIZE, 
                          collate_fn=collate_fn)

val_loader = DataLoader(dataset=dataset_val, 
                        batch_size=VAL_BATCH_SIZE, 
                        collate_fn=collate_fn, 
                        drop_last=True)

#%% createing the model, training and validation
model = TransformerModel(ninp=NUM_EMBED, 
                         nhead=NUM_HEADS, 
                         nhid=NUM_HIDDEN, 
                         nlayers=NUM_LAYERS, 
                         dropout=0.3)
model = model.to(device)
num_params = get_num_params(model)

print(model)
print(f"\n\n# heads  : {NUM_HEADS}")
print(f"# embed  : {NUM_EMBED}")
print(f"seq len  : {MAX_SEQ}")
print(f"# params : {num_params}")
# 
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
lr = 1e-3 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = AdamW(model.parameters(), lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=conf.PATIENCE-1, threshold=1e-4,verbose=1)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=LEARNING_RATE*1e-2,verbose=1)
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, t_total=EPOCHS)
#%%
losses = []
history = []
auc_max = 0
if TRAIN:
    print("\n\nTraining...:")
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        print(f"\n\n[Epoch {epoch}/{EPOCHS}]")
        print(f"Train: loss - {train_loss:.2f} acc - {train_acc:.4f} auc - {train_auc:.4f}")
        valid_loss, valid_acc, valid_auc = valid_epoch(model, val_loader, criterion)
        print(f"\nValid: loss - {valid_loss:.2f} acc - {valid_acc:.4f} auc - {valid_auc:.4f}")
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history.append({"epoch":epoch, "lr": lr, 
                        **{"train_auc": train_auc, "train_acc": train_acc}, 
                        **{"valid_auc": valid_auc, "valid_acc": valid_acc}})
        print(f"\nLearning rate : {lr} ")
        if valid_auc > auc_max:
            print(f"\n[Epoch {epoch}/{EPOCHS}] auc improved from {auc_max:.4f} to {valid_auc:.4f}") 
            print("saving model ...")
            auc_max = valid_auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 
                                    f"transformer_orig_auc_{auc_max:.4f}.pt"))
else:
    try:
        print("Loading state_dict...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n Using device: {device} \n')
        model_files = find_files(name='transformer',path=MODEL_DIR)
        model_file = model_files[0]
        # model_file = '/home/scao/Documents/kaggle-riiid-test/model/transformer_orig_auc_0.7182.pt'
        conf = dict(ninp=NUM_EMBED, 
                    nhead=NUM_HEADS, 
                    nhid=NUM_HIDDEN, 
                    nlayers=NUM_LAYERS, 
                    dropout=0.3)
        model = load_model(model_file, conf=conf)
        # model.load_state_dict(torch.load(MODEL_DIR+'transformer_orig_auc_0.7602.pt', map_location=device))
        print(f'\nLoaded {model_file}.\n\n')
        model.eval()
        model.to(device)
        # print(model)
    except:
        FileExistsError("No model found.")
# %%
# if not TRAIN and DEBUG:
#     for batch in train_loader:
#         content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
#         target_id = batch[1].to(device).long()

#         content_id = Variable(content_id.cuda())
#         part_id = Variable(part_id.cuda())
#         prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
#         mask = Variable(mask.cuda())
#         with torch.no_grad():
#             output = model(content_id, part_id, prior_question_elapsed_time, mask)
#         pred_probs = torch.softmax(output, dim=2)
#         # pred = (output_prob >= 0.50)
#         pred = torch.argmax(pred_probs, dim=2)
#         break

# %%
test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
# test_df = test_df[TEST_DTYPES.keys()]
test_df = test_df[TRAIN_DTYPES.keys()]
test_df = test_df[:NROWS_TEST]

# test_df = pd.read_csv(DATA_DIR+'example_test.csv', 
#                     # nrows=NROWS_TEST, 
#                     dtype=TEST_DTYPES, 
#                     usecols=TEST_DTYPES.keys())
test_df = test_df[test_df['content_type_id'] == 0]
test_df = preprocess(test_df, df_questions)
# d_test, _ = get_feats(test_df, max_seq=MAX_SEQ)
d_test = {}
user_id_to_idx_test = {}
grp = test_df.groupby("user_id", sort=False).tail(MAX_SEQ)
grp_user = grp.groupby("user_id", sort=False)
# grp_user = test_df.groupby("user_id", sort=False)
#%%
num_user_id_grp = len(grp_user)

for idx, row in grp_user.agg({
    "content_id":list, 
    "answered_correctly":list,
    "task_container_id":list, 
    "part_id":list, 
    "prior_question_elapsed_time":list
    }).reset_index().iterrows():
    
    # print('\n\n',idx, '\n', row)
    # print('\n\n', row["content_id"])
    
    # here we make a split whether a user has more than equal to 100 entries or less than that
    # if it's less than MAX_SEQ, then we need to PAD it using the PAD token defined as 0 by me in this cell block
    # also, padded will be True where we have done padding obviously, rest places it's False.
    if len(row["content_id"]) >= MAX_SEQ:
        d_test[idx] = {
            "user_id": row["user_id"],
            "content_id" : deque(row["content_id"], maxlen=MAX_SEQ),
            "answered_correctly" : deque(row["answered_correctly"] + padding, maxlen=MAX_SEQ),
            "task_container_id" : deque(row["task_container_id"], maxlen=MAX_SEQ),
            "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"], maxlen=MAX_SEQ),
            "part_id": deque(row["part_id"], maxlen=MAX_SEQ),
            "padded" : deque([False]*MAX_SEQ, maxlen=MAX_SEQ),
            "pred_mask": deque([True]*MAX_SEQ, maxlen=MAX_SEQ),
        }
    else:
        # we have to pad...
        num_padding = MAX_SEQ-len(row["content_id"])
        padding = [PAD] * num_padding
        d_test[idx] = {
            "user_id": row["user_id"],
            "content_id" : deque(row["content_id"] + padding, maxlen=MAX_SEQ),
            "answered_correctly" : deque(row["answered_correctly"] + padding, maxlen=MAX_SEQ),
            "task_container_id" : deque(row["task_container_id"] + padding, maxlen=MAX_SEQ),
            "prior_question_elapsed_time" : deque(row["prior_question_elapsed_time"] + padding, maxlen=MAX_SEQ),
            "part_id": deque(row["part_id"] + padding, maxlen=MAX_SEQ),
            "padded" : deque([False]*len(row["content_id"]) + [True]*num_padding, maxlen=MAX_SEQ),
            "pred_mask": deque([True]*len(row["content_id"]) + [False]*num_padding, maxlen = MAX_SEQ)
        }
    user_id_to_idx_test[row["user_id"]] = idx

#%%
# dataset_test = RiiidTest(d=d_test)

# test_loader = DataLoader(dataset=dataset_test, batch_size=VAL_BATCH_SIZE, 
#                                 collate_fn=collate_fn_test, shuffle=False, drop_last=False)

dataset_test = RiiidVal(d=d_test)
test_loader = DataLoader(dataset=dataset_test, batch_size=VAL_BATCH_SIZE, 
                                collate_fn=collate_fn_val, shuffle=False, drop_last=False)
test_sample = next(iter(test_loader))
print(test_sample)
#%% Normal validation for sanity check
output_all = []
label_all = []
for idx, batch in enumerate(test_loader):
    content_id, _, part_id, prior_question_elapsed_time, mask, labels, pred_mask = batch
    target_id = batch[1].to(device).long()

    content_id = Variable(content_id.cuda())
    part_id = Variable(part_id.cuda())
    prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
    mask = Variable(mask.cuda())

    with torch.no_grad():
        output = model(content_id, part_id, prior_question_elapsed_time, 
                       mask_padding=mask)
    pred_probs = torch.softmax(output[pred_mask], dim=1)
    # pred = (output_prob >= 0.50)
    # pred = torch.argmax(pred_probs, dim=1)
    output_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy())
    label_all.extend(labels[pred_mask].reshape(-1).data.cpu().numpy())
    if len(test_loader) <=5:
        print(idx, len(batch[1]), len(target_id[pred_mask].reshape(-1)))
        print(len(pred_probs[:,1].reshape(-1)), len(labels[pred_mask].reshape(-1)))
        print("\n")
# test_df['answered_correctly'] = output_all

# y_true = test_df[test_df['content_type_id'] == 0]['answered_correctly']
y_true = np.array(label_all)
y_pred = np.array(output_all)
print('Validation auc:', roc_auc_score(y_true, y_pred))


# %% Using iter_env
print("Loading test set....")
test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
test_df = test_df[:SIMU_PUB_SIZE].copy()
# test_df = test_df[NROWS_TEST:NROWS_TEST+DEBUG_PUB_SIZE].copy() # okay
# test_df = test_df[:DEBUG_PUB_SIZE].copy() # not good
print("Loaded test.")
print(f"Simulated iter_env with {test_df['user_id'].nunique()} users.")
#%%
iter_test = Iter_Valid(test_df, max_user=1000)
predicted = []
def set_predict(df):
    predicted.append(df)

len_test = len(test_df)
prev_test_df = None
feature_cols =  ['content_id', 
        'answered_correctly', 
        'task_container_id', 
        'prior_question_elapsed_time',
        'part_id',
        ]
mask_cols = ['padded',
            'pred_mask']

# def update_users(d, uid_to_idx, new_df):
#     '''
#     Update the users for train dict d
#     '''
#     new_df = preprocess(new_df, df_questions)
#     d_new, uid_to_idx_new = get_feats(new_df)

#     for uid_new, idx_new_ in uid_to_idx_new.items():

#         mask = [not s for s in d_prev[idx_new_]['padded']]
#         prev = []
#         for col in feature_cols:
#             prev.append(np.array(d_prev[idx_new_][col])[mask])
        
#         if uid_new in uid_to_idx.keys():
            
#             idx_new_to_old = uid_to_idx[uid_new]
#             train_user_mask = [not s for s in d[idx_new_to_old]['padded']]
#             for idx_feat, feat in enumerate(feature_cols):
#                 d[idx_new_to_old][feat] = np.append(np.array(d[idx_new_to_old][feat])[train_user_mask],
#                                                     prev[idx_feat])
#         else:
#             for idx_feat, feat in enumerate(feature_cols):
#                 d[idx_new_to_old][feat] = prev[idx_feat]
        
#         if len(d[idx_new_to_old]['content_id']) >= MAX_SEQ:
#             for feat in feature_cols:   
#                 d[idx_new_to_old][feat] = deque(d[idx_new_to_old][feat], maxlen=MAX_SEQ)
#         else:
#             num_padding = MAX_SEQ - len(d[idx_new_to_old]['content_id'])
#             for feat in feature_cols:   
#                 d[idx_new_to_old][feat] = deque(d[idx_new_to_old][feat] + [PAD]*num_padding, maxlen=MAX_SEQ)

#     return d

def collate_fn_val(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded, labels, pred_mask = zip(*batch)
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    labels = torch.Tensor(labels)
    pred_mask = torch.Tensor(pred_mask).bool()
    # remember the order
    return content_id, task_id, part_id, prior_question_elapsed_time, padded, labels, pred_mask

def collate_fn_test(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded, pred_mask = zip(*batch)
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    pred_mask = torch.Tensor(pred_mask).bool()
    # remember the order
    return content_id, task_id, part_id, prior_question_elapsed_time, padded, pred_mask

def update_users(d, d_new, uid_to_idx, uid_to_idx_new, test_flag=False):
    '''
    Add the user's features from d to d_new
    During inference:
    1. add user's feature from previous df to train df (old=prev test, new=train)
    2. after reading current test df, add user's from train df to test df (old=train, new=current test)

    '''
    feature_cols =  ['content_id', 
            'answered_correctly', 
            'task_container_id', 
            'prior_question_elapsed_time',
            'part_id',
            ]
    mask_cols = ['padded',
            'pred_mask']
    for uid_test, idx_test in uid_to_idx_new.items():
        if uid_test in uid_to_idx.keys():
            idx_train = uid_to_idx[uid_test]
            old_user_mask = [not s for s in d[idx_train]['padded']]

            old_user = []
            for col in feature_cols:
                old_user.append(np.array(d[idx_train][col])[old_user_mask])
            
            new_user_mask = [not s for s in d_new[idx_test]['padded']]
            len_user_pred = sum(new_user_mask)
            # print(len_user_pred)
            for idx_feat, feat in enumerate(feature_cols):
                new_user_update = np.append(old_user[idx_feat],
                                            np.array(d_new[idx_test][feat])[new_user_mask])
                len_user = len(new_user_update) # the length of the current user after update
                # print(len_user)
                if len_user >= MAX_SEQ:
                    d_new[idx_test][feat] = deque(new_user_update[-MAX_SEQ:], maxlen=MAX_SEQ)
                    
                else:
                    num_padding = MAX_SEQ - len_user
                    d_new[idx_test][feat] = deque(np.append(new_user_update, 
                    np.zeros(num_padding, dtype=int)), maxlen=MAX_SEQ)
                    
            if test_flag:
                assert MAX_SEQ >= len_user_pred
                if len_user >= MAX_SEQ:
                    d_new[idx_test]['padded'] = deque([False]*MAX_SEQ, maxlen=MAX_SEQ)
                    d_new[idx_test]['pred_mask'] = \
                        deque([False]*(MAX_SEQ-len_user_pred) + [True]*len_user_pred, maxlen=MAX_SEQ)
                else:       
                    num_padding = MAX_SEQ - len_user
                    
                    d_new[idx_test]['padded'] = deque([False]*len_user + [True]*num_padding, maxlen=MAX_SEQ)

                    d_new[idx_test]['pred_mask'] = \
                                deque([False]*(len_user-len_user_pred) + [True]*len_user_pred + [False]*num_padding, 
                                maxlen=MAX_SEQ)
    return d_new

#%%
with tqdm(total=len_test) as pbar:
    for idx, (current_test, current_prediction_df) in enumerate(iter_test):
        
        '''
        concised iter_env
        '''
        if prev_test_df is not None:
            '''Making use of answers to previous questions'''
            answers = eval(current_test["prior_group_answers_correct"].iloc[0])
            responses = eval(current_test["prior_group_responses"].iloc[0])
            prev_test_df['answered_correctly'] = answers
            prev_test_df['user_answer'] = responses

            prev_test_df = prev_test_df[prev_test_df['content_type_id'] == False]
            prev_test_df = preprocess(prev_test_df, df_questions)
            d_prev, user_id_to_idx_prev = get_feats(prev_test_df)
            d = update_users(d_prev, d, user_id_to_idx_prev, user_id_to_idx_train)

        # no labels
        # d_test, user_id_to_idx = get_feats_test(current_test)
        # dataset_test = RiiidTest(d=d_test)
        # test_loader = DataLoader(dataset=dataset_test, batch_size=VAL_BATCH_SIZE, 
        #                             collate_fn=collate_fn_test, shuffle=False, drop_last=False)

        prev_test_df = current_test.copy()

        '''Labels for verification'''
        current_test = preprocess(current_test, df_questions)
        d_test, user_id_to_idx_test = get_feats_val(current_test, max_seq=MAX_SEQ)
        d_test = update_users(d, d_test, user_id_to_idx_train, user_id_to_idx_test, test_flag=True)
        dataset_test = RiiidVal(d=d_test)
        test_loader = DataLoader(dataset=dataset_test, 
                                    batch_size=TEST_BATCH_SIZE, 
                                    collate_fn=collate_fn_val, shuffle=False, drop_last=False)

        # the problem with current feature gen is that 
        # using groupby user_id sorts the user_id and makes it different from the 
        # test_df's order

        output_all = []
        labels_all = []
        for _, batch in enumerate(test_loader):
            content_id, _, part_id, prior_question_elapsed_time, mask, labels, pred_mask = batch
            target_id = batch[1].to(device).long()

            content_id = Variable(content_id.cuda())
            part_id = Variable(part_id.cuda())
            prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
            mask = Variable(mask.cuda())

            with torch.no_grad():
                output = model(content_id, part_id, prior_question_elapsed_time, mask_padding= mask)

            pred_probs = torch.softmax(output[pred_mask], dim=1)
            output_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy())
            labels_all.extend(labels[~mask].reshape(-1).data.numpy())
        '''prediction code ends'''

        current_test['answered_correctly'] = output_all
        set_predict(current_test.loc[:,['row_id', 'answered_correctly']])
        pbar.update(len(current_test))

        # if DEBUG:
        #     tqdm.write(f"  Iteration {idx}")
        #     if idx == 2: break

#%%
y_true = test_df[test_df.content_type_id == 0]['answered_correctly']
y_pred = pd.concat(predicted)['answered_correctly']
print('\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))
# %%

'''
All below are debugging scripts
'''


# for idx in d_prev.keys():
#     print(d_prev[idx]['user_id'])

# for prev_user_id, idx_prev in user_id_to_idx_prev.items():
                
#     prev_content = d_prev[idx_prev]['content_id']
#     prev_ac = d_prev[idx_prev]['answered_correctly']
#     prev_task = d_prev[idx_prev]['task_container_id']
#     prev_time = d_prev[idx_prev]['prior_question_elapsed_time']
#     prev_part = d_prev[idx_prev]['part_id']
#     prev_padded = d_prev[idx_prev]['padded']
# # %%
# tmp1 = d_prev[idx_prev]['content_id']

# %% 
d_test, user_id_to_idx_test = get_feats(current_test, max_seq=MAX_SEQ)
d_new = d_test
uid_to_idx = user_id_to_idx_train
uid_to_idx_new = user_id_to_idx_test
'''
Add the user's features from d to d_new
During inference:
1. add user's feature from previous df to train df (old=prev test, new=train)
2. after reading current test df, add user's from train df to test df (old=train, new=current test)
'''
for idx, (uid_test, idx_test) in enumerate(uid_to_idx_new.items()):
    print(idx)
    
    if uid_test in uid_to_idx.keys():
        idx_train = uid_to_idx[uid_test]
        old_user_mask = [not s for s in d[idx_train]['padded']]

        old_user = []
        for col in feature_cols:
            old_user.append(np.array(d[idx_train][col])[old_user_mask])
        
        new_user_mask = [not s for s in d_new[idx_test]['padded']]
        for idx_feat, feat in enumerate(feature_cols):
            new_user_update = np.append(old_user[idx_feat],
                                        np.array(d_new[idx_test][feat])[new_user_mask])
            if len(new_user_update) >= MAX_SEQ:
                d_new[idx_test][feat] = deque(new_user_update, maxlen=MAX_SEQ)

            else:
                num_padding = MAX_SEQ - len(new_user_update)
                d_new[idx_test][feat] = deque(np.append(new_user_update, 
                np.zeros(num_padding, dtype=int)), maxlen=MAX_SEQ)

# %%
for idx,batch in enumerate(train_loader):
    content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
    content_id = Variable(content_id.cuda())
    part_id = Variable(part_id.cuda())
    prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
    mask = Variable(mask.cuda())
    labels = Variable(labels.cuda().long())
    optimizer.zero_grad()

    with torch.no_grad():
        output = model(content_id, part_id, prior_question_elapsed_time,
                    #    mask_padding=mask,
                       )
    break
# %%
pred_probs = torch.softmax(output[~mask], dim=1)
pred = torch.argmax(pred_probs, dim=1)
labels = labels[~mask]
print((pred == labels).sum().item()/len(pred))
# %% this is the old iter_env code

# with tqdm(total=len_test) as pbar:
#     for idx, (current_test, current_prediction_df) in enumerate(iter_test):
        

#         if prev_test_df is not None:
#             '''Making use of answers to previous questions'''
#             answers = eval(current_test["prior_group_answers_correct"].iloc[0])
#             responses = eval(current_test["prior_group_responses"].iloc[0])
#             prev_test_df['answered_correctly'] = answers
#             prev_test_df['user_answer'] = responses

#             prev_test_df = prev_test_df[prev_test_df['content_type_id'] == False]
#             prev_test_df = preprocess(prev_test_df, df_questions)
#             d_prev, user_id_to_idx_prev = get_feats(prev_test_df)



#             for prev_user_id, idx_prev in user_id_to_idx_prev.items():

#                 prev_mask = [not s for s in d_prev[idx_prev]['padded']]
#                 # prev_content = np.array(d_prev[idx_prev]['content_id'])[prev_mask]
#                 # prev_ac = np.array(d_prev[idx_prev]['answered_correctly'])[prev_mask]
#                 # prev_task = np.array(d_prev[idx_prev]['task_container_id'])[prev_mask]
#                 # prev_time = np.array(d_prev[idx_prev]['prior_question_elapsed_time'])[prev_mask]
#                 # prev_part = np.array(d_prev[idx_prev]['part_id'])[prev_mask]

#                 prev = []
#                 for col in feature_cols:
#                     prev.append(np.array(d_prev[idx_prev][col])[prev_mask])
                
#                 if prev_user_id in user_id_to_idx_train.keys():
                    
#                     idx_prev_user = user_id_to_idx_train[prev_user_id]
#                     train_user_mask = [not s for s in d[idx_prev_user]['padded']]
#                     # d[idx_prev_user]['content_id'] = np.append(
#                     #                 np.array(d[idx_prev_user]['content_id'])[train_user_mask],
#                     #                 prev_content)
#                     # d[idx_prev_user]['answered_correctly'] = np.append(
#                     #                 np.array(d[idx_prev_user]['answered_correctly'])[train_user_mask],
#                     #                 prev_ac)
#                     # d[idx_prev_user]['task_container_id'] = np.append(
#                     #                 np.array(d[idx_prev_user]['task_container_id'])[train_user_mask],
#                     #                 prev_task)
#                     # d[idx_prev_user]['prior_question_elapsed_time'] = np.append(
#                     #                 np.array(d[idx_prev_user]['prior_question_elapsed_time'])[train_user_mask],
#                     #                 prev_time)
#                     # d[idx_prev_user]['part_id'] = np.append(
#                     #                 np.array(d[idx_prev_user]['part_id'])[train_user_mask],
#                     #                 prev_part)
#                     for idx_feat, feat in enumerate(feature_cols):
#                         prev_user_update = np.append(np.array(d[idx_prev_user][feat])[train_user_mask],
#                                                            prev[idx_feat])

#                         if len(prev_user_update) >= MAX_SEQ:
#                             d[idx_prev_user][feat] = deque(prev_user_update, maxlen=MAX_SEQ)

#                         else:
#                             num_padding = MAX_SEQ - len(prev_user_update)
#                             d[idx_prev_user][feat] = deque(np.append(prev_user_update, 
#                             np.zeros(num_padding, dtype=int)), maxlen=MAX_SEQ)
                        
#         # no labels
#         # d_test, user_id_to_idx = get_feats_test(current_test)
#         # dataset_test = RiiidTest(d=d_test)
#         # test_loader = DataLoader(dataset=dataset_test, batch_size=VAL_BATCH_SIZE, 
#         #                             collate_fn=collate_fn_test, shuffle=False, drop_last=False)

#         prev_test_df = current_test.copy()

#         '''Labels for verification'''
#         current_test = preprocess(current_test, df_questions)
#         d_test, user_id_to_idx_test = get_feats(current_test, max_seq=MAX_SEQ)
#         d_test = update_users(d, d_test, user_id_to_idx_train, user_id_to_idx_test)
#         dataset_test = Riiid(d=d_test)
#         test_loader = DataLoader(dataset=dataset_test, 
#                                     batch_size=TEST_BATCH_SIZE, 
#                                     collate_fn=collate_fn, shuffle=False, drop_last=False)

#         # the problem with current feature gen is that 
#         # using groupby user_id sorts the user_id and makes it different from the 
#         # test_df's order

#         output_all = []
#         labels_all = []
#         for _, batch in enumerate(test_loader):
#             content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
#             target_id = batch[1].to(device).long()

#             content_id = Variable(content_id.cuda())
#             part_id = Variable(part_id.cuda())
#             prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
#             mask = Variable(mask.cuda())

#             with torch.no_grad():
#                 output = model(content_id, part_id, prior_question_elapsed_time, 
#                               mask_src=future_mask, mask_padding= mask)

#             pred_probs = torch.softmax(output[~mask], dim=1)
#             output_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy())
#             labels_all.extend(labels[~mask].reshape(-1).data.numpy())
#         '''prediction code ends'''

#         current_test['answered_correctly'] = output_all
#         set_predict(current_test.loc[:,['row_id', 'answered_correctly']])
#         pbar.update(len(current_test))

#         # if DEBUG:
#         #     tqdm.write(f"  Iteration {idx}")
#         #     if idx == 2: break

#%%
class RiiidVal(Dataset):
    '''
    d[idx]["padded"]: future mask for the sequence
    '''
    def __init__(self, d):
        super(RiiidVal, self).__init__()
        self.d = d
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        # you can return a dict of these as well etc etc...
        # remember the order
        return idx, self.d[idx]["content_id"], self.d[idx]["task_container_id"], \
    self.d[idx]["part_id"], self.d[idx]["prior_question_elapsed_time"], self.d[idx]["padded"], \
    self.d[idx]["answered_correctly"],  self.d[idx]["pred_mask"]


def update_users(d, d_new, uid_to_idx, uid_to_idx_new, test_flag=False):
    '''
    Add the user's features from d to d_new
    During inference:
    1. add user's feature from previous df to train df (old=prev test, new=train)
    2. after reading current test df, add user's from train df to test df (old=train, new=current test)

    '''
    feature_cols =  ['content_id', 
            'answered_correctly', 
            'task_container_id', 
            'prior_question_elapsed_time',
            'part_id',
            ]
    mask_cols = ['padded',
            'pred_mask']
    for uid_test, idx_test in uid_to_idx_new.items():
        if uid_test in uid_to_idx.keys():
            idx_train = uid_to_idx[uid_test]
            old_user_mask = [not s for s in d[idx_train]['padded']]

            old_user = []
            for col in feature_cols:
                old_user.append(np.array(d[idx_train][col])[old_user_mask])
            
            new_user_mask = [not s for s in d_new[idx_test]['padded']]
            len_user_pred = sum(new_user_mask)
            # print(len_user_pred)
            for idx_feat, feat in enumerate(feature_cols):
                new_user_update = np.append(old_user[idx_feat],
                                            np.array(d_new[idx_test][feat])[new_user_mask])
                len_user = len(new_user_update) # the length of the current user after update
                # print(len_user)
                if len_user >= MAX_SEQ:
                    d_new[idx_test][feat] = deque(new_user_update[-MAX_SEQ:], maxlen=MAX_SEQ)
                    
                else:
                    num_padding = MAX_SEQ - len_user
                    d_new[idx_test][feat] = deque(np.append(new_user_update, 
                    np.zeros(num_padding, dtype=int)), maxlen=MAX_SEQ)

            print(f'\nTotal len for user {uid_test}: {len_user}')
 
            if test_flag:
                assert MAX_SEQ >= len_user_pred
                if len_user >= MAX_SEQ:
                    d_new[idx_test]['padded'] = deque([False]*MAX_SEQ, maxlen=MAX_SEQ)
                    print("\nSeq more or equal than max_seq")
                    print(MAX_SEQ-len_user_pred,len_user_pred)
                    d_new[idx_test]['pred_mask'] = \
                        deque([False]*(MAX_SEQ-len_user_pred) + [True]*len_user_pred, maxlen=MAX_SEQ)
                else:       
                    num_padding = MAX_SEQ - len_user
                    print("\nSeq less than max_seq")
                    print(MAX_SEQ-len_user+len_user_pred,len_user_pred, num_padding)
                    d_new[idx_test]['padded'] = deque([False]*len_user + [True]*num_padding, maxlen=MAX_SEQ)
                    '''
                    Debugging notes:
                    the bugged version uses MAX_SEQ - len_user + len_user_pred as the 
                    first mask, which was wrong
                    '''
                    d_new[idx_test]['pred_mask'] = \
                                deque([False]*(len_user-len_user_pred) + [True]*len_user_pred + [False]*num_padding, 
                                maxlen=MAX_SEQ)
        else:
            for feat in feature_cols:
                assert MAX_SEQ == len(d_new[idx_test][feat])
    return d_new

def collate_fn_val(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded, labels, pred_mask = zip(*batch)
 
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    labels = torch.Tensor(labels)
    for i in range(len(pred_mask)):
        # print(len(pred_mask[i]))
        # print("\n")
        if len(pred_mask[i])==80 or len(pred_mask[i])==128:
            print(content_id[i])
            print(pred_mask[i])   
    pred_mask = torch.Tensor(pred_mask).bool()
    print((content_id).shape)
    print((labels).shape)
    print((pred_mask).shape)
    return content_id, task_id, part_id, prior_question_elapsed_time, padded, labels, pred_mask

def collate_fn(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded, labels = zip(*batch)
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    labels = torch.Tensor(labels)
    # remember the order
    return content_id, task_id, part_id, prior_question_elapsed_time, padded, labels


print("Loading test set....")
test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
# test_df = test_df[:SIMU_PUB_SIZE].copy()
# test_df = test_df[NROWS_TEST:NROWS_TEST+DEBUG_PUB_SIZE].copy() # okay
test_df = test_df[:DEBUG_PUB_SIZE].copy() # not good
print("Loaded test.")
print(f"Simulated iter_env with {test_df['user_id'].nunique()} users.")

iter_test = Iter_Valid(test_df, max_user=1000)
predicted = []
def set_predict(df):
    predicted.append(df)

for idx, (current_test, current_prediction_df) in enumerate(iter_test):
    break

current_test = preprocess(current_test, df_questions)
d_test, user_id_to_idx_test = get_feats_val(current_test, max_seq=MAX_SEQ)
d_test = update_users(d, d_test, user_id_to_idx_train, user_id_to_idx_test, test_flag=True)
dataset_test = RiiidVal(d=d_test)

test_loader = DataLoader(dataset=dataset_test, 
                            batch_size=TEST_BATCH_SIZE, 
                            collate_fn=collate_fn_val, 
                            shuffle=False, drop_last=False)

d_test_compare, _ = get_feats(current_test, max_seq=MAX_SEQ)
dataset_test_compare = Riiid(d=d_test_compare)
test_loader_compare = DataLoader(dataset=dataset_test_compare, 
                            batch_size=TEST_BATCH_SIZE, 
                            collate_fn=collate_fn, 
                            shuffle=False, drop_last=False)

# the problem with current feature gen is that 
# using groupby user_id sorts the user_id and makes it different from the 
# test_df's order
sample0 = dataset_test.__getitem__(0)
sample1 = dataset_test_compare.__getitem__(0)

# for i in range(1,len(sample0)):
#     print(len(sample0[i]))
# print(type(sample0[-1]))

# for i in range(1,len(sample1)):
#     print(len(sample1[i]))
# print(type(sample1[-1]))

for idx, batch in enumerate(test_loader_compare):
    if idx == len(test_loader_compare):
        print(1)

for idx, batch in enumerate(test_loader):
    if idx == len(test_loader):
        print(1)
# %%
