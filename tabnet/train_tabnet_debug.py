#%%
import sys
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

HOME =  "/home/scao/Documents/kaggle-riiid-test/"
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'

sys.path.append(HOME)
from utils import *


CONTENT_TYPE_ID = "content_type_id"
CONTENT_ID = "content_id"
TARGET = "answered_correctly"
USER_ID = "user_id"
PRIOR_QUESTION_TIME = 'prior_question_elapsed_time'
PRIOR_QUESTION_EXPLAIN = 'prior_question_had_explanation'
TASK_CONTAINER_ID = "task_container_id"
TIMESTAMP = "timestamp" 
ROW_ID = 'row_id'
NROWS_TRAIN = 1_000_000
NROWS_VALID = 10_000

DEBUG = False
VALIDATION = False
MAX_EPOCH = 50

BATCH_SIZE = 2**13
# %%
start = time()
print("Loading training...")
train_df = pd.read_parquet(DATA_DIR+'cv1_train.parquet')
valid_df = pd.read_parquet(DATA_DIR+'cv1_valid.parquet')
questions_df = pd.read_csv(DATA_DIR+'questions.csv')
print(f"Loaded training in {time()-start} seconds.")
# %%
# funcs for user stats with loop
def add_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(tqdm(df[['user_id','answered_correctly']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        answered_correctly_sum_u_dict[row[0]] += row[1]
        count_u_dict[row[0]] += 1
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def add_user_feats_without_update(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(df[['user_id']].values):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def update_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    for row in df[['user_id','answered_correctly','content_type_id']].values:
        if row[2] == 0:
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1
# %%

feature_cols = [ROW_ID, USER_ID, CONTENT_ID, CONTENT_TYPE_ID, 
                TARGET, PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN]
train_df = train_df[feature_cols]
valid_df = valid_df[feature_cols]

if DEBUG:
    train_df = train_df[:NROWS_TRAIN]
    valid_df = valid_df[:NROWS_VALID]

train_df = train_df.loc[train_df[CONTENT_TYPE_ID] == False].reset_index(drop=True)
valid_df = valid_df.loc[valid_df[CONTENT_TYPE_ID] == False].reset_index(drop=True)

# answered correctly average for each content
content_df = train_df[[CONTENT_ID, TARGET]].groupby([CONTENT_ID]).agg(['mean']).reset_index()
content_df.columns = [CONTENT_ID, 'answered_correctly_avg_c']
train_df = pd.merge(train_df, content_df, on=['content_id'], how="left")
valid_df = pd.merge(valid_df, content_df, on=['content_id'], how="left")

# user stats features with loops
answered_correctly_sum_u_dict = defaultdict(int)
count_u_dict = defaultdict(int)
train_df = add_user_feats(train_df, answered_correctly_sum_u_dict, count_u_dict)
valid_df = add_user_feats(valid_df, answered_correctly_sum_u_dict, count_u_dict)

# fill with mean value for prior_question_elapsed_time
# note that `train_df.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.

prior_question_elapsed_time_mean = train_df[PRIOR_QUESTION_TIME].dropna().values.mean()
# should we fill nan with mean??? 
# this may intuitively work as the user's first question might fall into the average correctness
train_df['prior_question_elapsed_time_mean'] = train_df[PRIOR_QUESTION_TIME].fillna(prior_question_elapsed_time_mean)
valid_df['prior_question_elapsed_time_mean'] = valid_df[PRIOR_QUESTION_TIME].fillna(prior_question_elapsed_time_mean)

# part
train_df = pd.merge(train_df, questions_df[['question_id', 'part']], 
                    left_on = 'content_id', right_on = 'question_id', how = 'left')
valid_df = pd.merge(valid_df, questions_df[['question_id', 'part']], 
                    left_on = 'content_id', right_on = 'question_id', how = 'left')

# changing dtype to avoid lightgbm error
train_df[PRIOR_QUESTION_EXPLAIN] = train_df[PRIOR_QUESTION_EXPLAIN].fillna(False).astype('int8')
valid_df[PRIOR_QUESTION_EXPLAIN] = valid_df[PRIOR_QUESTION_EXPLAIN].fillna(False).astype('int8')
# %% TABNET

FEATS = ['answered_correctly_avg_u', 'answered_correctly_sum_u', 'count_u',       
          'answered_correctly_avg_c', 'part', 'prior_question_had_explanation', 
          'prior_question_elapsed_time']
drop_cols = list(set(train_df.columns) - set(FEATS))
y_tr = train_df[TARGET]
y_va = valid_df[TARGET]
train_df.drop(drop_cols, axis=1, inplace=True)
valid_df.drop(drop_cols, axis=1, inplace=True)
gc.collect();


X_train, y_train = train_df[FEATS].values, y_tr.values
X_valid, y_valid = valid_df[FEATS].values, y_va.values

# TabNet does not allow Nan values
# A better fillna method might improve scores
X_train = np.nan_to_num(X_train, nan=-1)
X_valid = np.nan_to_num(X_valid, nan=-1)

del train_df, y_tr
gc.collect();
# %% the reference model
# model = TabNetClassifier(n_d=64, n_a=32, n_steps=2, gamma=1.2,
#                          n_independent=2, n_shared=2,
#                          lambda_sparse=0., seed=0,
#                          clip_value=1,
#                          mask_type='entmax',
#                          device_name='cuda',
#                          optimizer_fn=torch.optim.Adam,
#                          optimizer_params=dict(lr=2e-2),
#                          scheduler_params=dict(max_lr=0.05,
#                                                steps_per_epoch=int(X_train.shape[0] / BATCH_SIZE),
#                                                epochs=MAX_EPOCH,
#                                                final_div_factor=1e3,
#                                                is_batch_level=True),
#                          scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
#                          verbose=1,)

# model.fit(X_train=X_train, y_train=y_train,
#           eval_set=[(X_valid, y_valid)],
#           eval_name=["valid"],
#           eval_metric=["auc"],
#           batch_size=BATCH_SIZE,
#           virtual_batch_size=256,
#           max_epochs=MAX_EPOCH,
#           drop_last=True,
#           pin_memory=True
#          )

#%% modification 1

model = TabNetClassifier(n_d=64, n_a=32, n_steps=2, gamma=1.2,
                         n_independent=2, n_shared=2,
                         lambda_sparse=0, seed=1127,
                         clip_value=1.,
                         mask_type='entmax',
                         device_name='cuda',
                         optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=1e-3),
                         scheduler_params=dict(T_0=5,
                                               eta_min=1e-5),
                         scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                         verbose=1,)

model.fit(X_train=X_train, y_train=y_train,
          eval_set=[(X_valid, y_valid)],
          eval_name=["valid"],
          eval_metric=["auc"],
          batch_size=BATCH_SIZE,
          patience=10,
          virtual_batch_size=128,
          max_epochs=MAX_EPOCH,
          drop_last=True,
          pin_memory=True,
         )

# %%
