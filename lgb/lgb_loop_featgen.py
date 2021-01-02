#%% feature generation for inference
import os
from lightgbm.engine import train
import numpy as np
import pandas as pd
from collections import defaultdict
import datatable as dt
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from jupyterthemes import jtplot
jtplot.style(theme='onedork', context='notebook', ticks=True, grid=False)

import random
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc
import pickle
import zipfile

# HOME = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
# MODEL_DIR = HOME+'/model/'
# DATA_DIR = HOME+'/data/'

HOME = os.path.abspath(os.path.join('.', os.pardir))
print(HOME, '\n\n')
MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 

from utils import *
from iter_env import *
get_system()
get_seed(1227)
from utils_lgb import *


pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 10)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('max_colwidth', 20)
# %%
CONTENT_TYPE_ID = "content_type_id"
CONTENT_ID = "content_id"
target = "answered_correctly"
target = target
USER_ID = "user_id"
PRIOR_QUESTION_TIME = 'prior_question_elapsed_time'
PRIOR_QUESTION_EXPLAIN = 'prior_question_had_explanation'
TASK_CONTAINER_ID = "task_container_id"
TIMESTAMP = "timestamp" 
ROW_ID = 'row_id'

TRAIN_DTYPES = {
    TIMESTAMP: 'int64',
    USER_ID: 'int32', 
    CONTENT_ID: 'int16', 
    CONTENT_TYPE_ID:'int8', 
    TASK_CONTAINER_ID: 'int16',
    target: 'int8', 
    PRIOR_QUESTION_TIME: 'float32', 
    PRIOR_QUESTION_EXPLAIN: 'bool'
}

DEBUG = False # only using a fraction of the data

if DEBUG:
    NROWS_TEST = 25_000
    NROWS_TRAIN = 5_000_000
    NROWS_VAL = 1_000_000
else:
    NROWS_TEST = 25_000
    NROWS_TRAIN = 100_000_000
    NROWS_VAL = 1_000_000
# %%
train_parquet = DATA_DIR+'cv2_train.parquet'
features = ['timestamp', 
            'user_id', 
            'answered_correctly',
            'content_id', 
            'content_type_id', 
            'prior_question_elapsed_time', 
            'prior_question_had_explanation']
                
with timer("Loading train"):
    train = pd.read_parquet(train_parquet, 
                            columns=list(TRAIN_DTYPES.keys())).astype(TRAIN_DTYPES)
    train = train[:NROWS_TRAIN]

train = train.loc[train.content_type_id == False].reset_index(drop = True)

# Changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')

# Fill prior question elapsed time with the mean
prior_question_elapsed_time_mean = train['prior_question_elapsed_time'].dropna().mean()
train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, 
                                            inplace = True)
gc.collect()
# %%
# Funcion for user stats with loops
def get_features_row(row):
    
    '''
    after re-assignment
    row[0]: 'user_id',
    row[1]: 'answered_correctly', 
    row[2]: 'content_id', 
    row[3]: 'prior_question_elapsed_time', 
    row[4]: 'prior_question_had_explanation',
    row[5]: 'timestamp'
    '''
   
    num = row[0] # index
    row = row[1:]
    user_id = row[0]

    # Client features assignation
    # ------------------------------------------------------------------
    if answered_correctly_u_count[user_id] != 0:
        answered_correctly_u_avg[num] = \
        answered_correctly_u_sum[user_id] / answered_correctly_u_count[user_id]

        elapsed_time_u_avg[num] = \
        elapsed_time_u_sum[user_id] / answered_correctly_u_count[user_id]

        explanation_u_avg[num] = \
        explanation_u_sum[user_id] / answered_correctly_u_count[user_id]

    else:
        answered_correctly_u_avg[num] = np.nan

        elapsed_time_u_avg[num] = np.nan

        explanation_u_avg[num] = np.nan

    if len(timestamp_u[user_id]) == 0:
        timestamp_u_recency_1[num] = np.nan
        timestamp_u_recency_2[num] = np.nan
        timestamp_u_recency_3[num] = np.nan

    elif len(timestamp_u[user_id]) == 1:
        timestamp_u_recency_1[num] = row[5] - timestamp_u[user_id][0]
        timestamp_u_recency_2[num] = np.nan
        timestamp_u_recency_3[num] = np.nan

    elif len(timestamp_u[user_id]) == 2:
        timestamp_u_recency_1[num] = row[5] - timestamp_u[user_id][1]
        timestamp_u_recency_2[num] = row[5] - timestamp_u[user_id][0]
        timestamp_u_recency_3[num] = np.nan

    elif len(timestamp_u[user_id]) == 3:
        timestamp_u_recency_1[num] = row[5] - timestamp_u[user_id][2]
        timestamp_u_recency_2[num] = row[5] - timestamp_u[user_id][1]
        timestamp_u_recency_3[num] = row[5] - timestamp_u[user_id][0]

    if len(timestamp_u_incorrect[user_id]) == 0:
        timestamp_u_incorrect_recency[num] = np.nan
    else:
        timestamp_u_incorrect_recency[num] = \
        row[5] - timestamp_u_incorrect[user_id][0]

    # ------------------------------------------------------------------
    # Question features assignation
    if answered_correctly_q_count[row[2]] != 0:
        answered_correctly_q_avg[num] = \
        answered_correctly_q_sum[row[2]] / answered_correctly_q_count[row[2]]
        elapsed_time_q_avg[num] = elapsed_time_q_sum[row[2]] / answered_correctly_q_count[row[2]]
        explanation_q_avg[num] = explanation_q_sum[row[2]] / answered_correctly_q_count[row[2]]
    else:
        answered_correctly_q_avg[num] = np.nan
        elapsed_time_q_avg[num] = np.nan
        explanation_q_avg[num] = np.nan
    # ------------------------------------------------------------------
    # Client Question assignation
    answered_correctly_uq_count[num] = answered_correctly_uq[user_id][row[2]]
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Client features updates
    answered_correctly_u_count[user_id] += 1
    elapsed_time_u_sum[user_id] += row[3]
    explanation_u_sum[user_id] += int(row[4])

    if len(timestamp_u[user_id]) == 3:
        timestamp_u[user_id].pop(0)
        timestamp_u[user_id].append(row[5])
    else:
        timestamp_u[user_id].append(row[5])

    # ------------------------------------------------------------------
    # Question features updates
    answered_correctly_q_count[row[2]] += 1
    elapsed_time_q_sum[row[2]] += row[3]
    explanation_q_sum[row[2]] += int(row[4])
    # ------------------------------------------------------------------
    # Client Question updates
    answered_correctly_uq[user_id][row[2]] += 1

    # ------------------------------------------------------------------
    # Flag for training and inference
    # ------------------------------------------------------------------
    # Client features updates
    answered_correctly_u_sum[user_id] += row[1]
    if row[1] == 0:
        if len(timestamp_u_incorrect[user_id]) == 1:
            timestamp_u_incorrect[user_id].pop(0)
            timestamp_u_incorrect[user_id].append(row[5])
        else:
            timestamp_u_incorrect[user_id].append(row[5])

    # ------------------------------------------------------------------
    # Question features updates
    answered_correctly_q_sum[row[2]] += row[1]
    # ------------------------------------------------------------------



def add_features_df(df):
    user_df = pd.DataFrame({'answered_correctly_u_avg': answered_correctly_u_avg, 
                        'elapsed_time_u_avg': elapsed_time_u_avg, 
                        'explanation_u_avg': explanation_u_avg, 
                        'answered_correctly_q_avg': answered_correctly_q_avg, 
                        'elapsed_time_q_avg': elapsed_time_q_avg, 
                        'explanation_q_avg': explanation_q_avg, 
                        'answered_correctly_uq_count': answered_correctly_uq_count, 
                        'timestamp_u_recency_1': timestamp_u_recency_1, 
                        'timestamp_u_recency_2': timestamp_u_recency_2,
                        'timestamp_u_recency_3': timestamp_u_recency_3, 
                        'timestamp_u_incorrect_recency': timestamp_u_incorrect_recency})
    
    df = pd.concat([df, user_df], axis = 1)
# %%
train_len = len(train)

# -----------------------------------------------------------------------
# Client features
answered_correctly_u_avg = np.zeros(train_len, dtype = np.float32)
elapsed_time_u_avg = np.zeros(train_len, dtype = np.float32)
explanation_u_avg = np.zeros(train_len, dtype = np.float32)
timestamp_u_recency_1 = np.zeros(train_len, dtype = np.float32)
timestamp_u_recency_2 = np.zeros(train_len, dtype = np.float32)
timestamp_u_recency_3 = np.zeros(train_len, dtype = np.float32)
timestamp_u_incorrect_recency = np.zeros(train_len, dtype = np.float32)
# -----------------------------------------------------------------------
# Question features
answered_correctly_q_avg = np.zeros(train_len, dtype = np.float32)
elapsed_time_q_avg = np.zeros(train_len, dtype = np.float32)
explanation_q_avg = np.zeros(train_len, dtype = np.float32)

# -----------------------------------------------------------------------
# User Question
answered_correctly_uq_count = np.zeros(train_len, dtype = np.int32)

# Client dictionaries, global var to be updated
answered_correctly_u_count = defaultdict(int)
answered_correctly_u_sum = defaultdict(int)
elapsed_time_u_sum = defaultdict(int)
explanation_u_sum = defaultdict(int)
timestamp_u = defaultdict(list)
timestamp_u_incorrect = defaultdict(list)

# Question dictionaries, global var to be updated
answered_correctly_q_count = defaultdict(int)
answered_correctly_q_sum = defaultdict(int)
elapsed_time_q_sum = defaultdict(int)
explanation_q_sum = defaultdict(int)

# Client Question dictionary, if the user has not answer a questions, then the value is a defaultdict(int)
answered_correctly_uq = defaultdict(lambda: defaultdict(int))
# %%
iters = train[['user_id',
          'answered_correctly', 
          'content_id', 
          'prior_question_elapsed_time', 
          'prior_question_had_explanation',
          'timestamp']].itertuples()

with timer("User feature calculation"):
    for _row in tqdm(iters, total=train_len):
        get_features_row(_row)
gc.collect()

add_features_df(train)

#%% export train with features



# %%
with open(DATA_DIR+'answered_correctly_u_count.pickle', 'wb') as f:
    pickle.dump(answered_correctly_u_count, f, protocol=pickle.HIGHEST_PROTOCOL)   

with open(DATA_DIR+'answered_correctly_u_sum.pickle', 'wb') as f:
    pickle.dump(answered_correctly_u_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(DATA_DIR+'elapsed_time_u_sum.pickle', 'wb') as f:
    pickle.dump(elapsed_time_u_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(DATA_DIR+'explanation_u_sum.pickle', 'wb') as f:
    pickle.dump(explanation_u_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(DATA_DIR+'answered_correctly_q_count.pickle', 'wb') as f:
    pickle.dump(answered_correctly_q_count, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(DATA_DIR+'answered_correctly_q_sum.pickle', 'wb') as f:
    pickle.dump(answered_correctly_q_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(DATA_DIR+'elapsed_time_q_sum.pickle', 'wb') as f:
    pickle.dump(elapsed_time_q_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(DATA_DIR+'explanation_q_sum.pickle', 'wb') as f:
    pickle.dump(explanation_q_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(DATA_DIR+'timestamp_u.pickle', 'wb') as f:
    pickle.dump(timestamp_u, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(DATA_DIR+'timestamp_u_incorrect.pickle', 'wb') as f:
    pickle.dump(timestamp_u_incorrect, f, protocol=pickle.HIGHEST_PROTOCOL)

#%%
answered_correctly_uq_dict = defaultdict(int)
for num, row in enumerate(train[['user_id']].values):
    answered_correctly_uq_dict[row[0]] = answered_correctly_uq[row[0]]
    
with open(DATA_DIR+'answered_correctly_uq_dict.pickle', 'wb') as f:
    pickle.dump(answered_correctly_uq_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%

with open(DATA_DIR+'answered_correctly_uq_dict.pickle', 'rb') as f:
    answered_correctly_uq_dict = pickle.load(f)
with timer("Loading user-question dict"):
    answered_correctly_uq = defaultdict(lambda: defaultdict(int))
    for key in answered_correctly_uq_dict.keys():
        answered_correctly_uq[key] = answered_correctly_uq_dict[key]
# %%
