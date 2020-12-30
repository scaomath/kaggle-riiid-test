#%%
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

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(HOME) 
MODEL_DIR = HOME+'/model/'
DATA_DIR = HOME+'/data/'

from utils import *
from iter_env import *
# from .utils_lgb import *
get_system()
get_seed(1227)

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 10)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('max_colwidth', 20)
# %%
'''
Version notes:
1. baseline: CV 0.7765, iter_env CV 0.7503
2. removing 'part': CV 0.7759
'''


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

DEBUG = True # only using a fraction of the data

if DEBUG:
    NROWS_TEST = 25_000
    NROWS_TRAIN = 5_000_000
    NROWS_VAL = 1_000_000
else:
    NROWS_TEST = 250_000
    NROWS_TRAIN = 50_000_000
    NROWS_VAL = 2_000_000

# %%
# Funcion for user stats with loops
def add_features(df, 
                 answered_correctly_u_count, 
                 answered_correctly_u_sum, 
                 elapsed_time_u_sum, 
                 explanation_u_sum, 
                 timestamp_u, 
                 timestamp_u_incorrect, 
                 answered_correctly_q_count, 
                 answered_correctly_q_sum, 
                 elapsed_time_q_sum, 
                 explanation_q_sum, 
                 answered_correctly_uq, 
                 update = True):
    # -----------------------------------------------------------------------
    # Client features
    answered_correctly_u_avg = np.zeros(len(df), dtype = np.float32)

    elapsed_time_u_avg = np.zeros(len(df), dtype = np.float32)

    explanation_u_avg = np.zeros(len(df), dtype = np.float32)

    timestamp_u_recency_1 = np.zeros(len(df), dtype = np.float32)

    timestamp_u_recency_2 = np.zeros(len(df), dtype = np.float32)

    timestamp_u_recency_3 = np.zeros(len(df), dtype = np.float32)

    timestamp_u_incorrect_recency = np.zeros(len(df), dtype = np.float32)
    # -----------------------------------------------------------------------
    # Question features
    answered_correctly_q_avg = np.zeros(len(df), dtype = np.float32)
    elapsed_time_q_avg = np.zeros(len(df), dtype = np.float32)
    explanation_q_avg = np.zeros(len(df), dtype = np.float32)
    
    # -----------------------------------------------------------------------
    # User Question
    answered_correctly_uq_count = np.zeros(len(df), dtype = np.int32)
    
    # -----------------------------------------------------------------------
    total_len = len(df)
    
    # with tqdm(total=total_len) as pbar:
    for num, row in enumerate(df[['user_id',
                                    'answered_correctly', 
                                    'content_id', 
                                    'prior_question_elapsed_time', 
                                    'prior_question_had_explanation',
                                    'timestamp']].values):

        # Client features assignation
        # ------------------------------------------------------------------
        if answered_correctly_u_count[row[0]] != 0:
            answered_correctly_u_avg[num] = \
            answered_correctly_u_sum[row[0]] / answered_correctly_u_count[row[0]]

            elapsed_time_u_avg[num] = \
            elapsed_time_u_sum[row[0]] / answered_correctly_u_count[row[0]]

            explanation_u_avg[num] = \
            explanation_u_sum[row[0]] / answered_correctly_u_count[row[0]]

        else:
            answered_correctly_u_avg[num] = np.nan

            elapsed_time_u_avg[num] = np.nan

            explanation_u_avg[num] = np.nan

        if len(timestamp_u[row[0]]) == 0:
            timestamp_u_recency_1[num] = np.nan
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan

        elif len(timestamp_u[row[0]]) == 1:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan

        elif len(timestamp_u[row[0]]) == 2:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_3[num] = np.nan

        elif len(timestamp_u[row[0]]) == 3:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][0]

        if len(timestamp_u_incorrect[row[0]]) == 0:
            timestamp_u_incorrect_recency[num] = np.nan
        else:
            timestamp_u_incorrect_recency[num] = \
            row[5] - timestamp_u_incorrect[row[0]][0]

        # ------------------------------------------------------------------
        # Question features assignation
        if answered_correctly_q_count[row[2]] != 0:
            answered_correctly_q_avg[num] = answered_correctly_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            elapsed_time_q_avg[num] = elapsed_time_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            explanation_q_avg[num] = explanation_q_sum[row[2]] / answered_correctly_q_count[row[2]]
        else:
            answered_correctly_q_avg[num] = np.nan
            elapsed_time_q_avg[num] = np.nan
            explanation_q_avg[num] = np.nan
        # ------------------------------------------------------------------
        # Client Question assignation
        answered_correctly_uq_count[num] = answered_correctly_uq[row[0]][row[2]]
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Client features updates
        answered_correctly_u_count[row[0]] += 1
        elapsed_time_u_sum[row[0]] += row[3]
        explanation_u_sum[row[0]] += int(row[4])

        if len(timestamp_u[row[0]]) == 3:
            timestamp_u[row[0]].pop(0)
            timestamp_u[row[0]].append(row[5])
        else:
            timestamp_u[row[0]].append(row[5])

        # ------------------------------------------------------------------
        # Question features updates
        answered_correctly_q_count[row[2]] += 1
        elapsed_time_q_sum[row[2]] += row[3]
        explanation_q_sum[row[2]] += int(row[4])
        # ------------------------------------------------------------------
        # Client Question updates
        answered_correctly_uq[row[0]][row[2]] += 1

        # ------------------------------------------------------------------
        # Flag for training and inference
        if update:
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[5])
                else:
                    timestamp_u_incorrect[row[0]].append(row[5])

            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_sum[row[2]] += row[1]
                # ------------------------------------------------------------------
            # if num % 100 == 0:
            #     pbar.update(100)
            
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
    return df
# %%
def update_features(df, 
                    answered_correctly_u_sum, 
                    answered_correctly_q_sum, 
                    timestamp_u_incorrect):
    for row in df[['user_id', 
                   'answered_correctly', 
                   'content_id', 
                   'content_type_id', 
                   'timestamp']].values:
        if row[3] == 0:
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[4])
                else:
                    timestamp_u_incorrect[row[0]].append(row[4])
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_sum[row[2]] += row[1]
            # ------------------------------------------------------------------
            
#%%
train_parquet = DATA_DIR+'cv5_train.parquet'
valid_parquet = DATA_DIR+'cv5_valid.parquet'
question_file = DATA_DIR+'questions.csv'

# Read data
features = ['timestamp', 
            'user_id', 
            'answered_correctly',
            'content_id', 
            'content_type_id', 
            'prior_question_elapsed_time', 
            'prior_question_had_explanation']
                
with timer("Loading train and valid"):
    train = pd.read_parquet(train_parquet, 
    columns=list(TRAIN_DTYPES.keys()))
    train = train.astype(TRAIN_DTYPES)
    train = train.iloc[:NROWS_TRAIN]

    valid = pd.read_parquet(valid_parquet, 
    columns=list(TRAIN_DTYPES.keys()))
    valid = valid.astype(TRAIN_DTYPES)
    valid = valid.iloc[:NROWS_VAL]


# Filter by content_type_id to discard lectures
train = train.loc[train.content_type_id == False].reset_index(drop = True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop = True)

# Changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = \
train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = \
valid.prior_question_had_explanation.fillna(False).astype('int8')

# Fill prior question elapsed time with the mean
prior_question_elapsed_time_mean = \
train['prior_question_elapsed_time'].dropna().mean()
train['prior_question_elapsed_time']\
.fillna(prior_question_elapsed_time_mean, inplace = True)
valid['prior_question_elapsed_time']\
.fillna(prior_question_elapsed_time_mean, inplace = True)

# Merge with question dataframe
questions_df = pd.read_csv(question_file)
questions_df['part'] = questions_df['part'].astype(np.int32)
questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)

train = pd.merge(train, questions_df[['question_id', 'part']], 
                    left_on = 'content_id', 
                    right_on = 'question_id', 
                    how = 'left')
valid = pd.merge(valid, questions_df[['question_id', 'part']], 
                    left_on = 'content_id', 
                    right_on = 'question_id', 
                    how = 'left')

# Client dictionaries
answered_correctly_u_count = defaultdict(int)
answered_correctly_u_sum = defaultdict(int)
elapsed_time_u_sum = defaultdict(int)
explanation_u_sum = defaultdict(int)
timestamp_u = defaultdict(list)
timestamp_u_incorrect = defaultdict(list)

# Question dictionaries
answered_correctly_q_count = defaultdict(int)
answered_correctly_q_sum = defaultdict(int)
elapsed_time_q_sum = defaultdict(int)
explanation_q_sum = defaultdict(int)

# Client Question dictionary
answered_correctly_uq = defaultdict(lambda: defaultdict(int))

print('\nUser feature calculation started...\n')
train = add_features(train, 
                    answered_correctly_u_count, 
                    answered_correctly_u_sum, 
                    elapsed_time_u_sum, 
                    explanation_u_sum, 
                    timestamp_u, 
                    timestamp_u_incorrect, 
                    answered_correctly_q_count, 
                    answered_correctly_q_sum, 
                    elapsed_time_q_sum, 
                    explanation_q_sum, 
                    answered_correctly_uq)

valid = add_features(valid, 
                    answered_correctly_u_count, 
                    answered_correctly_u_sum, 
                    elapsed_time_u_sum, 
                    explanation_u_sum, 
                    timestamp_u, 
                    timestamp_u_incorrect, 
                    answered_correctly_q_count, 
                    answered_correctly_q_sum, 
                    elapsed_time_q_sum, 
                    explanation_q_sum, 
                    answered_correctly_uq)
gc.collect()
print('\nUser feature calculation completed...\n')

features_dicts = {
    'answered_correctly_u_count': answered_correctly_u_count,
    'answered_correctly_u_sum': answered_correctly_u_sum,
    'elapsed_time_u_sum': elapsed_time_u_sum,
    'explanation_u_sum': explanation_u_sum,
    'answered_correctly_q_count': answered_correctly_q_count,
    'answered_correctly_q_sum': answered_correctly_q_sum,
    'elapsed_time_q_sum': elapsed_time_q_sum,
    'explanation_q_sum': explanation_q_sum,
    'answered_correctly_uq': answered_correctly_uq,
    'timestamp_u': timestamp_u,
    'timestamp_u_incorrect': timestamp_u_incorrect
}
    
# %% training and evaluation
    
target = 'answered_correctly'
# Features to train and predict
features = ['prior_question_elapsed_time', 
            'prior_question_had_explanation', 
            # 'part', 
            'answered_correctly_u_avg', 
            'elapsed_time_u_avg', 
            'explanation_u_avg',
            'answered_correctly_q_avg', 
            'elapsed_time_q_avg', 
            'explanation_q_avg', 
            'answered_correctly_uq_count', 
            'timestamp_u_recency_1',
            'timestamp_u_recency_2', 
            'timestamp_u_recency_3', 
            'timestamp_u_incorrect_recency']


print(f'Traning with {train.shape[0]} rows and {len(features)} features')    
drop_cols = list(set(train.columns).difference(features))
y_train = train[target].values.astype('int8')
y_val = valid[target].values.astype('int8')
# Drop unnecessary columns
train.drop(drop_cols, axis = 1, inplace = True)
valid.drop(drop_cols, axis = 1, inplace = True)
gc.collect()

lgb_train = lgb.Dataset(train[features].astype(np.float32), label=y_train)
lgb_valid = lgb.Dataset(valid[features].astype(np.float32), label=y_val)

del train, y_train
gc.collect()

params = {'objective': 'binary', 
            'seed': SEED,
            'metric': 'auc',
            'max_bin': 450,
            'min_child_weight': 0.05,
            'min_data_in_leaf': 512,
            'num_leaves': 200,
            'feature_fraction': 0.75,
            'learning_rate': 0.05,
            'bagging_freq': 5,
            'bagging_fraction': 0.7
            }

model = lgb.train(
    params = params,
    train_set = lgb_train,
    num_boost_round = 5000,
    valid_sets = [lgb_train, lgb_valid],
    early_stopping_rounds = 50,
    verbose_eval = 50
)
val_auc = roc_auc_score(y_val, model.predict(valid[features]))
print(f'AUC score for the validation data is: {val_auc:.4f}')

model.save_model(MODEL_DIR+f'lgb_loop_fold_0_auc_{val_auc:.4f}.txt') 

feature_importance = model.feature_importance()
feature_importance = pd.DataFrame({'Features': features, 
                                    'Importance': feature_importance})\
                        .sort_values('Importance', ascending = False)

fig = plt.figure(figsize = (8, 10))
fig.suptitle('Feature Importance', fontsize = 20)
plt.tick_params(axis = 'x', labelsize = 12)
plt.tick_params(axis = 'y', labelsize = 12)
plt.xlabel('Importance', fontsize = 15)
plt.ylabel('Features', fontsize = 15)
sns.barplot(x = feature_importance['Importance'], 
            y = feature_importance['Features'], orient = 'h')
plt.show()

  

#%% inference
# Get feature dict
answered_correctly_u_count = features_dicts['answered_correctly_u_count']
answered_correctly_u_sum = features_dicts['answered_correctly_u_sum']
elapsed_time_u_sum = features_dicts['elapsed_time_u_sum']
explanation_u_sum = features_dicts['explanation_u_sum']
answered_correctly_q_count = features_dicts['answered_correctly_q_count']
answered_correctly_q_sum = features_dicts['answered_correctly_q_sum']
elapsed_time_q_sum = features_dicts['elapsed_time_q_sum']
explanation_q_sum = features_dicts['explanation_q_sum']
answered_correctly_uq = features_dicts['answered_correctly_uq']
timestamp_u = features_dicts['timestamp_u']
timestamp_u_incorrect = features_dicts['timestamp_u_incorrect']

#%%

valid_df = pd.read_parquet(DATA_DIR+'cv5_valid.parquet')
valid_df = valid_df[:NROWS_TEST]
iter_test = Iter_Valid(valid_df, max_user=1000)

predicted = []
def set_predict(df):
    predicted.append(df)

previous_test_df = None
len_test= len(valid_df[valid_df['content_type_id'] == 0])

with tqdm(total=len_test) as pbar:
    for (test_df, sample_prediction_df) in iter_test:
        if previous_test_df is not None:
            previous_test_df[target] = eval(test_df["prior_group_answers_correct"].iloc[0])
            update_features(previous_test_df, answered_correctly_u_sum, answered_correctly_q_sum, timestamp_u_incorrect)
        previous_test_df = test_df.copy()
        
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop = True)
        
        test_df['prior_question_had_explanation'] = \
        test_df.prior_question_had_explanation.fillna(False).astype('int8')
        
        test_df['prior_question_elapsed_time'].\
        fillna(prior_question_elapsed_time_mean, inplace = True)
        test_df = pd.merge(test_df, questions_df[['question_id', 'part']], 
                        left_on = 'content_id', 
                        right_on = 'question_id', 
                        how = 'left')
        test_df[target] = 0.66
        
        test_df = add_features(test_df, 
                            answered_correctly_u_count, 
                            answered_correctly_u_sum, 
                            elapsed_time_u_sum, 
                            explanation_u_sum, 
                            timestamp_u, 
                            timestamp_u_incorrect, 
                            answered_correctly_q_count, 
                            answered_correctly_q_sum, 
                            elapsed_time_q_sum, 
                            explanation_q_sum, 
                            answered_correctly_uq, 
                            update = False)
        
        test_df[target] = model.predict(test_df[features])
        set_predict(test_df.loc[:,['row_id', target]])
        pbar.update(len(test_df))

print('\nJob Done')

#%%
y_true = valid_df[valid_df.content_type_id == 0][target]
y_pred = pd.concat(predicted)[target]
print('\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))
# %%
