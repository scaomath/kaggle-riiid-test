import os
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import pickle

HOME = os.path.abspath(os.path.join('.', os.pardir))
print(HOME, '\n\n')
MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *

PRIOR_Q_ELAPSED_TIME_MEAN = 13005.081

CONTENT_TYPE_ID = "content_type_id"
CONTENT_ID = "content_id"
TARGET = "answered_correctly"
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
    TARGET: 'int8', 
    PRIOR_QUESTION_TIME: 'float32', 
    PRIOR_QUESTION_EXPLAIN: 'bool'
}


questions_df = pd.read_csv(os.path.join(DATA_DIR, 'questions.csv'))
questions_df['part'] = questions_df['part'].astype(np.int32)
questions_df['bundle_id'] = questions_df['bundle_id'].astype(np.int32)

def preprocess_lgb(df):
    '''
    preprocess for loop features generation
    '''

    # Filter by content_type_id to discard lectures
    df = df.loc[df.content_type_id == False].reset_index(drop = True)

    # Changing dtype to avoid lightgbm error
    df['prior_question_had_explanation'] = df.prior_question_had_explanation.fillna(False).astype('int8')

    # Fill prior question elapsed time with the mean
    df['prior_question_elapsed_time'].fillna(PRIOR_Q_ELAPSED_TIME_MEAN, inplace = True)

    # Merge with question dataframe
    df = pd.merge(df, questions_df[['question_id', 'part', 'tags']], 
                    left_on = 'content_id', 
                    right_on = 'question_id', 
                    how = 'left')
    return df

def add_features(row):
    
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

if __name__ == "__main__":
    with timer("Loading and processing train"):
        train_parquet = DATA_DIR+'/cv2_train.parquet'
        train = pd.read_parquet(train_parquet,  columns=list(TRAIN_DTYPES.keys())).astype(TRAIN_DTYPES)
        train = preprocess_lgb(train)
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

    # -----------------------------------------------------------------------
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
    iters = train[['user_id',
            'answered_correctly', 
            'content_id', 
            'prior_question_elapsed_time', 
            'prior_question_had_explanation',
            'timestamp']].itertuples()

    with timer("User feature calculation"):
        for _row in tqdm(iters, total=train_len):
            add_features(_row)


    with open(DATA_DIR+'/answered_correctly_u_count.pickle', 'wb') as f:
        pickle.dump(answered_correctly_u_count, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR+'/answered_correctly_u_sum.pickle', 'wb') as f:
        pickle.dump(answered_correctly_u_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR+'/elapsed_time_u_sum.pickle', 'wb') as f:
        pickle.dump(elapsed_time_u_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR+'/explanation_u_sum.pickle', 'wb') as f:
        pickle.dump(explanation_u_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR+'/answered_correctly_q_count.pickle', 'wb') as f:
        pickle.dump(answered_correctly_q_count, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR+'/answered_correctly_q_sum.pickle', 'wb') as f:
        pickle.dump(answered_correctly_q_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR+'/elapsed_time_q_sum.pickle', 'wb') as f:
        pickle.dump(elapsed_time_q_sum, f, protocol=pickle.HIGHEST_PROTOCOL)

        
    with open(DATA_DIR+'/explanation_q_sum.pickle', 'wb') as f:
        pickle.dump(explanation_q_sum, f, protocol=pickle.HIGHEST_PROTOCOL)
     
    with open(DATA_DIR+'/timestamp_u.pickle', 'wb') as f:
        pickle.dump(timestamp_u, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR+'/timestamp_u_incorrect.pickle', 'wb') as f:
        pickle.dump(timestamp_u_incorrect, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    del answered_correctly_u_count, timestamp_u_incorrect, timestamp_u, explanation_q_sum
    del elapsed_time_q_sum, answered_correctly_q_sum, answered_correctly_q_count, explanation_u_sum
    del elapsed_time_u_sum, answered_correctly_u_sum