import os
import gc
import sys

import pickle
from time import time

import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
sns.set()
DEFAULT_FIG_WIDTH = 20
sns.set_context("paper", font_scale=1.2) 

from sakt import *

HOME = os.path.dirname(os.path.abspath(__file__))
print(HOME, '\n\n')
MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
get_system()

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

def main_sakt(data_name = 'train.csv'):
    '''
    get the user group for sakt model baseline
        
    data_name = 'cv2_train.parquet'
    data_name = 'cv2_valid.parquet'
    '''
    filename = data_name.split('.')[0]
    print('\n')
    with timer("Loading data"):
        # data_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), 
        #                     usecols=TRAIN_COLS, dtype=TRAIN_DTYPES)
        # data_df = pd.read_parquet(os.path.join(DATA_DIR, 'cv2_train.parquet'),
        #                         columns=list(TRAIN_DTYPES.keys())).astype(TRAIN_DTYPES)
        data_df = pd.read_parquet(os.path.join(DATA_DIR, data_name),
                                columns=list(TRAIN_DTYPES.keys())).astype(TRAIN_DTYPES)
    print('\n')
    with timer("Processing data"):
        data_df = data_df[data_df[CONTENT_TYPE_ID] == False]
        data_df = data_df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)
        group = data_df[[USER_ID, CONTENT_ID, TARGET, TIMESTAMP]]\
                    .groupby(USER_ID)\
                    .apply(lambda r: (r[CONTENT_ID].values, 
                                     r[TARGET].values,
                                     r[TIMESTAMP].values))
    print('\n')
    with timer("Dumping user group to pickle"):
        with open(os.path.join(DATA_DIR, f'sakt_group_{filename}_timed.pickle'), 'wb') as f:
            pickle.dump(group, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main_sakt('cv2_train.parquet')

