#%%
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score
import torch

HOME =  "/home/scao/Documents/kaggle-riiid-test/"
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
# sys.path.append(HOME)
from sakt import *
from utils import *
from iter_env import *
# from common import *

PRIVATE = False
DEBUG = False
LAST_N = 100
VAL_BATCH_SIZE = 25_600
SIMU_PUB_SIZE = 25_000
MAX_SEQ = 100

class conf:
    METRIC_ = "max"
    FILLNA_VAL = 14_000 # for prior question elapsed time, rounded average in train
    TQDM_INT = 8
    WORKERS = 8 # 0
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    VAL_BATCH_SIZE = 4096
    NUM_EMBED = 128
    NUM_HEADS = 8
    NUM_SKILLS = 13523 # len(skills)
    NUM_TIME = 300 # when scaled by 1000 and round, priori question time's unique values
    MAX_SEQ = 150
    SCALING = 1 # scaling before sigmoid
    PATIENCE = 8 # overfit patience

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

if DEBUG:
    test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    test_df[:SIMU_PUB_SIZE].to_parquet(DATA_DIR+'test_pub_simu.parquet')

#%%
print("Loading test set....")
if PRIVATE:
    test_df = pd.read_pickle(DATA_DIR+'cv2_valid.pickle')
    train_df = pd.read_parquet(DATA_DIR+'cv2_train.parquet')
else:
    test_df = pd.read_parquet(DATA_DIR+'test_pub_simu.parquet')
    train_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
print("Loaded test.")


train_df[PRIOR_QUESTION_TIME].fillna(conf.FILLNA_VAL, inplace=True) 
        # FILLNA_VAL different than all current values
train_df[PRIOR_QUESTION_TIME] = round(train_df[PRIOR_QUESTION_TIME] / TIME_SCALING).astype(np.int16)
train_df[PRIOR_QUESTION_EXPLAIN] = train_df[PRIOR_QUESTION_EXPLAIN].astype(np.float16).fillna(0).astype(np.int8)

train_df = train_df[train_df[CONTENT_TYPE_ID] == False]
train_df = train_df.sort_values([TIMESTAMP], ascending=True).reset_index(drop = True)

train_group = train_df[[USER_ID, CONTENT_ID, PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN, TARGET]]\
    .groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, 
                    r[PRIOR_QUESTION_TIME].values,
                    r[PRIOR_QUESTION_EXPLAIN].values,
                    r[TARGET].values))

example_test = pd.read_csv(DATA_DIR+'example_test.csv')

df_questions = pd.read_csv(DATA_DIR+'questions.csv')

iter_test = Iter_Valid(test_df, max_user=1000)

predicted = []
def set_predict(df):
    predicted.append(df)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n\nUsing device: {device}')
# model_file = find_sakt_model()
model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_head_8_embed_128_auc_0.7529.pt'
tmp = model_file.split('_')
structure = {'n_skills': 13523, 'n_embed': int(tmp[4]), 'n_head':int(tmp[2])}
model_name = model_file.split('/')[-1]
print(f'\nLoading {model_name}...\n')
model = load_sakt_model_new(model_file, structure=structure)
print(f'\nLoaded {model_name}.\n')
model.eval()

#%%
len_test = len(test_df)
with tqdm(total=len_test) as pbar:
    prev_test_df = None
    for (current_test, current_prediction_df) in iter_test:
        if prev_test_df is not None:
            '''Making use of answers to previous questions'''
            answers = eval(current_test["prior_group_answers_correct"].iloc[0])
            responses = eval(current_test["prior_group_responses"].iloc[0])
            prev_test_df['answered_correctly'] = answers
            prev_test_df['user_answer'] = responses
            prev_test_df = prev_test_df[prev_test_df[CONTENT_TYPE_ID] == False]
            prev_group = prev_test_df[[USER_ID, CONTENT_ID, 
                                      PRIOR_QUESTION_TIME, PRIOR_QUESTION_EXPLAIN, TARGET]]\
                                    .groupby(USER_ID)\
                                    .apply(lambda r: (r[CONTENT_ID].values, 
                                                    r[PRIOR_QUESTION_TIME].values,
                                                    r[PRIOR_QUESTION_EXPLAIN].values,
                                                    r[TARGET].values))
            for prev_user_id in prev_group.index:
                prev_group_content = prev_group[prev_user_id][0]
                prev_group_ac = prev_group[prev_user_id][1]
                prev_group_time = prev_group[prev_user_id][2]
                prev_group_exp = prev_group[prev_user_id][3]
                
                if prev_user_id in train_group.index:
                    train_group[prev_user_id] = (np.append(train_group[prev_user_id][0],prev_group_content), 
                                        np.append(train_group[prev_user_id][1],prev_group_ac),
                                        np.append(train_group[prev_user_id][2],prev_group_time),
                                        np.append(train_group[prev_user_id][3],prev_group_exp))
    
                else:
                    train_group[prev_user_id] = (prev_group_content,
                                        prev_group_ac,
                                        prev_group_time,
                                        prev_group_exp)
                
                if len(train_group[prev_user_id][0])>MAX_SEQ:
                    new_group_content = train_group[prev_user_id][0][-MAX_SEQ:]
                    new_group_ac = train_group[prev_user_id][1][-MAX_SEQ:]
                    new_group_time = train_group[prev_user_id][2][-MAX_SEQ:]
                    new_group_exp = train_group[prev_user_id][3][-MAX_SEQ:]
                    train_group[prev_user_id] = (new_group_content,
                                        new_group_ac,
                                        new_group_time,
                                        new_group_exp)

        current_test[PRIOR_QUESTION_TIME].fillna(conf.FILLNA_VAL, inplace=True) 
        # FILLNA_VAL different than all current values
        current_test[PRIOR_QUESTION_TIME] = round(current_test[PRIOR_QUESTION_TIME] / TIME_SCALING).astype(np.int16)
        current_test[PRIOR_QUESTION_EXPLAIN] = current_test[PRIOR_QUESTION_EXPLAIN].astype(np.float16).fillna(0).astype(np.int8)
        previous_test_df = current_test.copy()
        current_test = current_test[current_test[CONTENT_TYPE_ID] == 0]
        
        '''prediction code here'''
        test_dataset = TestDatasetNew(train_group, current_test, 
                                    conf.NUM_SKILLS)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.VAL_BATCH_SIZE, 
                                    shuffle=False, drop_last=False)

        outs = []

        for item in test_dataloader:
            x = item[0].to(device).long()
            target_id = item[1].to(device).long()
            prior_q_time = item[2].to(device).long()
            priori_q_explain = item[3].to(device).long()

            with torch.no_grad():
                output, att_weight = model(x, target_id, prior_q_time, priori_q_explain)

            output = torch.sigmoid(output)
            output = output[:, -1]
            outs.extend(output.view(-1).data.cpu().numpy())
        
        '''prediction code ends'''

        current_test[TARGET] = outs
        set_predict(current_test.loc[:,[ROW_ID, TARGET]])
        pbar.set_description(f"Current test length: {len(current_test):6d}")
        pbar.update(len(current_test))

y_true = test_df[test_df.content_type_id == 0].answered_correctly
y_pred = pd.concat(predicted).answered_correctly
print('\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))
