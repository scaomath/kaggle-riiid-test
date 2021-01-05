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
'''
Version notes:
1. Testing new features from 
https://www.kaggle.com/gaozhanfire/riiid-lgbm-val-0-788-feature-importance-updated
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


#%%
train_parquet = DATA_DIR+'cv2_train.parquet'
valid_parquet = DATA_DIR+'cv2_valid.parquet'
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
    train = train.iloc[:NROWS_TRAIN]
    train = train.astype(TRAIN_DTYPES)

    valid = pd.read_parquet(valid_parquet, 
    columns=list(TRAIN_DTYPES.keys()))
    valid = valid.iloc[:NROWS_VAL]
    valid = valid.astype(TRAIN_DTYPES)

# Filter by content_type_id to discard lectures
train = train.loc[train.content_type_id == False].reset_index(drop = True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop = True)
#%%
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

#%%
def add_features_new(all_data,
                    question_u_count_dict,
                    question_u_last_bundle_count_dict,
                    elapsed_time_u_sum_dict,
                    explanation_u_sum_dict,
                    answered_correctly_u_count_dict,
                    answered_correctly_u_sum_dict,
                    answered_correctly_uq_dict,
                    part_user_count_dict,
                    part_user_sum_dict,
                    timestamp_u_correct_dict,
                    timestamp_u_incorrect_dict,
                    timestamp_u_dict,
                    user_tag_acc_count_dict,
                    user_tag_acc_sum_dict,
                    update=True):
    # user features
    len_data = len(all_data)
    # in old version
    answered_correctly_u_avg = np.zeros(len_data, dtype = np.float32)
    
    # new
    answered_correctly_u_count = np.zeros(len_data, dtype = np.float32)

    # in old ver user + question 
    answered_correctly_uq_count = np.zeros(len_data, dtype = np.int32)

    # in old
    elapsed_time_u_avg = np.zeros(len_data, dtype = np.float32)

    # in old
    explanation_u_avg = np.zeros(len_data, dtype = np.float32)
    
    # new
    part_user_count = np.zeros(len_data, dtype = np.float32)
    part_user_mean = np.zeros(len_data, dtype = np.float32)
    
    # new, might lead to leakage 
    # b/c the prediction for test might be wrong
    # refer to the rolling mean features
    # might increase this to last 100?
    question_correct_rate_last_20_sum = np.zeros(len_data, dtype = np.float32)
    
    # in old, might add a rolling mean?
    timestamp_u_correct_recency_1 = np.zeros(len_data, dtype = np.float32)

    # in old
    timestamp_u_incorrect_recency_1 = np.zeros(len_data, dtype = np.float32)
    
    # new
    timestamp_u_diff_1 = np.zeros(len_data, dtype = np.float32)
    timestamp_u_diff_2 = np.zeros(len_data, dtype = np.float32)
    timestamp_u_diff_3 = np.zeros(len_data, dtype = np.float32)
    
    # new
    user_tag_acc_count = np.zeros(len_data, dtype = np.float32)
    user_tag_acc_max = np.zeros(len_data, dtype = np.float32)
    user_tag_acc_min = np.zeros(len_data, dtype = np.float32)
    
    list_last_user_task_table=[]####定义数组 用来保存旧组的信息

    list_last_user_task_table_un_back=[]####定义数组 用来保存旧组的信息

    flag_current_task = 0

    all_data_temp=all_data[['user_id', # row[0]
                            'task_container_id', # row[1]
                            'content_id', # row[2]
                            'answered_correctly', # row[3]
                            'prior_question_elapsed_time',# row[4]
                            'prior_question_had_explanation', # row[5]
                            'timestamp',# row[6]
                            'part',# row[7]
                            'tags',# row[8]
                            ]].values

    for num, row in tqdm(enumerate(all_data)):
        # row=all_data_temp[num]
        user_id = row[0]
        timestamp = row[6]
        prior_q_elapsed_time = row[4]
        prior_q_had_explanation = row[5]
        task_container_id = row[1]
        part = row[7]
        if num+1!=len_data:
            row2=all_data_temp[num+1]
        else:
            row2=[-100 for i in range(len(row))]
        
        
        ####*********  elapsed_time_u_avg_xiuzheng和explanation_u_avg_xiuzheng
        if timestamp!=0:##如果时间戳不是0的时候
            if flag_current_task==0:
                question_u_count_dict[user_id] += question_u_last_bundle_count_dict[user_id]

                elapsed_time_u_sum_dict[user_id] += prior_q_elapsed_time*question_u_last_bundle_count_dict[user_id]
                
                explanation_u_sum_dict[user_id] += prior_q_had_explanation*question_u_last_bundle_count_dict[user_id]
            
            elapsed_time_u_avg[num] = elapsed_time_u_sum_dict[user_id]/question_u_count_dict[user_id]
            explanation_u_avg[num] = explanation_u_sum_dict[user_id]/question_u_count_dict[user_id]
            ###⑥只需要当前组的prior（也就是上一组的平均时间或者是否解答），就可以计算了
        
        else:##时间戳为0的时候，肯定是不知道当前组的用时和解答情况的
            elapsed_time_u_avg[num] = np.nan
            explanation_u_avg[num] = np.nan

        flag_current_task=1
        
        ###①求这个特征，需要不断的记录上一组一共有多少道题，到最后用 
        ### （不断累加（每组多少道题*每道题平均时间））/总做题次数
        ###②需要把记录这组有多少道题放在后面计算，在前面计算平均时间并且填充到特征数组里
        
        list_last_user_task_table_un_back.append([user_id])
        ###没换人换组的时候，先不断保存旧组的信息,并且在换人换组的时候也要保存，以防那次信息没被用到
        
        if user_id!=row2[0] or task_container_id!=row2[1]:###换了一个task
            flag_current_task=0
            question_u_last_bundle_count_dict[user_id]=len(list_last_user_task_table_un_back)
            list_last_user_task_table_un_back=[]
            ###在即将换task的时候，把旧组需要换成新组（更换成新组之前，需要先把旧组的信息在上面用完）
            
        ####*********
        
        ####*********   answered_correctly_u_avg、
        #### answered_correctly_u_count和answered_correctly_uq_count
        if answered_correctly_u_count_dict[user_id] != 0:
            answered_correctly_u_avg[num] = answered_correctly_u_sum_dict[user_id] / answered_correctly_u_count_dict[user_id]
            answered_correctly_u_count[num] = answered_correctly_u_count_dict[user_id]
        else:
            answered_correctly_u_avg[num] = 0.67
            answered_correctly_u_count[num] = 0

        answered_correctly_uq_count[num] = answered_correctly_uq_dict[user_id][row[2]]
        ####*********
        
        ####*********   part_user_count和part_user_mean
        if part_user_count_dict[user_id][part]==0:
            part_user_count[num] = 0
            part_user_mean[num] = 0.67
        else:
            part_user_count[num] = part_user_count_dict[user_id][part]
            part_user_mean[num] = part_user_sum_dict[user_id][part]/part_user_count_dict[user_id][part]
        ####*********
        
        ####*********   question_correct_rate_last_20_mean
#         question_correct_rate_last_20_sum[num]=question_correct_last_20_sum_dict[user_id]
        ####*********
        
        
        ####*********   timestamp_u_correct_recency_1，timestamp_u_incorrect_recency_1
        if len(timestamp_u_correct_dict[user_id]) == 0:
            timestamp_u_correct_recency_1[num] = np.nan
        elif len(timestamp_u_correct_dict[user_id]) == 1:
            timestamp_u_correct_recency_1[num] = timestamp - timestamp_u_correct_dict[user_id][0]
            
        if len(timestamp_u_incorrect_dict[user_id]) == 0:
            timestamp_u_incorrect_recency_1[num] = np.nan
        elif len(timestamp_u_incorrect_dict[user_id]) == 1:
            timestamp_u_incorrect_recency_1[num] = timestamp - timestamp_u_incorrect_dict[user_id][0]
        ####*********

        ####*********   timestamp_u_diff_1_2，timestamp_u_diff_2_3，timestamp_u_diff_3_end
        if len(timestamp_u_dict[user_id]) == 0:
            timestamp_u_diff_1[num] = np.nan
            timestamp_u_diff_2[num] = np.nan
            timestamp_u_diff_3[num] = np.nan
        elif len(timestamp_u_dict[user_id]) == 1:
            timestamp_u_diff_1[num] = timestamp - timestamp_u_dict[user_id][0]
            timestamp_u_diff_2[num] = np.nan
            timestamp_u_diff_3[num] = np.nan
        elif len(timestamp_u_dict[user_id]) == 2:
            timestamp_u_diff_1[num] = timestamp - timestamp_u_dict[user_id][1]
            timestamp_u_diff_2[num] = timestamp_u_dict[user_id][1] - timestamp_u_dict[user_id][0]
            timestamp_u_diff_3[num] = np.nan
        elif len(timestamp_u_dict[user_id]) == 3:
            timestamp_u_diff_1[num] = timestamp - timestamp_u_dict[user_id][2]
            timestamp_u_diff_2[num] = timestamp_u_dict[user_id][2] - timestamp_u_dict[user_id][1]
            timestamp_u_diff_3[num] = timestamp_u_dict[user_id][1] - timestamp_u_dict[user_id][0]

        ####*********
        
        ####*********   user_tag_acc_count，user_tag_acc_max，user_tag_acc_min
        if pd.isnull(row[8]):
            user_tag_acc_count[num] = np.nan
            user_tag_acc_max[num] = np.nan
            user_tag_acc_min[num] = np.nan
            continue
        else:
            tag_list_un_back=row[8].split()
            row_all_tag_sum=0
            row_all_tag_count=0
            row_max_tag_mean=-1###尽量搞小
            row_min_tag_mean=1000###尽量搞大

            for single_tag in tag_list_un_back:
                ###先做需要更新的###
                single_tag_sum=user_tag_acc_sum_dict[user_id][single_tag]
                single_tag_count=user_tag_acc_count_dict[user_id][single_tag]
                row_all_tag_sum+=single_tag_sum
                row_all_tag_count+=single_tag_count
                if single_tag_count==0:
                    single_tag_mean=0.67
                else:
                    single_tag_mean=single_tag_sum/single_tag_count
                row_max_tag_mean=max(single_tag_mean,row_max_tag_mean)
                row_min_tag_mean=min(single_tag_mean,row_min_tag_mean)
            if row_all_tag_count==0:
                user_tag_acc_count[num]=0
                user_tag_acc_max[num] = 0.67
                user_tag_acc_min[num] = 0.67
            else:
                user_tag_acc_count[num]=row_all_tag_count
                user_tag_acc_max[num] = row_max_tag_mean
                user_tag_acc_min[num] = row_min_tag_mean
        ####*********
        
        if update:
            answered_correctly_u_count_dict[user_id] += 1
            answered_correctly_u_sum_dict[user_id] += row[3]
            answered_correctly_uq_dict[user_id][row[2]] += 1
            part_user_count_dict[user_id][part] += 1
            part_user_sum_dict[user_id][part] += row[3]
#             if question_correct_last_20_count_dict[user_id]+1<=20:
#                 question_correct_last_20_count_dict[user_id]+=1
#                 question_correct_last_20_sum_dict[user_id]+=row[3]
#                 question_correct_last_20_all_dict[user_id].append(row[3])
#             else:
#                 question_correct_last_20_sum_dict[user_id]+=row[3]
#                 question_correct_last_20_sum_dict[user_id]-=question_correct_last_20_all_dict[user_id][-1]
#                 question_correct_last_20_all_dict[user_id].pop(0)
#                 question_correct_last_20_all_dict[user_id].append(row[3])
            
            tag_list=row[8].split()
            for single_tag in tag_list:
                ######更新一下 user-tag
                user_tag_acc_count_dict[user_id][single_tag] += 1
                user_tag_acc_sum_dict[user_id][single_tag] += row[3]
            
            #'user_id',"task_container_id", 'content_id', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp','part'
            list_last_user_task_table.append([user_id,
                                              task_container_id,
                                              row[2],
                                              row[3],
                                              prior_q_elapsed_time,
                                              prior_q_had_explanation,
                                              timestamp,
                                              part])
            ###没换人换组的时候，先不断保存旧组的信息,并且在换人换组的时候也要保存，以防那次信息没被用到

            if user_id!=row2[0] or task_container_id!=row2[1]:###换了一个task
                
                if len(timestamp_u_dict[user_id]) == 3:
                    timestamp_u_dict[user_id].pop(0)
                    timestamp_u_dict[user_id].append(timestamp)
                else:
                    timestamp_u_dict[user_id].append(timestamp)
                
                ####由于bundle下面包含很多question，每个question都有一个correct，所以需要用列表存储“旧的一整个组”的correct 
                for single_row_last_user_task_table in list_last_user_task_table:
                    if single_row_last_user_task_table[3]==1:
                        if len(timestamp_u_correct_dict[user_id]) == 1:
                            ###这里，就使用user_id就行，因为list_last_user_task_que_timestamp里全都是当前user-task的信息，而非下一个user-task的信息
                            timestamp_u_correct_dict[user_id].pop(0)
                            timestamp_u_correct_dict[user_id].append(single_row_last_user_task_table[6])
                        else:
                            timestamp_u_correct_dict[user_id].append(single_row_last_user_task_table[6])
                    else:
                        if len(timestamp_u_incorrect_dict[user_id]) == 1:###这里，就使用user_id就行，因为list_last_user_task_que_timestamp里全都是当前user-task的信息，而非下一个user-task的信息
                            timestamp_u_incorrect_dict[user_id].pop(0)
                            timestamp_u_incorrect_dict[user_id].append(single_row_last_user_task_table[6])
                        else:
                            timestamp_u_incorrect_dict[user_id].append(single_row_last_user_task_table[6])
                list_last_user_task_table=[]
                ###在即将换task的时候，把旧组需要换成新组（更换成新组之前，需要先把旧组的信息在上面用完）

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
        user_id = row[0]
        if row[3] == 0:
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_sum[user_id] += task_container_id
            if task_container_id == 0:
                if len(timestamp_u_incorrect[user_id]) == 1:
                    timestamp_u_incorrect[user_id].pop(0)
                    timestamp_u_incorrect[user_id].append(prior_q_elapsed_time)
                else:
                    timestamp_u_incorrect[user_id].append(prior_q_elapsed_time)
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_sum[row[2]] += task_container_id
            # ------------------------------------------------------------------
            
#%%

print('\nUser feature calculation started...\n')
add_features_new(train)
add_features_new(valid)

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

# Delete some training data to experiment faster
if DEBUG:
    train = train.sample(15_000_000, random_state = SEED)
gc.collect()
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
            'bagging_freq': 10,
            'bagging_fraction': 0.80
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
sns.barplot(x = feature_importance['Importance'], y = feature_importance['Features'], orient = 'h')
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

NROWS_TEST = 50_000
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
        
        test_df = add_features_test(test_df, 
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
                            update = True)
        
        test_df[target] =  model.predict(test_df[features])
        set_predict(test_df.loc[:,['row_id', target]])
        pbar.update(len(test_df))

print('\nJob Done')

#%%
y_true = valid_df[valid_df.content_type_id == 0][target]
y_pred = pd.concat(predicted)[target]
print('\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))
# %% 
'''
New add_feature function from 
https://www.kaggle.com/gaozhanfire/riiid-lgbm-val-0-788-feature-importance-updated
'''


#     all_data['answered_correctly_u_avg']=answered_correctly_u_avg
#     all_data['answered_correctly_u_count']=answered_correctly_u_count
#     all_data['answered_correctly_uq_count']=answered_correctly_uq_count
#     all_data['elapsed_time_u_avg_xiuzheng']=elapsed_time_u_avg
#     all_data['explanation_u_avg_xiuzheng']=explanation_u_avg
#     all_data['part_user_count']=part_user_count
#     all_data['part_user_mean']=part_user_mean
#     all_data['timestamp_u_correct_recency_1']=timestamp_u_correct_recency_1
#     all_data['timestamp_u_incorrect_recency_1']=timestamp_u_incorrect_recency_1
#     all_data['timestamp_u_diff_1_2']=timestamp_u_diff_1
#     all_data['timestamp_u_diff_2_3']=timestamp_u_diff_2
#     all_data['timestamp_u_diff_3_end']=timestamp_u_diff_3
#     all_data['part_user_count']=part_user_count
#     all_data['part_user_mean']=part_user_mean
#     all_data['user_tag_acc_count']=user_tag_acc_count
#     all_data['user_tag_acc_max']=user_tag_acc_max
#     all_data['user_tag_acc_min']=user_tag_acc_min
    
       
# add_features()