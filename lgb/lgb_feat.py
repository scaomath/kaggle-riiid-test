#%%
import numpy as np
import pandas as pd
from collections import defaultdict
import datatable as dt
import lightgbm as lgb
from matplotlib import pyplot as plt
import random
from sklearn.metrics import roc_auc_score
import gc
import pickle
import zipfile
HOME = "/home/scao/Documents/kaggle-riiid-test/"
sys.path.append(HOME) 
from utils import *
get_system()
# %%

'''
Pandas/Dask Study notes:
1. df.drop_duplicates() returns a df, yet df.unique() returns a numpy array

'''






'''
Columns placeholder and preprocessing params
'''
CONTENT_TYPE_ID = "content_type_id"
CONTENT_ID = "content_id"
TARGET = "answered_correctly"
target = TARGET
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

MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
# %%
train_df = dt.fread(DATA_DIR+'train.csv', columns=set(TRAIN_DTYPES.keys())).to_pandas()

# %%
lectures_df = pd.read_csv(DATA_DIR+'lectures.csv')
lectures_df['type_of'] = lectures_df['type_of'].replace('solving question', 'solving_question')

lectures_df = pd.get_dummies(lectures_df, columns=['part', 'type_of'])

part_lectures_columns = [column for column in lectures_df.columns if column.startswith('part')]

types_of_lectures_columns = [column for column in lectures_df.columns if column.startswith('type_of_')]
# %%
train_lectures = train_df[train_df.content_type_id == True]\
                .merge(lectures_df, 
                        left_on='content_id', 
                        right_on='lecture_id', 
                        how='left')
user_lecture_stats_part = train_lectures.groupby('user_id',as_index = False)[part_lectures_columns + types_of_lectures_columns].sum()
# %%
lecturedata_types_dict = {   
    'user_id': 'int32', 
    'part_1': 'int8',
    'part_2': 'int8',
    'part_3': 'int8',
    'part_4': 'int8',
    'part_5': 'int8',
    'part_6': 'int8',
    'part_7': 'int8',
    'type_of_concept': 'int8',
    'type_of_intention': 'int8',
    'type_of_solving_question': 'int8',
    'type_of_starter': 'int8'
}
user_lecture_stats_part = user_lecture_stats_part.astype(lecturedata_types_dict)
# %%
for column in user_lecture_stats_part.columns:
    #bool_column = column + '_boolean'
    if(column !='user_id'):
        user_lecture_stats_part[column] = (user_lecture_stats_part[column] > 0).astype('int8')
# %%
train_lectures[train_lectures.user_id==5382]
#%%
user_lecture_stats_part[user_lecture_stats_part.user_id==5382]
# %%
user_lecture_stats_part.tail()
# %%
cum = train_df.groupby('user_id')['content_type_id'].agg(['cumsum', 'cumcount'])
train_df['user_lecture_cumsum'] = cum['cumsum'] 
train_df['user_lecture_lv'] = cum['cumsum'] / cum['cumcount']


train_df.user_lecture_lv=train_df.user_lecture_lv.astype('float16')
train_df.user_lecture_cumsum=train_df.user_lecture_cumsum.astype('int8')
user_lecture_agg = train_df.groupby('user_id')['content_type_id'].agg(['sum', 'count'])
# %%
train_df['prior_question_had_explanation'].fillna(False, inplace=True)
train_df = train_df.astype(TRAIN_DTYPES)
train_df = train_df[train_df[TARGET] != -1].reset_index(drop=True)
prior_question_elapsed_time_mean=train_df['prior_question_elapsed_time'].mean()
train_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)
# %%
max_timestamp_u = train_df[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
#max_timestamp_u = train_df[['user_id','timestamp']].groupby(['user_id']).agg(['max'])
max_timestamp_u.columns = ['user_id', 'max_time_stamp']
# %%
train_df['lagtime'] = train_df.groupby('user_id')['timestamp'].shift()
train_df['lagtime']=train_df['timestamp']-train_df['lagtime']
train_df['lagtime'].fillna(0, inplace=True)
train_df.lagtime=train_df.lagtime.astype('int32')
#train_df.drop(columns=['timestamp'], inplace=True)
# %%
lagtime_agg = train_df.groupby('user_id')['lagtime'].agg(['mean'])
train_df['lagtime_mean'] = train_df['user_id'].map(lagtime_agg['mean'])
train_df.lagtime_mean=train_df.lagtime_mean.astype('int32')
# %%
user_prior_question_elapsed_time = train_df[['user_id','prior_question_elapsed_time']].groupby(['user_id']).tail(1)
#max_timestamp_u = train_df[['user_id','timestamp']].groupby(['user_id']).agg(['max'])
user_prior_question_elapsed_time.columns = ['user_id', 'prior_question_elapsed_time']
# %%
train_df['delta_prior_question_elapsed_time'] = train_df.groupby('user_id')['prior_question_elapsed_time'].shift()
train_df['delta_prior_question_elapsed_time']=train_df['prior_question_elapsed_time']-train_df['delta_prior_question_elapsed_time']
train_df['delta_prior_question_elapsed_time'].fillna(0, inplace=True)
# %%
train_df.delta_prior_question_elapsed_time=train_df.delta_prior_question_elapsed_time.astype('int32')
# %% hours by a 1000*3600 scaling
train_df['timestamp']=train_df['timestamp']/(1000*3600)
train_df.timestamp=train_df.timestamp.astype('int16')
#
# %%
train_df['lag'] = train_df.groupby('user_id')[TARGET].shift()

cum = train_df.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
train_df['user_correctness'] = cum['cumsum'] / cum['cumcount']
train_df['user_correct_cumsum'] = cum['cumsum']
train_df['user_correct_cumcount'] = cum['cumcount']
train_df.drop(columns=['lag'], inplace=True)

# train_df['user_correctness'].fillna(1, inplace=True)
train_df['user_correct_cumsum'].fillna(0, inplace=True)
#train_df['user_correct_cumcount'].fillna(0, inplace=True)
train_df.user_correctness=train_df.user_correctness.astype('float16')
train_df.user_correct_cumcount=train_df.user_correct_cumcount.astype('int16')
train_df.user_correct_cumsum=train_df.user_correct_cumsum.astype('int16')
# %%
train_df.prior_question_had_explanation=train_df.prior_question_had_explanation.astype('int8')

train_df['lag'] = train_df.groupby('user_id')['prior_question_had_explanation'].shift()
# %%
cum = train_df.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
train_df['explanation_mean'] = cum['cumsum'] / cum['cumcount']
train_df['explanation_cumsum'] = cum['cumsum'] 
train_df.drop(columns=['lag'], inplace=True)

train_df['explanation_mean'].fillna(0, inplace=True)
train_df['explanation_cumsum'].fillna(0, inplace=True)
train_df.explanation_mean=train_df.explanation_mean.astype('float16')
train_df.explanation_cumsum=train_df.explanation_cumsum.astype('int16')
# %%
del cum
gc.collect()
# %%
train_df["attempt_no"] = 1
train_df.attempt_no=train_df.attempt_no.astype('int8')
train_df["attempt_no"] = train_df[["user_id","content_id",'attempt_no']]\
    .groupby(["user_id","content_id"])["attempt_no"].cumsum()
# %%
explanation_agg = train_df.groupby('user_id')['prior_question_had_explanation'].agg(['sum', 'count'])
explanation_agg=explanation_agg.astype('int16')
#train_df.drop(columns=['prior_question_had_explanation'], inplace=True)
# %%
user_agg = train_df.groupby('user_id')[target].agg(['sum', 'count'])
content_agg = train_df.groupby('content_id')[target].agg(['sum', 'count','var'])
task_container_agg = train_df.groupby('task_container_id')[target].agg(['sum', 'count','var'])

#prior_question_elapsed_time_agg = train_df.groupby('user_id')['prior_question_elapsed_time'].agg(['sum', 'count'])
# %%
user_agg=user_agg.astype('int16')
content_agg=content_agg.astype('float32')
task_container_agg=task_container_agg.astype('float32')
# %%
attempt_no_agg=train_df.groupby(["user_id","content_id"])["attempt_no"].agg(['sum'])
attempt_no_agg=attempt_no_agg.astype('int8')
#attempt_series = train_df[['user_id', 'content_id','attempt_no']].groupby(['user_id','content_id'])['attempt_no'].max()
# %%
train_df['content_count'] = train_df['content_id'].map(content_agg['count']).astype('int32')
train_df['content_sum'] = train_df['content_id'].map(content_agg['sum']).astype('int32')
train_df['content_correctness'] = train_df['content_id'].map(content_agg['sum'] / content_agg['count'])
train_df.content_correctness=train_df.content_correctness.astype('float16')
train_df['task_container_sum'] = train_df['task_container_id'].map(task_container_agg['sum']).astype('int32')
train_df['task_container_std'] = train_df['task_container_id'].map(task_container_agg['var']).astype('float16')
train_df['task_container_correctness'] = train_df['task_container_id'].map(task_container_agg['sum'] / task_container_agg['count'])
train_df.task_container_correctness=train_df.task_container_correctness.astype('float16')
# %%
questions_df = pd.read_csv(DATA_DIR+'questions.csv', 
    usecols=[0, 1, 3, 4],
    dtype={'question_id': 'int16',
           'bundle_id': 'int16', 
           'part': 'int8', 
           'tags': 'str'}
)
questions_df['part_bundle_id']=questions_df['part']*100_000+questions_df['bundle_id']
questions_df.part_bundle_id=questions_df.part_bundle_id.astype('int32')
tag = questions_df["tags"].str.split(" ", n = 10, expand = True)
tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
#

tag.fillna(0, inplace=True)
tag = tag.astype('int16')
questions_df =  pd.concat([questions_df,tag],axis=1).drop(['tags'],axis=1)
# %%
questions_df.rename(columns={'question_id':'content_id'}, inplace=True)
# %%
questions_df['content_correctness'] = questions_df['content_id'].map(content_agg['sum'] / content_agg['count'])
questions_df.content_correctness=questions_df.content_correctness.astype('float16')
questions_df['content_correctness_std'] = questions_df['content_id'].map(content_agg['var'])
questions_df.content_correctness_std=questions_df.content_correctness_std.astype('float16')
# %%

part_agg = questions_df.groupby('part')['content_correctness'].agg(['mean', 'var'])
questions_df['part_correctness_mean'] = questions_df['part'].map(part_agg['mean'])
questions_df['part_correctness_std'] = questions_df['part'].map(part_agg['var'])
questions_df.part_correctness_mean=questions_df.part_correctness_mean.astype('float16')
questions_df.part_correctness_std=questions_df.part_correctness_std.astype('float16')
# %%
bundle_agg = questions_df.groupby('bundle_id')['content_correctness'].agg(['mean'])
questions_df['bundle_correctness'] = questions_df['bundle_id'].map(bundle_agg['mean'])
questions_df.bundle_correctness=questions_df.bundle_correctness.astype('float16')
# %%
tags1_agg = questions_df.groupby('tags1')['content_correctness'].agg(['mean', 'var'])
questions_df['tags1_correctness_mean'] = questions_df['tags1'].map(tags1_agg['mean'])
questions_df['tags1_correctness_std'] = questions_df['tags1'].map(tags1_agg['var'])
questions_df.tags1_correctness_mean=questions_df.tags1_correctness_mean.astype('float16')
questions_df.tags1_correctness_std=questions_df.tags1_correctness_std.astype('float16')
# %%
questions_df.drop(columns=['content_correctness'], inplace=True)
# %%
print(questions_df.dtypes)
del bundle_agg
del part_agg
del tags1_agg
gc.collect()
# %%
train_df['user_correctness'].fillna(1, inplace=True)
train_df['attempt_no'].fillna(1, inplace=True)
#
train_df.fillna(0, inplace=True)
# %%
features = [
    #'user_id',
    'timestamp',
    'lagtime',
    'lagtime_mean',
    'content_id',
    'task_container_id',
    'user_lecture_cumsum',
    'user_lecture_lv',
    'prior_question_elapsed_time',
    'delta_prior_question_elapsed_time',
    'user_correctness',
    'user_correct_cumcount',
    'user_correct_cumsum',
    'content_correctness',
    'content_correctness_std',
    'content_count',
    'content_sum',
    'task_container_correctness',
    'task_container_std',
    'task_container_sum',
    'bundle_correctness',
    'attempt_no',
    'part',
    'part_correctness_mean',
    'part_correctness_std',
    'tags1',
    'tags1_correctness_mean',
    'tags1_correctness_std',
    'tags2',
    'tags3',
    'tags4',
    'tags5',
    'tags6',
    'bundle_id',
    'part_bundle_id',
    'explanation_mean', 
    'explanation_cumsum',
    'prior_question_had_explanation',
#     'part_1',
#     'part_2',
#     'part_3',
#     'part_4',
#     'part_5',
#     'part_6',
#     'part_7',
#     'type_of_concept',
#     'type_of_intention',
#     'type_of_solving_question',
#     'type_of_starter'
]
categorical_columns= [
    #'user_id',
    'content_id',
    'task_container_id',
    'part',        
    'tags1',
    'tags2',
    'tags3',
    'tags4',
    'tags5',
    'tags6',
    'bundle_id',
    'part_bundle_id',
    'prior_question_had_explanation',
#     'part_1',
#     'part_2',
#     'part_3',
#     'part_4',
#     'part_5',
#     'part_6',
#     'part_7',
#     'type_of_concept',
#     'type_of_intention',
#     'type_of_solving_question',
#     'type_of_starter'
]
# %%
flag_lgbm=True
clfs = list()


trains=list()
valids=list()

N_FOLD=1
# for i in range(N_FOLD):
#train_df=train_df.reset_index(drop=True)
train_df_clf=train_df.sample(n=20_000_000)
print('sample end')
#train_df.drop(train_df_clf.index, inplace=True)
#print('train_df drop end')


del train_df
gc.collect()

users=train_df_clf['user_id'].drop_duplicates()

users=users.sample(frac=0.02)
users_df=pd.DataFrame()
users_df['user_id']=users.values


valid_df_newuser = pd.merge(train_df_clf, users_df, on=['user_id'], how='inner',right_index=True)
del users_df
del users
gc.collect()
#
train_df_clf.drop(valid_df_newuser.index, inplace=True)

#-----------
#train_df_clf=train_df_clf.sample(frac=0.2)
#train_df_clf.drop(valid_df_newuser.index, inplace=True)
train_df_clf = pd.merge(train_df_clf, questions_df, on='content_id', how='left',right_index=True)#
valid_df_newuser = pd.merge(valid_df_newuser, questions_df, on='content_id', how='left',right_index=True)#

#     train_df_clf = pd.merge(train_df_clf, user_lecture_stats_part, on='user_id', how="left",right_index=True)
#     valid_df_newuser = pd.merge(valid_df_newuser, user_lecture_stats_part, on='user_id', how="left",right_index=True)

valid_df=train_df_clf.sample(frac=0.1)
train_df_clf.drop(valid_df.index, inplace=True)

valid_df = valid_df.append(valid_df_newuser)
del valid_df_newuser
gc.collect()
#

print('train_df length：',len(train_df_clf))
print('valid_df length：',len(valid_df))
trains.append(train_df_clf)
valids.append(valid_df)

del valid_df
gc.collect()
    #train_df=train_df.reset_index(drop=True)
# %%
# assert len(trains) <= N_FOLD
i = 0
tr_data = lgb.Dataset(trains[i][features], label=trains[i][target])
va_data = lgb.Dataset(valids[i][features], label=valids[i][target])

#     del train_df_clf
#     del valid_df
#     gc.collect()

# del trains
# del valids
# gc.collect()

#%%

params = {
            'num_leaves': 100,
            'max_bin': 700,
            'min_child_weight': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.58,
            'min_data_in_leaf': 1024,
            'objective': 'binary',
            'max_depth': -1,
            'learning_rate': 0.03,
            "boosting_type": "gbdt",
            "bagging_seed": 802,
            "metric": 'auc',
            "verbosity": -1,
            'lambda_l1': 2,
            'lambda_l2': 0.6,
            'random_state': 1127
         }

# params = {'boosting_type': 'gbdt',
#           'max_depth' : -1,
#           'objective': 'binary',
#           'nthread': 3, # Updated from nthread
#           'num_leaves': 64,
#           'learning_rate': 0.05,
#           'max_bin': 512,
#           'subsample_for_bin': 200,
#           'subsample': 1,
#           'subsample_freq': 1,
#           'colsample_bytree': 0.8,
#           'reg_alpha': 5,
#           'reg_lambda': 10,
#           'min_split_gain': 0.5,
#           'min_child_weight': 1,
#           'min_child_samples': 5,
#           'scale_pos_weight': 1,
#           'num_class' : 1,
#           'metric' : 'binary_error'}

model = lgb.train(
    params, 
    tr_data,
#         train_df[features],
#         train_df[target],
    num_boost_round=5000,
    #valid_sets=[(train_df[features],train_df[target]), (valid_df[features],valid_df[target])], 
    valid_sets=[tr_data, va_data],
    early_stopping_rounds=80,
    feature_name=features,
    categorical_feature=categorical_columns,
    verbose_eval=50
)

clfs.append(model)

#%%
#print('auc:', roc_auc_score(valid_df[target], model.predict(valid_df[features])))
#model.save_model(f'model.txt')
lgb.plot_importance(model, importance_type='gain', max_num_features=20)
plt.show()

#    
# del trains
# del valids
# gc.collect()
# %%
val_auc = model.best_score['valid_1']['auc']
model.save_model(MODEL_DIR+f'lgb_base_auc_{val_auc:.4f}.txt')
# with open(MODEL_DIR+f'lgb_base_auc_{val_auc:.4f}.pkl', 'wb') as f:
#     pickle.dump(model, f)
# load model with pickle to predict
# with open('model.pkl', 'rb') as fin:
#     pkl_bst = pickle.load(fin)


archive = zipfile.ZipFile(MODEL_DIR+'lgb_base_auc_0.7727.zip', 'r')
model_txt = archive.read('lgb_base_auc_0.7727.txt')
# %%
params = {
            'num_leaves': 160,
            'max_bin': 700,
            'min_child_weight': 0.05,
            'feature_fraction': 0.7,
            "bagging_freq": 3,
            'bagging_fraction': 0.7,
            'min_data_in_leaf': 1024,
            'objective': 'binary',
            'max_depth': -1,
            'learning_rate': 0.01,
            "boosting_type": "gbdt",
            "bagging_seed": 802,
            "metric": 'auc',
            "verbosity": -1,
            'lambda_l1': 2,
            'lambda_l2': 0.6,
            'random_state': 1127
         }

# %%
# %%
