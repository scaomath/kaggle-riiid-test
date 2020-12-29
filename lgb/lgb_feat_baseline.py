#%%
import os
from lightgbm.engine import train
import numpy as np
import pandas as pd
from collections import defaultdict
import datatable as dt
import lightgbm as lgb
from matplotlib import pyplot as plt
import random
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc
import pickle
import zipfile
# HOME = "/home/scao/Documents/kaggle-riiid-test/"
HOME = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(HOME) 
MODEL_DIR = HOME+'/model/'
DATA_DIR = HOME+'/data/'

from utils import *
from iter_env import *
get_system()
get_seed(1227)

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 10)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('max_colwidth', 20)
# %%

'''
New version of LGB feat gen as of Dec 18

1. Baseline running on Dec 24, debugging ver (first 12m rows), CV 0.7759, iter_env CV: 0.7473


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

N_FOLD = 1
NROWS_TRAIN = 5_000_000
NROWS_TRAIN_START = 12_000_000
NROWS_TEST = 50_000
# MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
# DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
DEBUG = False
TRAIN = False
# %%
with timer("Loading train"):
    if DEBUG:
        train_df = dt.fread(DATA_DIR+'train.csv', 
                    columns=set(TRAIN_DTYPES.keys())).to_pandas()
        train_df = train_df[NROWS_TRAIN_START:NROWS_TRAIN_START+NROWS_TRAIN]
        
    else:
        train_df = pd.read_parquet(DATA_DIR+'cv5_train.parquet',
                                    columns=list(TRAIN_DTYPES.keys()))
        train_df = train_df.astype(TRAIN_DTYPES)
        train_df = train_df[:40_000_000]
gc.collect()    
# %%

lectures_df = pd.read_csv(DATA_DIR+'lectures.csv')
lectures_df['type_of'] = lectures_df['type_of'].replace('solving question', 'solving_question')

lectures_df = pd.get_dummies(lectures_df, columns=['part', 'type_of'])

part_lectures_columns = [column for column in lectures_df.columns if column.startswith('part')]

types_of_lectures_columns = [column for column in lectures_df.columns if column.startswith('type_of_')]

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
user_lecture_agg = train_df.groupby('user_id')['content_type_id'].agg(['sum', 'count'])
user_lecture_agg=user_lecture_agg.astype('int16')
# %%
#1= if the event was the user watching a lecture.
cum = train_df.groupby('user_id')['content_type_id'].agg(['cumsum', 'cumcount'])
cum['cumcount']=cum['cumcount']+1
train_df['user_interaction_count'] = cum['cumcount'] 
train_df['user_interaction_timestamp_mean'] = train_df['timestamp']/cum['cumcount'] 
train_df['user_lecture_sum'] = cum['cumsum'] 
train_df['user_lecture_lv'] = cum['cumsum'] / cum['cumcount']

train_df.user_lecture_lv=train_df.user_lecture_lv.astype('float16')
train_df.user_lecture_sum=train_df.user_lecture_sum.astype('int16')
train_df.user_interaction_count=train_df.user_interaction_count.astype('int16')
train_df['user_interaction_timestamp_mean']=train_df['user_interaction_timestamp_mean']/(1000*3600)
train_df.user_interaction_timestamp_mean=train_df.user_interaction_timestamp_mean.astype('float32')

#%%
del cum
gc.collect()
# %%
train_df['prior_question_had_explanation'].fillna(False, inplace=True)
train_df = train_df.astype(TRAIN_DTYPES)
train_df = train_df[train_df[target] != -1].reset_index(drop=True)
#%% Whether the content was explained (unused)
content_explanation_agg=train_df[["content_id","prior_question_had_explanation",target]].groupby(["content_id","prior_question_had_explanation"])[target].agg(['mean'])

content_explanation_agg=content_explanation_agg.unstack()

content_explanation_agg=content_explanation_agg.reset_index()
content_explanation_agg.columns = ['content_id', 'content_explanation_false_mean','content_explanation_true_mean']

content_explanation_agg.content_id=content_explanation_agg.content_id.astype('int16')
content_explanation_agg.content_explanation_false_mean=content_explanation_agg.content_explanation_false_mean.astype('float16')
content_explanation_agg.content_explanation_true_mean=content_explanation_agg.content_explanation_true_mean.astype('float16')
# %% attempt numbers
train_df["attempt_no"] = 1
train_df.attempt_no=train_df.attempt_no.astype('int8')
#
attempt_no_agg=train_df.groupby(["user_id","content_id"])["attempt_no"].agg(['sum']).astype('int8')
#attempt_no_agg=attempt_no_agg.astype('int8')
train_df["attempt_no"] = train_df[["user_id","content_id",'attempt_no']].groupby(["user_id","content_id"])["attempt_no"].cumsum()
attempt_no_agg=attempt_no_agg[attempt_no_agg['sum'] >1]
# %% timestamp the maximum timestamp of every user
prior_question_elapsed_time_mean=train_df['prior_question_elapsed_time'].mean()
train_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)

#%%
max_timestamp_u = train_df[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u.columns = ['user_id', 'max_time_stamp']
max_timestamp_u.user_id=max_timestamp_u.user_id.astype('int32')

# %% lagtime is the difference in timestamp
train_df['lagtime'] = train_df.groupby('user_id')['timestamp'].shift()

max_timestamp_u2 = train_df[['user_id','lagtime']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u2.columns = ['user_id', 'max_time_stamp2']
max_timestamp_u2.user_id=max_timestamp_u2.user_id.astype('int32')

train_df['lagtime']=train_df['timestamp']-train_df['lagtime']
lagtime_mean=train_df['lagtime'].mean()
train_df['lagtime'].fillna(lagtime_mean, inplace=True)

train_df['lagtime']=train_df['lagtime']/(1000*3600)
train_df.lagtime=train_df.lagtime.astype('float32')

lagtime_agg = train_df.groupby('user_id')['lagtime'].agg(['mean'])
train_df['lagtime_mean'] = train_df['user_id'].map(lagtime_agg['mean'])
train_df.lagtime_mean=train_df.lagtime_mean.astype('int32')
# lagtime_agg=lagtime_agg.astype('int32')
# %% lagtime2 is the difference in timestamp with 2 steps
train_df['lagtime2'] = train_df.groupby('user_id')['timestamp'].shift(2)

max_timestamp_u3 = train_df[['user_id','lagtime2']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u3.columns = ['user_id', 'max_time_stamp3']
max_timestamp_u3.user_id=max_timestamp_u3.user_id.astype('int32')

train_df['lagtime2']=train_df['timestamp']-train_df['lagtime2']
lagtime_mean2=train_df['lagtime2'].mean()
train_df['lagtime2'].fillna(lagtime_mean2, inplace=True)

train_df['lagtime2']=train_df['lagtime2']/(1000*3600)
train_df.lagtime2=train_df.lagtime2.astype('float32')

# lagtime_agg2 = train_df.groupby('user_id')['lagtime2'].agg(['mean'])
# train_df['lagtime_mean2'] = train_df['user_id'].map(lagtime_agg2['mean'])
# train_df.lagtime_mean2=train_df.lagtime_mean2.astype('int32')
# lagtime_agg2=lagtime_agg2.astype('int32')
#%%
train_df['lagtime3'] = train_df.groupby('user_id')['timestamp'].shift(3)

train_df['lagtime3']=train_df['timestamp']-train_df['lagtime3']
lagtime_mean3=train_df['lagtime3'].mean()
train_df['lagtime3'].fillna(lagtime_mean3, inplace=True)
train_df['lagtime3']=train_df['lagtime3']/(1000*3600)
train_df.lagtime3=train_df.lagtime3.astype('float32')

#%%
train_df['timestamp']=train_df['timestamp']/(1000*3600)
train_df.timestamp=train_df.timestamp.astype('float16')

# %% user_prior_question_elapsed_time unused
user_prior_question_elapsed_time = train_df[['user_id','prior_question_elapsed_time']].groupby(['user_id']).tail(1)
user_prior_question_elapsed_time.columns = ['user_id', 'prior_question_elapsed_time']

train_df['delta_prior_question_elapsed_time'] = train_df.groupby('user_id')['prior_question_elapsed_time'].shift()
train_df['delta_prior_question_elapsed_time']=train_df['prior_question_elapsed_time']-train_df['delta_prior_question_elapsed_time']

delta_prior_question_elapsed_time_mean=train_df['delta_prior_question_elapsed_time'].mean()
train_df['delta_prior_question_elapsed_time'].fillna(delta_prior_question_elapsed_time_mean, inplace=True)
train_df.delta_prior_question_elapsed_time=train_df.delta_prior_question_elapsed_time.astype('int32')
# %%
lag_shift = 1 # is 2 in some previous version
train_df['lag'] = train_df.groupby('user_id')[target].shift(lag_shift)
cum = train_df.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
##cum['cumcount']=cum['cumcount']+1
user_agg = train_df.groupby('user_id')['lag'].agg(['sum', 'count']).astype('int16')
cum['cumsum'].fillna(0, inplace=True)

train_df['user_correctness'] = cum['cumsum'] / cum['cumcount']
train_df['user_correct_count'] = cum['cumsum']
train_df['user_uncorrect_count'] = cum['cumcount']-cum['cumsum']
#train_df['user_answer_count'] = cum['cumcount']
train_df.drop(columns=['lag'], inplace=True)
train_df['user_correctness'].fillna(0.66, inplace=True)
train_df.user_correctness=train_df.user_correctness.astype('float16')
train_df.user_correct_count=train_df.user_correct_count.astype('int16')
train_df.user_uncorrect_count=train_df.user_uncorrect_count.astype('int16')
#train_df.user_answer_count=train_df.user_answer_count.astype('int16')

#%%
del cum
gc.collect()
# %% explanation_agg is not used
train_df.prior_question_had_explanation=train_df.prior_question_had_explanation.astype('int8')
explanation_agg = train_df.groupby('user_id')['prior_question_had_explanation'].agg(['sum', 'count'])
explanation_agg=explanation_agg.astype('int16')
# explanation_agg.sum=explanation_agg.sum.astype('int16')
# explanation_agg.count=explanation_agg.count.astype('int16')
#explanation_agg.var=explanation_agg.var.astype('float16')
# %%
cum = train_df.groupby('user_id')['prior_question_had_explanation'].agg(['cumsum', 'cumcount'])
cum['cumcount']=cum['cumcount']+1
train_df['explanation_mean'] = cum['cumsum'] / cum['cumcount']
train_df['explanation_true_count'] = cum['cumsum'] 
train_df['explanation_false_count'] =  cum['cumcount']-cum['cumsum']
#train_df.drop(columns=['lag'], inplace=True)

train_df.explanation_mean=train_df.explanation_mean.astype('float16')
train_df.explanation_true_count=train_df.explanation_true_count.astype('int16')
train_df.explanation_false_count=train_df.explanation_false_count.astype('int16')

#%%
del cum
gc.collect()
# %% question_id features
# total correct, total for questions and bundles
content_agg = train_df.groupby('content_id')[target].agg(['sum', 'count', 'var'])
task_container_agg = train_df.groupby('task_container_id')[target].agg(['sum', 'count','var'])
# missing_content_id = set(range(13523)).difference(content_agg.index)
# missing_content = pd.DataFrame(missing_content_id, columns=['content_id']).set_index('content_id')
content_agg=content_agg.astype('float32')
task_container_agg=task_container_agg.astype('float32')
# %%
train_df['task_container_uncor_count'] = train_df['task_container_id'].map(task_container_agg['count']-task_container_agg['sum']).astype('int32')
train_df['task_container_cor_count'] = train_df['task_container_id'].map(task_container_agg['sum']).astype('int32')
train_df['task_container_std'] = train_df['task_container_id'].map(task_container_agg['var']).astype('float16')
train_df['task_container_correctness'] = train_df['task_container_id'].map(task_container_agg['sum'] / task_container_agg['count'])
train_df.task_container_correctness=train_df.task_container_correctness.astype('float16')

#train_df.drop(columns=['prior_question_had_explanation'], inplace=True)
# %% these two are not used (neither useful)
content_elapsed_time_agg=train_df.groupby('content_id')['prior_question_elapsed_time'].agg(['mean'])
content_had_explanation_agg=train_df.groupby('content_id')['prior_question_had_explanation'].agg(['mean'])

# %% question data (most important)
questions_df = pd.read_csv(DATA_DIR+'questions.csv', 
    usecols=[0, 1, 3, 4],
    dtype={'question_id': 'int16',
           'bundle_id': 'int16', 
           'part': 'int8', 
           'tags': 'str'}
)

bundle_agg = questions_df.groupby('bundle_id')['question_id'].agg(['count'])
questions_df['content_sub_bundle'] = questions_df['bundle_id'].map(bundle_agg['count']).astype('int8')

questions_df['tags'].fillna('188', inplace=True) # why 188?

#%%


#%%
def gettags(tags,num):
    tags_splits=tags.split(" ")
    result='' 
    for t in tags_splits:
        x=int(t)
        if(x<32*(num+1) and x>=32*num):#num 
            result=result+' '+t
    return result

for num in range(0,6):
    questions_df["tags"+str(num)] = questions_df["tags"].apply(lambda row: gettags(row,num))
    le = LabelEncoder()
    le.fit(np.unique(questions_df['tags'+str(num)].values))
    #questions_df[['tags'+str(num)]=
    questions_df['tags'+str(num)]=questions_df[['tags'+str(num)]].apply(le.transform)

# %%
questions_df_dict = {   
    'tags0': 'int8',
    'tags1': 'int8',
    'tags2': 'int8',
    'tags3': 'int8',
    'tags4': 'int8',
    'tags5': 'int8',
    #'tags6': 'int8',
    #'tags7': 'int8'
}
questions_df = questions_df.astype(questions_df_dict)
# %% no tags in this version
questions_df.drop(columns=['tags'], inplace=True)
questions_df['part_bundle_id']=questions_df['part']*100000+questions_df['bundle_id']
questions_df.part_bundle_id=questions_df.part_bundle_id.astype('int32')

# tag = questions_df["tags"].str.split(" ", n = 10, expand = True)
# tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
# #
# tag.fillna(0, inplace=True)
# tag = tag.astype('int16')
# questions_df =  pd.concat([questions_df,tag],axis=1).drop(['tags'],axis=1)
# %%
questions_df.rename(columns={'question_id':'content_id'}, inplace=True)
questions_df = pd.merge(questions_df, content_explanation_agg, 
                        on='content_id', 
                        how='left', right_index=True)#
# questions_df.content_explanation_false_mean=questions_df.content_explanation_false_mean.astype('float16')
# questions_df.content_explanation_true_mean=questions_df.content_explanation_true_mean.astype('float16')

# %% 
questions_df['content_correctness'] = questions_df['content_id'].map(content_agg['sum'] / content_agg['count'])
questions_df.content_correctness=questions_df.content_correctness.astype('float16')
questions_df['content_correctness_std'] = questions_df['content_id'].map(content_agg['var'])
questions_df.content_correctness_std=questions_df.content_correctness_std.astype('float16')

#%%
# if it is not full train then the questions might not be covered
print(np.where(questions_df['content_correctness'].isnull()))

#%% if it is full data, fillna is un-necessary
questions_df['content_uncorrect_count'] = questions_df['content_id'].map(content_agg['count']-content_agg['sum']).fillna(0).astype('int32')
questions_df['content_correct_count'] = questions_df['content_id'].map(content_agg['sum']).fillna(0).astype('int32')

questions_df['content_elapsed_time_mean'] = questions_df['content_id'].map(content_elapsed_time_agg['mean'])
questions_df.content_elapsed_time_mean=questions_df.content_elapsed_time_mean.astype('float16')
questions_df['content_had_explanation_mean'] = questions_df['content_id'].map(content_had_explanation_agg['mean'])
questions_df.content_had_explanation_mean=questions_df.content_had_explanation_mean.astype('float16')
#%%
del content_elapsed_time_agg, content_had_explanation_agg, content_explanation_agg
gc.collect()
#%%
part_agg = questions_df.groupby('part')['content_correctness'].agg(['mean', 'var'])
questions_df['part_correctness_mean'] = questions_df['part'].map(part_agg['mean'])
questions_df['part_correctness_std'] = questions_df['part'].map(part_agg['var'])
questions_df.part_correctness_mean=questions_df.part_correctness_mean.astype('float16')
questions_df.part_correctness_std=questions_df.part_correctness_std.astype('float16')

part_agg = questions_df.groupby('part')['content_uncorrect_count'].agg(['sum'])
questions_df['part_uncor_count'] = questions_df['part'].map(part_agg['sum']).astype('int32')
#
part_agg = questions_df.groupby('part')['content_correct_count'].agg(['sum'])
questions_df['part_cor_count'] = questions_df['part'].map(part_agg['sum']).astype('int32')

#%%
bundle_agg = questions_df.groupby('bundle_id')['content_correctness'].agg(['mean'])
questions_df['bundle_correctness_mean'] = questions_df['bundle_id'].map(bundle_agg['mean'])
questions_df.bundle_correctness_mean=questions_df.bundle_correctness_mean.astype('float16')

# bundle_agg = questions_df.groupby('bundle_id')['content_uncorrect_count'].agg(['sum'])
# questions_df['bundle_uncor_count'] = questions_df['bundle_id'].map(bundle_agg['sum']).astype('int32')
# #
# bundle_agg = questions_df.groupby('bundle_id')['content_correct_count'].agg(['sum'])
# questions_df['bundle_cor_count'] = questions_df['bundle_id'].map(bundle_agg['sum']).astype('int32')
#%%
del content_agg, bundle_agg, part_agg
#del tags1_agg
gc.collect()

#%%
features_dict = {
    #'user_id',
    'timestamp':'float16',#
    'user_interaction_count':'int16',
    'user_interaction_timestamp_mean':'float32',
    'lagtime':'float32',#
    'lagtime2':'float32',
    'lagtime3':'float32',
    #'lagtime_mean':'int32',
    'content_id':'int16',
    'task_container_id':'int16',
    'user_lecture_sum':'int16',#
    'user_lecture_lv':'float16',##
    'prior_question_elapsed_time':'float32',#
    'delta_prior_question_elapsed_time':'int32',#
    'user_correctness':'float16',#
    'user_uncorrect_count':'int16',#
    'user_correct_count':'int16',#
    #'content_correctness':'float16',
    'content_correctness_std':'float16',
    'content_correct_count':'int32',
    'content_uncorrect_count':'int32',#
    'content_elapsed_time_mean':'float16',
    'content_had_explanation_mean':'float16',
    'content_explanation_false_mean':'float16',
    'content_explanation_true_mean':'float16',
    'task_container_correctness':'float16',
    'task_container_std':'float16',
    'task_container_cor_count':'int32',#
    'task_container_uncor_count':'int32',#
    'attempt_no':'int8',#
    'part':'int8',
    'part_correctness_mean':'float16',
    'part_correctness_std':'float16',
    'part_uncor_count':'int32',
    'part_cor_count':'int32',
    'tags0': 'int8',
    'tags1': 'int8',
    'tags2': 'int8',
    'tags3': 'int8',
    'tags4': 'int8',
    'tags5': 'int8',
   # 'tags6': 'int8',
   # 'tags7': 'int8',
#     'tags0_correctness_mean':'float16',
#     'tags1_correctness_mean':'float16',
#     'tags2_correctness_mean':'float16',
#     'tags4_correctness_mean':'float16',
#     'bundle_id':'int16',
#     'bundle_correctness_mean':'float16',
#     'bundle_uncor_count':'int32',
#     'bundle_cor_count':'int32',
    'part_bundle_id':'int32',
    'content_sub_bundle':'int8',
    'prior_question_had_explanation':'int8',
    'explanation_mean':'float16', #
    #'explanation_var',#
    'explanation_false_count':'int16',#
    'explanation_true_count':'int16',#
   # 'community':'int8',
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
}
categorical_columns= [
    #'user_id',
    'content_id',
    'task_container_id',
    'part',
   # 'community',
    'tags0',
    'tags1',
    'tags2',
    'tags3',
    'tags4',
    'tags5',
    #'tags6',
    #'tags7',
    #'bundle_id',
    'part_bundle_id',
    'content_sub_bundle',
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

features=list(features_dict.keys())
# %%
flag_lgbm=True
clfs = list()
trains=list()
valids=list()

# train_df_clf = train_df[NROWS_TRAIN_START:NROWS_TRAIN_START+NROWS_TRAIN]
train_df_clf=train_df.copy()

del train_df
gc.collect()

#%%
users=train_df_clf['user_id'].drop_duplicates()

users=users.sample(frac=0.05)
users_df=pd.DataFrame()
users_df['user_id']=users.values


valid_df_newuser = pd.merge(train_df_clf, users_df, 
                            on=['user_id'], 
                            how='inner',
                            right_index=True)
del users_df
del users
gc.collect()
#%% 
# CV strategy is to add some unseen users
# then just sample train (need to change CV)
train_df_clf.drop(valid_df_newuser.index, inplace=True)

train_df_clf = pd.merge(train_df_clf, questions_df, 
                        on='content_id', 
                        how='left',
                        right_index=True)#

valid_df_newuser = pd.merge(valid_df_newuser, questions_df, 
                            on='content_id', 
                            how='left',
                            right_index=True)#

#     train_df_clf = pd.merge(train_df_clf, user_lecture_stats_part, on='user_id', how="left",right_index=True)
#     valid_df_newuser = pd.merge(valid_df_newuser, user_lecture_stats_part, on='user_id', how="left",right_index=True)

valid_df=train_df_clf.sample(frac=0.1)
train_df_clf.drop(valid_df.index, inplace=True)

valid_df = valid_df.append(valid_df_newuser)
del valid_df_newuser
gc.collect()
#%%

print('train_df length：',len(train_df_clf))
print('valid_df length：',len(valid_df))
print("Number of features: ", len(train_df_clf.columns))
trains.append(train_df_clf)
valids.append(valid_df)
#%%
del valid_df
gc.collect()
    #train_df=train_df.reset_index(drop=True)
#%%

params = {
            'num_leaves': 100,
            'max_bin': 200,
            'min_child_weight': 0.05,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.58,
            'min_data_in_leaf': 512,
            'objective': 'binary',
            'max_depth': -1,
            'learning_rate': 0.05,
            "boosting_type": "gbdt",
            "bagging_seed": 802,
            "metric": 'auc',
            "verbosity": -1,
            'lambda_l1': 2,
            'lambda_l2': 0.6,
            'random_state': 1127
         }





if TRAIN:
    for i in range(N_FOLD):
        #     tr_data = lgb.Dataset(trains[i][features], label=trains[i][target])
        #     va_data = lgb.Dataset(valids[i][features], label=valids[i][target])

        #Don't use DF to create lightgbm dataset, rather use np array:
        X_train_np = trains[i][features].values.astype(np.float32)
        X_valid_np = valids[i][features].values.astype(np.float32)
        #features = train.columns
        tr_data = lgb.Dataset(X_train_np, label=trains[i][target], 
                                feature_name=list(features))
        va_data = lgb.Dataset(X_valid_np, label=valids[i][target], 
                                feature_name=list(features))

        del X_train_np, X_valid_np
        gc.collect()

        model = lgb.train(
            params, 
            tr_data,
            num_boost_round=5000,
            valid_sets=[tr_data, va_data],
            early_stopping_rounds=50,
            feature_name=features,
            categorical_feature=categorical_columns,
            verbose_eval=50
        )
        val_auc = model.best_score['valid_1']['auc']
        model.save_model(MODEL_DIR+f'lgb_base_fold_{i}_auc_{val_auc:.4f}.txt')   
        clfs.append(model)
    del trains, valids, tr_data, va_data
    gc.collect()
else:
    for _ in range(N_FOLD):
        model = lgb.Booster(model_file=MODEL_DIR+f'lgb_base_auc_0.7759.txt')
        clfs.append(model)


#%%

fig,ax = plt.subplots(figsize=(10,20))
lgb.plot_importance(model, ax=ax,importance_type='gain',max_num_features=50)
plt.show()

# %%

# with open(MODEL_DIR+f'lgb_base_auc_{val_auc:.4f}.pkl', 'wb') as f:
#     pickle.dump(model, f)
# load model with pickle to predict
# with open('model.pkl', 'rb') as fin:
#     pkl_bst = pickle.load(fin)

# archive = zipfile.ZipFile(MODEL_DIR+'lgb_base_auc_0.7727.zip', 'r')
# model_txt = archive.read('lgb_base_auc_0.7727.txt')
# %%

user_sum_dict = user_agg['sum'].astype('int16').to_dict(defaultdict(int))
user_count_dict = user_agg['count'].astype('int16').to_dict(defaultdict(int))

task_container_sum_dict = task_container_agg['sum'].astype('int32').to_dict(defaultdict(int))
task_container_count_dict = task_container_agg['count'].astype('int32').to_dict(defaultdict(int))
task_container_std_dict = task_container_agg['var'].astype('float16').to_dict(defaultdict(int))

explanation_sum_dict = explanation_agg['sum'].astype('int16').to_dict(defaultdict(int))
explanation_count_dict = explanation_agg['count'].astype('int16').to_dict(defaultdict(int))
#explanation_var_dict = explanation_agg['var'].astype('float16').to_dict(defaultdict(int))

user_lecture_sum_dict = user_lecture_agg['sum'].astype('int16').to_dict(defaultdict(int))
user_lecture_count_dict = user_lecture_agg['count'].astype('int16').to_dict(defaultdict(int))


#lagtime_mean_dict = lagtime_agg['mean'].astype('int32').to_dict(defaultdict(int))
#del prior_question_elapsed_time_agg
if not DEBUG:
    del user_agg, task_container_agg, explanation_agg, user_lecture_agg, #lagtime_agg
    gc.collect()
# %%
max_timestamp_u_dict=max_timestamp_u.set_index('user_id').to_dict()
max_timestamp_u_dict2=max_timestamp_u2.set_index('user_id').to_dict()
max_timestamp_u_dict3=max_timestamp_u3.set_index('user_id').to_dict()
user_prior_question_elapsed_time_dict=user_prior_question_elapsed_time.set_index('user_id').to_dict()

attempt_no_sum_dict = attempt_no_agg['sum'].to_dict(defaultdict(int))

if not DEBUG:
    #del question_elapsed_time_agg
    del max_timestamp_u, max_timestamp_u2, max_timestamp_u3
    del user_prior_question_elapsed_time, attempt_no_agg
    gc.collect()
# %%

def get_max_attempt(user_id,content_id):
    k = (user_id,content_id)

    if k in attempt_no_sum_dict.keys():
        attempt_no_sum_dict[k]+=1
        return attempt_no_sum_dict[k]

    attempt_no_sum_dict[k] = 1
    return attempt_no_sum_dict[k]

# def get_max_attempt(user_id,content_id):
#     global  attempt_no_agg
#     #k = (user_id,content_id)
#     if(len(attempt_no_agg[(attempt_no_agg['user_id']==user_id) & (attempt_no_agg['content_id'] ==content_id)])==1):
#     #if k in attempt_no_sum_dict.keys():
#         x= attempt_no_agg.loc[(attempt_no_agg['user_id']==user_id) & (attempt_no_agg['content_id'] ==content_id),'sum'].values 
#         attempt_no_agg.loc[(attempt_no_agg['user_id']==user_id) & (attempt_no_agg['content_id'] ==content_id),'sum']=x+1
#         return x+1
    
#     attempt_no_agg = attempt_no_agg.append([{'user_id':user_id,'content_id':content_id,'sum':1}], ignore_index=True)
#     return 1
# %%
valid_df = pd.read_parquet(DATA_DIR+'cv5_valid.parquet')
valid_df = valid_df[:NROWS_TEST]
iter_test = Iter_Valid(valid_df, max_user=1000)

predicted = []
def set_predict(df):
    predicted.append(df)

#%%

len_test = len(valid_df)
prior_test_df = None

with tqdm(total=len_test) as pbar:
    for (test_df, sample_prediction_df) in iter_test:    
        if prior_test_df is not None:
            prior_test_df[target] = eval(test_df['prior_group_answers_correct'].iloc[0])
            prior_test_df = prior_test_df[prior_test_df[target] != -1].reset_index(drop=True)
            #prior_test_df = prior_test_df[prior_test_df[target] != -1]
            prior_test_df['prior_question_had_explanation'].fillna(False, inplace=True)       
            prior_test_df.prior_question_had_explanation=prior_test_df.prior_question_had_explanation.astype('int8')
        
            user_ids = prior_test_df['user_id'].values
            #content_ids = prior_test_df['content_id'].values
            #task_container_ids = prior_test_df['task_container_id'].values
            #prior_question_had_explanations = prior_test_df['prior_question_had_explanation'].values
            targets = prior_test_df[target].values        
            
            for user_id, answered_correctly in zip(user_ids,targets):
                user_sum_dict[user_id] += answered_correctly
                user_count_dict[user_id] += 1
    #             user_sum_dict2[user_id] += answered_correctly
    #             user_count_dict2[user_id] += 1   
                

        prior_test_df = test_df.copy() 
            
        
        question_len=len( test_df[test_df['content_type_id'] == 0])
        test_df['prior_question_had_explanation'].fillna(False, inplace=True)
        test_df.prior_question_had_explanation=test_df.prior_question_had_explanation.astype('int8')
        test_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)
        

        user_lecture_sum = np.zeros(question_len, dtype=np.int16)
        user_lecture_count = np.zeros(question_len, dtype=np.int16) 
        
        user_sum = np.zeros(question_len, dtype=np.int16)
        user_count = np.zeros(question_len, dtype=np.int16)
    #     user_sum2 = np.zeros(question_len, dtype=np.int16)
    #     user_count2 = np.zeros(question_len, dtype=np.int16)

    #     user_sum_dict_test=defaultdict(int)
    #     user_count_dict_test=defaultdict(int)

        task_container_sum = np.zeros(question_len, dtype=np.int32)
        task_container_count = np.zeros(question_len, dtype=np.int32)
        task_container_std = np.zeros(question_len, dtype=np.float16)

        explanation_sum = np.zeros(question_len, dtype=np.int32)
        explanation_count = np.zeros(question_len, dtype=np.int32)
        delta_prior_question_elapsed_time = np.zeros(question_len, dtype=np.int32)

        attempt_no_count = np.zeros(question_len, dtype=np.int16)
        lagtime = np.zeros(question_len, dtype=np.float32)
        lagtime2 = np.zeros(question_len, dtype=np.float32)
        lagtime3 = np.zeros(question_len, dtype=np.float32)
        #lagtime_means = np.zeros(question_len, dtype=np.int32)
        #
    
        i=0
        for j, (user_id,prior_question_had_explanation,content_type_id,prior_question_elapsed_time,timestamp, content_id,task_container_id) in enumerate(zip(test_df['user_id'].values,test_df['prior_question_had_explanation'].values,test_df['content_type_id'].values,test_df['prior_question_elapsed_time'].values,test_df['timestamp'].values, test_df['content_id'].values, test_df['task_container_id'].values)):   
            #
            user_lecture_sum_dict[user_id] += content_type_id
            user_lecture_count_dict[user_id] += 1
            if(content_type_id==1):#
                x=1
    #             if(len(user_lecture_stats_part[user_lecture_stats_part.user_id==user_id])==0):
    #                 user_lecture_stats_part = user_lecture_stats_part.append([{'user_id':user_id}], ignore_index=True)
    #                 user_lecture_stats_part.fillna(0, inplace=True)
    #                 user_lecture_stats_part.loc[user_lecture_stats_part.user_id==user_id,part_lectures_columns + types_of_lectures_columns]+=lectures_df[lectures_df.lecture_id==content_id][part_lectures_columns + types_of_lectures_columns].values
    #             else:
    #                 user_lecture_stats_part.loc[user_lecture_stats_part.user_id==user_id,part_lectures_columns + types_of_lectures_columns]+=lectures_df[lectures_df.lecture_id==content_id][part_lectures_columns + types_of_lectures_columns].values
            if (content_type_id==0):#   
                user_lecture_sum[i] = user_lecture_sum_dict[user_id]
                user_lecture_count[i] = user_lecture_count_dict[user_id]
                    
                user_sum[i] = user_sum_dict[user_id]
                user_count[i] = user_count_dict[user_id]
    #             user_sum2[i] = user_sum_dict2[user_id]
    #             user_count2[i] = user_count_dict2[user_id]
        #         content_sum[i] = content_sum_dict[content_id]
        #         content_count[i] = content_count_dict[content_id]
                task_container_sum[i] = task_container_sum_dict[task_container_id]
                task_container_count[i] = task_container_count_dict[task_container_id]
                task_container_std[i]=task_container_std_dict[task_container_id]

                explanation_sum_dict[user_id] += prior_question_had_explanation
                explanation_count_dict[user_id] += 1
                explanation_sum[i] = explanation_sum_dict[user_id]
                explanation_count[i] = explanation_count_dict[user_id]

                if user_id in max_timestamp_u_dict['max_time_stamp'].keys():
                    lagtime[i]=timestamp-max_timestamp_u_dict['max_time_stamp'][user_id]
                    if(max_timestamp_u_dict2['max_time_stamp2'][user_id]==lagtime_mean2):#
                        lagtime2[i]=lagtime_mean2
                        lagtime3[i]=lagtime_mean3
                        #max_timestamp_u_dict3['max_time_stamp3'].update({user_id:lagtime_mean3})
                    else:
                        lagtime2[i]=timestamp-max_timestamp_u_dict2['max_time_stamp2'][user_id]
                        if(max_timestamp_u_dict3['max_time_stamp3'][user_id]==lagtime_mean3):
                            lagtime3[i]=lagtime_mean3 #
                        else:
                            lagtime3[i]=timestamp-max_timestamp_u_dict3['max_time_stamp3'][user_id]
                        
                        max_timestamp_u_dict3['max_time_stamp3'][user_id]=max_timestamp_u_dict2['max_time_stamp2'][user_id]
                            
                    max_timestamp_u_dict2['max_time_stamp2'][user_id]=max_timestamp_u_dict['max_time_stamp'][user_id]
                    max_timestamp_u_dict['max_time_stamp'][user_id]=timestamp
    #                 lagtime_means[i]=(lagtime_mean_dict[user_id]+lagtime[i])/2
    #                 lagtime_mean_dict[user_id]=lagtime_means[i]
                else:
                    lagtime[i]=lagtime_mean
                    max_timestamp_u_dict['max_time_stamp'].update({user_id:timestamp})
                    lagtime2[i]=lagtime_mean2#
                    max_timestamp_u_dict2['max_time_stamp2'].update({user_id:lagtime_mean2})
                    lagtime3[i]=lagtime_mean3#
                    max_timestamp_u_dict3['max_time_stamp3'].update({user_id:lagtime_mean3})
    #                 lagtime_mean_dict.update({user_id:lagtime_mean})
    #                 lagtime_means[i]=(lagtime_mean_dict[user_id]+lagtime[i])/2

                if user_id in user_prior_question_elapsed_time_dict['prior_question_elapsed_time'].keys():            
                    delta_prior_question_elapsed_time[i]=prior_question_elapsed_time-user_prior_question_elapsed_time_dict['prior_question_elapsed_time'][user_id]
                    user_prior_question_elapsed_time_dict['prior_question_elapsed_time'][user_id]=prior_question_elapsed_time
                else:           
                    delta_prior_question_elapsed_time[i]=delta_prior_question_elapsed_time_mean    
                    user_prior_question_elapsed_time_dict['prior_question_elapsed_time'].update({user_id:prior_question_elapsed_time})
                i=i+1 


            
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop=True)
        #test_df = test_df[test_df['content_type_id'] == 0]
        #right_index=True
        #test_df = pd.merge(test_df, questions_df, on='content_id', how='left',right_index=True)    
        #test_df = pd.concat([test_df.reset_index(drop=True), questions_df.reindex(test_df['content_id'].values).reset_index(drop=True)], axis=1)
        test_df=test_df.merge(questions_df.loc[questions_df.index.isin(test_df['content_id'])],
                    how='left', on='content_id', right_index=True)
        
        #test_df = pd.merge(test_df, user_lecture_stats_part, on=['user_id'], how="left",right_index=True)
        #test_df = pd.concat([test_df.reset_index(drop=True), user_lecture_stats_part.reindex(test_df['user_id'].values).reset_index(drop=True)], axis=1)
    #     test_df=test_df.merge(user_lecture_stats_part.loc[user_lecture_stats_part.index.isin(test_df['user_id'])],
    #                   how='left', on='user_id', right_index=True)
    
        test_df['user_lecture_lv'] = user_lecture_sum / user_lecture_count
        test_df['user_lecture_sum'] = user_lecture_sum
        
        test_df['user_interaction_count'] = user_lecture_count
        test_df['user_interaction_timestamp_mean'] = test_df['timestamp']/user_lecture_count
        
        test_df['user_correctness'] = user_sum / user_count
        test_df['user_uncorrect_count'] =user_count-user_sum
        test_df['user_correct_count'] =user_sum
        #test_df['user_answer_count'] =user_count
        
    #     test_df['user_correctness2'] = user_sum2 / user_count2
    #     test_df['user_uncorrect_count2'] =user_count2-user_sum2
    #     test_df['user_correct_count2'] =user_sum2
        #test_df['user_answer_count2'] =user_count2
        
        #    
        test_df['task_container_correctness'] = task_container_sum / task_container_count
        test_df['task_container_cor_count'] = task_container_sum 
        test_df['task_container_uncor_count'] =task_container_count-task_container_sum 
        test_df['task_container_std'] = task_container_std 
        #test_df['content_task_mean'] = content_task_mean 
        
        test_df['explanation_mean'] = explanation_sum / explanation_count
        test_df['explanation_true_count'] = explanation_sum
        test_df['explanation_false_count'] = explanation_count-explanation_sum 
        
        #
        test_df['delta_prior_question_elapsed_time'] = delta_prior_question_elapsed_time 
        
    
    
        test_df["attempt_no"] = test_df[["user_id", "content_id"]].apply(lambda row: get_max_attempt(row["user_id"], row["content_id"]), axis=1)
        test_df["lagtime"]=lagtime
        test_df["lagtime2"]=lagtime2
        test_df["lagtime3"]=lagtime3
        #test_df["lagtime_mean"]=lagtime_means

        

        test_df['timestamp']=test_df['timestamp']/(1000*3600)
        test_df.timestamp=test_df.timestamp.astype('float16')
        test_df['lagtime']=test_df['lagtime']/(1000*3600)
        test_df.lagtime=test_df.lagtime.astype('float32')
        test_df['lagtime2']=test_df['lagtime2']/(1000*3600)
        test_df.lagtime2=test_df.lagtime2.astype('float32')
        test_df['lagtime3']=test_df['lagtime3']/(1000*3600)
        test_df.lagtime3=test_df.lagtime3.astype('float32')
        test_df['user_interaction_timestamp_mean']=test_df['user_interaction_timestamp_mean']/(1000*3600)
        test_df.user_interaction_timestamp_mean=test_df.user_interaction_timestamp_mean.astype('float32')
        
        test_df['user_correctness'].fillna(0.66, inplace=True)
        #test_df['user_correctness2'].fillna(0.66, inplace=True)
        #
        #test_df = test_df.astype(features_dict)

        sub_preds = np.zeros(test_df.shape[0])
        for i, model in enumerate(clfs, 1):
            test_preds  = model.predict(test_df[features])
            sub_preds += test_preds
        test_df[target] = sub_preds / len(clfs)
        
    #     if(flag_lgbm):
    #         test_df[target] = model.predict(test_df[features])
    #     else:
    #         test_df[target] = model.predict(test_df[features].values)
        set_predict(test_df[['row_id', target]])
        pbar.update(len(test_df))
# %%
y_true = valid_df[valid_df.content_type_id == 0].answered_correctly
y_pred = pd.concat(predicted).answered_correctly
print('\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))

# %%
