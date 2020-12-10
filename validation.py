#%%
import pandas as pd
import random
import gc

random.seed(1127)

DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
PRIVATE = False
# %%
train = pd.read_csv(DATA_DIR+'train.csv',
                   dtype={'row_id': 'int64',
                          'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'task_container_id': 'int16',
                          'user_answer': 'int8',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'}
                   )
# %%
valid_split1 = train.groupby('user_id').tail(10)
train_split1 = train[~train.row_id.isin(valid_split1.row_id)]
valid_split1 = valid_split1[valid_split1.content_type_id == 0]
train_split1 = train_split1[train_split1.content_type_id == 0]
print(f'{train_split1.answered_correctly.mean():.3f} {valid_split1.answered_correctly.mean():.3f}')
print(len(train_split1), len(valid_split1))

max_timestamp_u = train[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u.columns = ['user_id', 'max_time_stamp']
MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()

#%%
def rand_time(max_time_stamp):
    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0,interval)
    return rand_time_stamp

max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(rand_time)
train = train.merge(max_timestamp_u, on='user_id', how='left')
train['virtual_time_stamp'] = train.timestamp + train['rand_time_stamp']

train = train.sort_values(['virtual_time_stamp', 'row_id']).reset_index(drop=True)
#%%
val_size = 2500000
for cv in range(5):
    valid = train[-val_size:]
    train = train[:-val_size]
    # check new users and new contents
    new_users = len(valid[~valid.user_id.isin(train.user_id)].user_id.unique())
    valid_question = valid[valid.content_type_id == 0]
    train_question = train[train.content_type_id == 0]
    new_contents = len(valid_question[~valid_question.content_id.isin(train_question.content_id)].content_id.unique())    
    print(f'cv{cv} {train_question.answered_correctly.mean():.3f} {valid_question.answered_correctly.mean():.3f} {new_users} {new_contents}')
    valid.to_pickle(f'cv{cv+1}_valid.pickle')
    train.to_pickle(f'cv{cv+1}_train.pickle')
