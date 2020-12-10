#%%
import gc
import random
import numpy as np
import pandas as pd
import cudf
import matplotlib.pyplot as plt


def fast_merge(left, right, key):
    return cudf.concat([left.reset_index(drop=True), 
                        right.reindex(left[key].values).reset_index(drop=True)], 
                        axis=1)

DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
SEED = 1127
random.seed(SEED)
np.random.seed(SEED)

# %%
TRAIN_DTYPES = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'answered_correctly':'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}

train = cudf.read_csv(DATA_DIR+'train.csv', dtype=TRAIN_DTYPES)
# %%  parameters for beta distributions
a = 2.2
b = 2.3
max_timestamp_u = train[['user_id','timestamp']].groupby(['user_id']).max()
max_timestamp_u.columns = ['max_timestamp']
max_timestamp_u['interval'] = max_timestamp_u.max_timestamp.max() - max_timestamp_u.max_timestamp
# max_timestamp_u['random'] = np.random.rand(len(max_timestamp_u))
max_timestamp_u['random'] = np.random.beta(a, b, len(max_timestamp_u))
max_timestamp_u['random_timestamp'] = max_timestamp_u.interval * max_timestamp_u.random
max_timestamp_u['random_timestamp'] = max_timestamp_u.random_timestamp.astype(int)
max_timestamp_u.drop(['interval', 'random'], axis=1, inplace=True)

#%%
train = fast_merge(train, max_timestamp_u, 'user_id')
train['virtual_timestamp'] = train.timestamp + train.random_timestamp
train.set_index(['virtual_timestamp', 'row_id'], inplace=True)
train.sort_index(inplace=True)
train.reset_index(inplace=True)
train.drop(columns=['max_timestamp', 'random_timestamp'], inplace=True)
# %%
last100m = train[-100_000_000:]
interval = 2_500_000
mean_max_timestamp = []
target_means = []
for i in range(40):
    start = i * interval
    user_list = last100m[start:start+interval].user_id.unique()
    mean_max_timestamp.append(max_timestamp_u[['max_timestamp']].reindex(user_list).mean())
    temp = last100m[last100m.answered_correctly != -1]
    target_means.append(temp[start:start+interval].answered_correctly.mean())
mean_max_timestamp = cudf.concat(mean_max_timestamp).to_pandas()
target_means = pd.Series(target_means)

target_means.plot()
plt.show()
# %%
last10m = train[-10000000:]
interval = 1_000_000
mean_max_timestamp = []
target_means = []
for i in range(10):
    start = i * interval
    user_list = last10m[start:start+interval].user_id.unique()
    mean_max_timestamp.append(max_timestamp_u[['max_timestamp']].reindex(user_list).mean())
    temp = last10m[last10m.answered_correctly != -1]
    target_means.append(temp[start:start+interval].answered_correctly.mean())
mean_max_timestamp = cudf.concat(mean_max_timestamp).to_pandas()
target_means = pd.Series(target_means)

target_means.plot()
plt.show()
# %%
val_size = 2_500_000
for cv in range(5):
    valid = train[-val_size:]
    train = train[:-val_size]
    valid.to_parquet(f'cv{cv+1}_valid.parquet')
    train.to_parquet(f'cv{cv+1}_train.parquet')
# %%
