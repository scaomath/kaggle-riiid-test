#%%
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
sys.path.append(HOME)
from sakt import *
from utils import *
from iter_env import *

PRIVATE = False
DEBUG = False
LAST_N = 100
VAL_BATCH_SIZE = 51_200
SIMU_PUB_SIZE = 25_000
MAX_SEQ = 100

if DEBUG:
    test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    test_df[:SIMU_PUB_SIZE].to_parquet(DATA_DIR+'test_pub_simu.parquet')

#%%
# if __name__ == "__main__":
if PRIVATE:
    test_df = pd.read_pickle(DATA_DIR+'cv2_valid.pickle')
else:
    test_df = pd.read_parquet(DATA_DIR+'test_pub_simu.parquet')


train_df = pd.read_parquet(DATA_DIR+'cv2_train.parquet')
train_df = train_df[TRAIN_DTYPES.keys()]
train_df = train_df[train_df[CONTENT_TYPE_ID] == False].reset_index(drop = True)
train_group = train_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

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
model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_head_10_embed_160_auc_0.7499.pt'
tmp = model_file.split('_')
structure = {'n_skills': 13523, 'n_embed': int(tmp[4]), 'n_head':int(tmp[2])}
model_name = model_file.split('/')[-1]
print(f'\nLoading {model_name}...\n')
model = load_sakt_model(model_file, structure=structure)
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
            prev_group = prev_test_df[[USER_ID, CONTENT_ID, TARGET]].groupby('user_id')\
                .apply(lambda r: (
                r[CONTENT_ID].values,
                r[TARGET].values))
            for prev_user_id in prev_group.index:
                prev_group_content = prev_group[prev_user_id][0]
                prev_group_ac = prev_group[prev_user_id][1]
                if prev_user_id in train_group.index:
                    train_group[prev_user_id] = (np.append(train_group[prev_user_id][0],
                                                        prev_group_content), 
                                                np.append(train_group[prev_user_id][1],
                                                        prev_group_ac))

                else:
                    train_group[prev_user_id] = (prev_group_content,prev_group_ac)
                if len(train_group[prev_user_id][0]) > MAX_SEQ:
                    new_group_content = train_group[prev_user_id][0][-MAX_SEQ:]
                    new_group_ac = train_group[prev_user_id][1][-MAX_SEQ:]
                    train_group[prev_user_id] = (new_group_content,new_group_ac)


        previous_test_df = current_test.copy()
        current_test = current_test[current_test[CONTENT_TYPE_ID] == 0]
        
        '''prediction code here'''
        test_dataset = TestDataset(train_group, current_test, 
                                    conf.NUM_SKILLS)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.VAL_BATCH_SIZE, 
                                    shuffle=False, drop_last=False)

        output_all = []
        for _, batch in enumerate(test_dataloader):
            x = batch[0].to(device).long()
            target_id = batch[1].to(device).long()

            with torch.no_grad():
                output, _ = model(x, target_id)

            pred_probs = torch.sigmoid(output[:, -1])
            output_all.extend(pred_probs.reshape(-1).data.cpu().numpy())
        '''prediction code ends'''

        current_test[TARGET] = output_all
        set_predict(current_test.loc[:,[ROW_ID, TARGET]])
        pbar.desc(f'Current batch test length: {len(current_test)}')
        pbar.update(len(current_test))

y_true = test_df[test_df.content_type_id == 0].answered_correctly
y_pred = pd.concat(predicted).answered_correctly
print('\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))


# %%
