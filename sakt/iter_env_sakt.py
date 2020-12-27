#%%
import sys
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score
import torch

# HOME =  "/home/scao/Documents/kaggle-riiid-test/"
# DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
# MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(HOME) 
MODEL_DIR = HOME+'/model/'
DATA_DIR = HOME+'/data/'
sys.path.append(HOME)

from sakt import *
from utils import *
from iter_env import *

NUM_SKILLS = 13523
PRIVATE = False
DEBUG = False
VAL_BATCH_SIZE = 4096
SIMU_PRI_SIZE = 250_000
SIMU_PUB_SIZE = 25_000


if DEBUG:
    test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    test_df[:SIMU_PUB_SIZE].to_parquet(DATA_DIR+'test_pub_simu.parquet')

#%%
# if __name__ == "__main__":


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
print("\nLoading train for inference...")
train_df = pd.read_parquet(DATA_DIR+'cv5_train.parquet',
                                    columns=list(TRAIN_DTYPES.keys()))
train_df = train_df.astype(TRAIN_DTYPES)
print("Loaded train.")


print("\nLoading private simulated test set...")
if PRIVATE:
    test_df = pd.read_parquet(DATA_DIR+'cv5_valid.parquet')
    test_df = test_df[:SIMU_PRI_SIZE]
else:
    test_df = pd.read_parquet(DATA_DIR+'test_pub_simu.parquet')
    test_df = test_df[:SIMU_PUB_SIZE]
print("Loaded test .")


train_df = train_df[TRAIN_DTYPES.keys()]
train_df = train_df[train_df[CONTENT_TYPE_ID] == False].reset_index(drop = True)
train_group = train_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
    .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

example_test = pd.read_csv(DATA_DIR+'example_test.csv')
df_questions = pd.read_csv(DATA_DIR+'questions.csv')

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n\nUsing device: {device}')
# model_file = find_sakt_model()
# model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_head_10_embed_160_auc_0.7499.pt'
# model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_layer_2_head_8_embed_256_seq_150_auc_0.7577.pt'
# model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_layer_1_head_8_embed_128_seq_150_auc_0.7604.pt'

# model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_head_8_embed_128_seq_150_auc_0.7584.pt'
# model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_scaled_layer_1_head_8_embed_128_seq_150_auc_0.7603.pt'
model_file = '/home/scao/Documents/kaggle-riiid-test/model/sakt_layer_1_head_8_embed_128_seq_150_auc_0.7605.pt'

# tmp = model_file.split('_')
# structure = {'n_skills': 13523, 'n_embed': int(tmp[4]), 'n_head':int(tmp[2])}
model_name = model_file.split('/')[-1]

print(f'\nLoading {model_name}...\n')
# model, conf = load_sakt_model(model_file)
# conf_print = [(param, val) for (param, val) in conf.items()]
# print(f'\nLoaded {model_name} \n with {conf}')

MAX_SEQ = 150
NUM_EMBED = 128
NUM_HEADS = 8
SCALING = 1
NUM_LAYERS = 1
model = SAKTModel(n_skill=NUM_SKILLS, 
                max_seq=MAX_SEQ, 
                embed_dim=NUM_EMBED, 
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS)
        
model = model.to(device)
model.load_state_dict(torch.load(model_file, map_location=device))

model.eval()

#%%
iter_test = Iter_Valid(test_df, max_user=1000)

predicted = []
def set_predict(df):
    predicted.append(df)

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
        test_dataset = TestDataset(train_group, current_test, NUM_SKILLS)
        test_dataloader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, 
                                    shuffle=False, drop_last=False)

        output_all = []
        for _, batch in enumerate(test_dataloader):
            x = batch[0].to(device).long()
            target_id = batch[1].to(device).long()

            with torch.no_grad():
                output, _ = model(x, target_id)

            pred_probs = torch.sigmoid(SCALING*output[:, -1])
            
            # post-processing
            pred_probs[pred_probs<0.25] = \
                0.25 - torch.exp(-1/(0.25-pred_probs[pred_probs<0.25]))

            output_all.extend(pred_probs.reshape(-1).data.cpu().numpy())
        '''prediction code ends'''

        current_test[TARGET] = output_all
        set_predict(current_test.loc[:,[ROW_ID, TARGET]])
        pbar.set_description(f'Current batch test length: {len(current_test)}')
        pbar.update(len(current_test))

y_true = test_df[test_df.content_type_id == 0].answered_correctly
y_pred = pd.concat(predicted).answered_correctly
print(f'\nValidation auc with scaling {SCALING}:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))


# %%
