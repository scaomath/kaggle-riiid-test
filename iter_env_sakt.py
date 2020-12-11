#%%
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import time
from sklearn.metrics import roc_auc_score
import torch

from sakt import *
from utils import *
from iter_env import *

DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'

PRIVATE = False
DEBUG = False
LAST_N = 100
VAL_BATCH_SIZE = 4096
SIMU_PUB_SIZE = 250_000

if DEBUG:
    test_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    test_df[:SIMU_PUB_SIZE].to_parquet(DATA_DIR+'test_pub_simu.parquet')



#%%

test_df = pd.read_pickle(DATA_DIR+'test_pub_simu.pickle')

df_questions = pd.read_csv(DATA_DIR+'questions.csv')
iter_test = Iter_Valid(test_df, max_user=1000)
predicted = []
def set_predict(df):
    predicted.append(df)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n\n Using device: {device} \n\n')
model_file = find_sakt_model()
model_name = model_file.split('/')[-1]
model = load_sakt_model(model_file)
print(f'\nLoaded {model_name}.')
model.eval()
print(model)

#%%
len_test = len(test_df)
with tqdm(total=len_test) as pbar:
    previous_test_df = None
    for (current_test, current_prediction_df) in iter_test:
        if previous_test_df is not None:
            answers = eval(current_test["prior_group_answers_correct"].iloc[0])
            responses = eval(current_test["prior_group_responses"].iloc[0])
            previous_test_df['answered_correctly'] = answers
            previous_test_df['user_answer'] = responses
            # your feature extraction and model training code here

        previous_test_df = current_test.copy()
        current_test = current_test[current_test.content_type_id == 0]
        
        '''prediction code here'''

        
        current_test_df = current_test[TRAIN_DTYPES.keys()]
        test_group = current_test_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
                    .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

        '''extract labels for verification purpose'''
        test_dataset = SAKTDataset(test_group, conf.NUM_SKILLS, subset="valid")
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size=conf.VAL_BATCH_SIZE, 
                                    shuffle=False, drop_last=False)

        '''No labels code goes here'''
        # test_dataset = TestDataset(test_group, conf.NUM_SKILLS, subset="valid")
        # test_dataloader = DataLoader(test_dataset, 
        #                             batch_size=conf.VAL_BATCH_SIZE, 
        #                             shuffle=False, 
        #                             num_workers=conf.WORKERS)

        output_all = []
        labels_all = []
        for _, batch in enumerate(test_dataloader):
            x = batch[0].to(device).long()
            target_id = batch[1].to(device).long()
            label = batch[2].to(device).float()

            with torch.no_grad():
                output, _ = model(x, target_id)

            pred_probs = torch.sigmoid(output[:, -1])
            output_all.extend(pred_probs.reshape(-1).data.cpu().numpy())
            labels_all.extend(label.reshape(-1).data.numpy())
        '''prediction code ends'''

        current_test['answered_correctly'] = output_all
        set_predict(current_test.loc[:,['row_id', 'answered_correctly']])
        pbar.update(len(current_test))

y_true = test_df[test_df.content_type_id == 0].answered_correctly
y_pred = pd.concat(predicted).answered_correctly
print('\nValidation auc:', roc_auc_score(y_true, y_pred))
print('# iterations:', len(predicted))



#%%
if __name__ == "__main__":
    if PRIVATE:
        test_df = pd.read_pickle(DATA_DIR+'cv2_valid.pickle')
    else:
        test_df = pd.read_pickle(DATA_DIR+'test_pub_simu.pickle')

    df_questions = pd.read_csv(DATA_DIR+'questions.csv')
    iter_test = Iter_Valid(test_df, max_user=1000)
    predicted = []
    def set_predict(df):
        predicted.append(df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n\n Using device: {device} \n\n')
    model_file = find_sakt_model()
    model_name = model_file.split('/')[-1]
    model = load_sakt_model(model_file)
    print(f'\nLoaded {model_name}.')
    model.eval()
    print(model)

    len_test = len(test_df)
    with tqdm(total=len_test) as pbar:
        previous_test_df = None
        for (current_test, current_prediction_df) in iter_test:
            if previous_test_df is not None:
                answers = eval(current_test["prior_group_answers_correct"].iloc[0])
                responses = eval(current_test["prior_group_responses"].iloc[0])
                previous_test_df['answered_correctly'] = answers
                previous_test_df['user_answer'] = responses
                # your feature extraction and model training code here

            previous_test_df = current_test.copy()
            current_test = current_test[current_test.content_type_id == 0]
            
            '''prediction code here'''

            
            current_test_df = current_test[TRAIN_DTYPES.keys()]
            test_group = current_test_df[[USER_ID, CONTENT_ID, TARGET]].groupby(USER_ID)\
                        .apply(lambda r: (r[CONTENT_ID].values, r[TARGET].values))

            '''
            extract labels for verification purpose
            '''
            test_dataset = SAKTDataset(test_group, conf.NUM_SKILLS, subset="valid")
            test_dataloader = DataLoader(test_dataset, 
                                        batch_size=conf.VAL_BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=conf.WORKERS)
            '''
            No labels code goes here'''
            # test_dataset = TestDataset(test_group, conf.NUM_SKILLS, subset="valid")
            # test_dataloader = DataLoader(test_dataset, 
            #                             batch_size=conf.VAL_BATCH_SIZE, 
            #                             shuffle=False, 
            #                             num_workers=conf.WORKERS)

            output_all = []
            labels_all = []
            for _, batch in enumerate(test_dataloader):
                x = batch[0].to(device).long()
                target_id = batch[1].to(device).long()
                label = batch[2].to(device).float()

                with torch.no_grad():
                    output, _ = model(x, target_id)

                pred_probs = torch.sigmoid(output[:, -1])
                output_all.extend(pred_probs.reshape(-1).data.cpu().numpy())
                labels_all.extend(label.reshape(-1).data.numpy())
            '''prediction code ends'''

            current_test['answered_correctly'] = output_all
            set_predict(current_test.loc[:,['row_id', 'answered_correctly']])
            pbar.update(len(current_test))

    y_true = test_df[test_df.content_type_id == 0].answered_correctly
    y_pred = pd.concat(predicted).answered_correctly
    print('\nValidation auc:', roc_auc_score(y_true, y_pred))
    print('# iterations:', len(predicted))

# %%
