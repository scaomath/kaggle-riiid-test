#%%
from sakt.sakt import MAX_SEQ
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from utils import *


DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
MODEL_DIR = f'/home/scao/Documents/kaggle-riiid-test/model/'
PRIVATE = False
DEBUG = False
MAX_SEQ = 150
VAL_BATCH_SIZE = 4096
TEST_BATCH_SIZE = 4096
SIMU_PUB_SIZE = 25_000

#%%
class Iter_Valid(object):
    def __init__(self, df, max_user=1000):
        df = df.reset_index(drop=True)
        self.df = df
        self.user_answer = df['user_answer'].astype(str).values
        self.answered_correctly = df['answered_correctly'].astype(str).values
        df['prior_group_responses'] = "[]"
        df['prior_group_answers_correct'] = "[]"
        self.sample_df = df[df['content_type_id'] == 0][['row_id']]
        self.sample_df['answered_correctly'] = 0
        self.len = len(df)
        self.user_id = df.user_id.values
        self.task_container_id = df.task_container_id.values
        self.content_type_id = df.content_type_id.values
        self.max_user = max_user
        self.current = 0
        self.pre_user_answer_list = []
        self.pre_answered_correctly_list = []

    def __iter__(self):
        return self
    
    def fix_df(self, user_answer_list, answered_correctly_list, pre_start):
        df= self.df[pre_start:self.current].copy()
        sample_df = self.sample_df[pre_start:self.current].copy()
        df.loc[pre_start,'prior_group_responses'] = '[' + ",".join(self.pre_user_answer_list) + ']'
        df.loc[pre_start,'prior_group_answers_correct'] = '[' + ",".join(self.pre_answered_correctly_list) + ']'
        self.pre_user_answer_list = user_answer_list
        self.pre_answered_correctly_list = answered_correctly_list
        return df, sample_df

    def __next__(self):
        added_user = set()
        pre_start = self.current
        pre_added_user = -1
        pre_task_container_id = -1

        user_answer_list = []
        answered_correctly_list = []
        while self.current < self.len:
            crr_user_id = self.user_id[self.current]
            crr_task_container_id = self.task_container_id[self.current]
            crr_content_type_id = self.content_type_id[self.current]
            if crr_content_type_id == 1:
                # no more than one task_container_id of "questions" from any single user
                # so we only care for content_type_id == 0 to break loop
                user_answer_list.append(self.user_answer[self.current])
                answered_correctly_list.append(self.answered_correctly[self.current])
                self.current += 1
                continue
            if crr_user_id in added_user and ((crr_user_id != pre_added_user) or (crr_task_container_id != pre_task_container_id)):
                # known user(not prev user or differnt task container)
                return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            if len(added_user) == self.max_user:
                if  crr_user_id == pre_added_user and crr_task_container_id == pre_task_container_id:
                    user_answer_list.append(self.user_answer[self.current])
                    answered_correctly_list.append(self.answered_correctly[self.current])
                    self.current += 1
                    continue
                else:
                    return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            added_user.add(crr_user_id)
            pre_added_user = crr_user_id
            pre_task_container_id = crr_task_container_id
            user_answer_list.append(self.user_answer[self.current])
            answered_correctly_list.append(self.answered_correctly[self.current])
            self.current += 1
        if pre_start < self.current:
            return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
        else:
            raise StopIteration()

if DEBUG:
    test_df = pd.read_pickle(DATA_DIR+'cv2_valid.pickle')
    test_df[:SIMU_PUB_SIZE].to_pickle(DATA_DIR+'test_pub_simu.pickle')

#%%
if __name__ == "__main__":
    get_system()
    try:
        from transformer.transformer import *
    except:
        ModuleNotFoundError('transformer not found')

    print("Loading test set....")
    if PRIVATE:
        test_df = pd.read_pickle(DATA_DIR+'cv2_valid.pickle')
    else:
        test_df = pd.read_pickle(DATA_DIR+'test_pub_simu.pickle')
        test_df = test_df[:SIMU_PUB_SIZE]
    
    df_questions = pd.read_csv(DATA_DIR+'questions.csv')
    train_df = pd.read_parquet(DATA_DIR+'cv2_valid.parquet')
    train_df = preprocess(train_df, df_questions)
    d, user_id_to_idx_train = get_feats(train_df)
    print("Loaded test.")
    iter_test = Iter_Valid(test_df, max_user=1000)
    predicted = []
    def set_predict(df):
        predicted.append(df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n\n Using device: {device} \n\n')
    model_files = find_files(name='transformer',path=MODEL_DIR)
    model_file = model_files[0]
    # model_file = '/home/scao/Documents/kaggle-riiid-test/model/transformer_head_8_embed_512_seq_150_auc_0.7515.pt'
    conf = dict(ninp=512, 
                nhead=8, 
                nhid=128, 
                nlayers=2, 
                dropout=0.3)
    model = load_model(model_file, conf=conf)
    print(f'\nLoaded {model_file}.')
    model.eval()
    print(model)

    prev_test_df = None
    len_test = len(test_df)
    with tqdm(total=len_test) as pbar:
        for idx, (current_test, current_prediction_df) in enumerate(iter_test):
            
            '''
            concised iter_env
            '''
            if prev_test_df is not None:
                '''Making use of answers to previous questions'''
                answers = eval(current_test["prior_group_answers_correct"].iloc[0])
                responses = eval(current_test["prior_group_responses"].iloc[0])
                prev_test_df['answered_correctly'] = answers
                prev_test_df['user_answer'] = responses

                prev_test_df = prev_test_df[prev_test_df['content_type_id'] == False]
                prev_test_df = preprocess(prev_test_df, df_questions)
                d_prev, user_id_to_idx_prev = get_feats(prev_test_df)
                d = update_users(d_prev, d, user_id_to_idx_prev, user_id_to_idx_train)

            # no labels
            # d_test, user_id_to_idx = get_feats_test(current_test)
            # dataset_test = RiiidTest(d=d_test)
            # test_loader = DataLoader(dataset=dataset_test, batch_size=VAL_BATCH_SIZE, 
            #                             collate_fn=collate_fn_test, shuffle=False, drop_last=False)

            prev_test_df = current_test.copy()

            '''Labels for verification'''
            current_test = preprocess(current_test, df_questions)
            d_test, user_id_to_idx_test = get_feats_val(current_test, max_seq=MAX_SEQ)
            d_test = update_users(d, d_test, user_id_to_idx_train, user_id_to_idx_test, test_flag=True)
            dataset_test = RiiidVal(d=d_test)
            test_loader = DataLoader(dataset=dataset_test, 
                                        batch_size=TEST_BATCH_SIZE, 
                                        collate_fn=collate_fn_val, shuffle=False, drop_last=False)

            # the problem with current feature gen is that 
            # using groupby user_id sorts the user_id and makes it different from the 
            # test_df's order

            output_all = []
            labels_all = []
            for _, batch in enumerate(test_loader):
                content_id, _, part_id, prior_question_elapsed_time, mask, labels, pred_mask = batch
                target_id = batch[1].to(device).long()

                content_id = Variable(content_id.cuda())
                part_id = Variable(part_id.cuda())
                prior_question_elapsed_time = Variable(prior_question_elapsed_time.cuda())
                mask = Variable(mask.cuda())

                with torch.no_grad():
                    output = model(content_id, part_id, prior_question_elapsed_time, mask_padding= mask)

                pred_probs = torch.softmax(output[pred_mask], dim=1)
                output_all.extend(pred_probs[:,1].reshape(-1).data.cpu().numpy())
                labels_all.extend(labels[~mask].reshape(-1).data.numpy())
            '''prediction code ends'''

            current_test['answered_correctly'] = output_all
            set_predict(current_test.loc[:,['row_id', 'answered_correctly']])
            pbar.update(len(current_test))

    y_true = test_df[test_df.content_type_id == 0]['answered_correctly']
    y_pred = pd.concat(predicted)['answered_correctly']
    print('\nValidation auc:', roc_auc_score(y_true, y_pred))
    print('# iterations:', len(predicted))

# %%
