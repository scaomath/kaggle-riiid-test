#%%
import os
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import datatable as dt
import lightgbm as lgb
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from jupyterthemes import jtplot
jtplot.style(theme='onedork', context='notebook', ticks=True, grid=False)

import random, math
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pickle
import zipfile

import gc
import warnings 
warnings.filterwarnings('ignore')

HOME = os.path.abspath(os.path.join('.', os.pardir))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(HOME, '\n', CURRENT_DIR, '\n')
MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
sys.path.append(CURRENT_DIR) 
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
#%%
DEBUG = True

if DEBUG:
    NROWS_TRAIN = 100_000
else:
    NROWS_TRAIN = 20_000_000

question_metadata_dir = DATA_DIR+'question_metadata.csv'
lesson_metadata_dir = DATA_DIR+'lesson_metadata.csv'
pickle_dir= DATA_DIR+'stage.pickle'
# model_dir = DATA_DIR+'classifier.model'

# %% trueskill
'''
https://www.kaggle.com/zyy2016/0-780-unoptimized-lgbm-interesting-features
https://www.kaggle.com/ahmedsalah142/interesting-features-copied-0-78-for-training
'''
import trueskill

print(f"{str(datetime.datetime.now())} Import Completed")
env = trueskill.TrueSkill(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164, draw_probability=0)
env.make_as_global()
# %%

train_df = pd.read_csv(DATA_DIR+'train.csv', 
                        low_memory=False,  nrows=NROWS_TRAIN ,
                       dtype={'row_id': 'int64', 
                            'timestamp': 'int64', 
                            'user_id': 'int32', 
                            'content_id': 'int16', 
                            'content_type_id': 'int8',
                                'task_container_id': 'int16', 
                                'user_answer': 'int8', 
                                'answered_correctly': 'int8', 
                                'prior_question_elapsed_time': 'float32', 
                                'prior_question_had_explanation': 'boolean'
                                }
                      )
# %%
def win_probability(team1, team2):
    '''
    Calculate the win possibility based on two Trueskill objects
    :param team1:User TrueSkill Object
    :param team2:Question Trueskill Object
    :return: Winning Prob
    '''
    delta_mu = team1.mu - team2.mu
    sum_sigma = sum([team1.sigma ** 2, team2.sigma ** 2])
    size = 2
    denom = math.sqrt(size * (0.05 * 0.05) + sum_sigma)
    ts = trueskill.global_env()
    
    return [ts.cdf(elem / denom) for elem in delta_mu]
def win_probability_orig(team1, team2):
    '''
    Calculate the win possibility based on two Trueskill objects
    :param team1:User TrueSkill Object
    :param team2:Question Trueskill Object
    :return: Winning Prob
    '''
    delta_mu = team1.mu - team2.mu
    sum_sigma = sum([team1.sigma ** 2, team2.sigma ** 2])
    size = 2
    denom = math.sqrt(size * (0.05 * 0.05) + sum_sigma)
    ts = trueskill.global_env()
    
    return ts.cdf(delta_mu / denom) 
# %%
class user:
    '''
    User Class
    __slots__ = ['question_answered_num', 'question_answered_num_agg_field', 'question_answered_mean_accuracy',
    'question_answered_mean_accuracy_agg_field','question_answered_mean_difficulty_weighted_accuracy',
        'question_answered_mean_difficulty_weighted_accuracy_agg_field','max_solved_difficulty',
        'max_solved_difficulty_agg_field','min_wrong_difficulty','min_wrong_difficulty_agg_field',
        'lessons_overall','lessons_overall_agg_field','session_time','since_last_session_time',
        '_mmr_object','_mmr_object_agg_field','_most_liked_guess','_last_session_start_time','_first_action_time',
        '_question_num_dict','_first_processed_flag']
    '''

    def __init__(self):
        '''
        Init User Class
        :param None
        :return: None
        '''

        # Counting 
        self.question_answered_num = 0  
        self.question_answered_num_agg_field = [0] * 7 

        # Correct Rate
        self.question_answered_mean_accuracy = 0  
        self.question_answered_mean_accuracy_agg_field = [0] * 7  
        self.question_answered_mean_difficulty_weighted_accuracy = 0  
        self.question_answered_mean_difficulty_weighted_accuracy_agg_field = [0] * 7  
        # Min/Max stuff
        self.max_solved_difficulty = 1
        self.max_solved_difficulty_agg_field = [1] * 7
        self.min_wrong_difficulty = 0 
        self.min_wrong_difficulty_agg_field = [0] * 7  

        # Lessons stuff
        self.lessons_overall = 0 
        self.lessons_overall_agg_field = [0] * 7  

        # Session timing 
        self.session_time = 0  
        self.since_last_session_time = 0  

        # Features need some processing
        self._mmr_object = trueskill.setup(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164,
                                           draw_probability=0).Rating() 
        self._mmr_object_agg_field = [trueskill.setup(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164,
                                                      draw_probability=0).Rating()] * 7
        self._most_liked_guess = [0] * 4 
        self._last_session_start_time = 0  
        self._first_action_time = 0  
        self._question_num_dict = {}  
        self._first_processed_flag = False  

    def update_user(self, data: pd.DataFrame):
        '''
        Update user with one row of DataFrame
        :param data: pandas DataFrame
        :return: None
        '''
        _temp = None

        # Judging whether user are watching courses
        if data['content_type_id'] == 0:
            # Content Type = 0,means User are answering Questions.

            # Counting Part
            self.question_answered_num = self.question_answered_num + 1
            question_field = int(data['content_field'])
            self.question_answered_num_agg_field[question_field - 1] = int(self.question_answered_num_agg_field[
                                                                               question_field - 1]) + 1

            # Average Correct Rate
            if data['answered_correctly'] == 1:
                self.question_answered_mean_accuracy = \
                    (self.question_answered_mean_accuracy * (
                            self.question_answered_num - 1) + 1) / self.question_answered_num

                self.question_answered_mean_accuracy_agg_field[question_field - 1] = \
                    (self.question_answered_mean_accuracy_agg_field[question_field - 1] * (
                            self.question_answered_num_agg_field[question_field - 1] - 1) + 1) \
                    / self.question_answered_num_agg_field[question_field - 1]

                self.question_answered_mean_difficulty_weighted_accuracy = \
                    (self.question_answered_mean_difficulty_weighted_accuracy * (self.question_answered_num - 1) + (
                            1 - data['mean_question_accuracy']) * 3) \
                    / self.question_answered_num

                self.question_answered_mean_difficulty_weighted_accuracy_agg_field[question_field - 1] = \
                    (self.question_answered_mean_difficulty_weighted_accuracy_agg_field[question_field - 1] * (
                            self.question_answered_num_agg_field[question_field - 1] - 1) + (
                             1 - data['mean_question_accuracy']) * 3) \
                    / self.question_answered_num_agg_field[question_field - 1]


            else:
                self.question_answered_mean_accuracy = \
                    (self.question_answered_mean_accuracy * (
                            self.question_answered_num - 1)) / self.question_answered_num

                self.question_answered_mean_accuracy_agg_field[question_field - 1] = \
                    (self.question_answered_mean_accuracy_agg_field[question_field - 1] * (
                            self.question_answered_num_agg_field[question_field - 1] - 1)) / \
                    self.question_answered_num_agg_field[question_field - 1]

                self.question_answered_mean_difficulty_weighted_accuracy = \
                    (self.question_answered_mean_difficulty_weighted_accuracy * (self.question_answered_num - 1)) \
                    / self.question_answered_num

                self.question_answered_mean_difficulty_weighted_accuracy_agg_field[question_field - 1] = \
                    (self.question_answered_mean_difficulty_weighted_accuracy_agg_field[question_field - 1] * (
                            self.question_answered_num_agg_field[question_field - 1] - 1)) \
                    / self.question_answered_num_agg_field[question_field - 1]

            # Min/Max Part

            if data['answered_correctly'] == 1:
                if data['mean_question_accuracy'] < self.max_solved_difficulty:
                    self.max_solved_difficulty = data['mean_question_accuracy']
                if data['mean_question_accuracy'] < self.max_solved_difficulty_agg_field[question_field - 1]:
                    self.max_solved_difficulty_agg_field[question_field - 1] = data['mean_question_accuracy']
            else:
                if data['mean_question_accuracy'] > self.min_wrong_difficulty:
                    self.min_wrong_difficulty = data['mean_question_accuracy']
                if data['mean_question_accuracy'] > self.min_wrong_difficulty_agg_field[question_field - 1]:
                    self.min_wrong_difficulty_agg_field[question_field - 1] = data['mean_question_accuracy']

            # Guessing Part
            if data['answered_correctly'] == 0:
                self._most_liked_guess[int(data['user_answer'])] = self._most_liked_guess[
                                                                       int(data['user_answer'])] + 1

            # Session Timing part
            if self._first_action_time == 0:
                self._first_action_time = data['timestamp']
                self._last_session_start_time = data['timestamp']
            else:
                if data['timestamp'] - self._last_session_start_time >= 7200 * 1000:
                    self.since_last_session_time = (data[
                                                        'timestamp'] - self._last_session_start_time) / 1000 / 3600
                    self._last_session_start_time = data['timestamp']
                    self.session_time = 0
                else:
                    self.session_time = (data['timestamp'] - self._last_session_start_time) / 1000 / 60

            # Answer history part
            if str(data['content_id']) in self._question_num_dict:
                self._question_num_dict[str(data['content_id'])] = self._question_num_dict[str(data['content_id'])] + 1
            else:
                self._question_num_dict[str(data['content_id'])] = 1

            # Trueskill part
            if data['answered_correctly'] == 1:
                self._mmr_object, _temp = \
                    trueskill.rate_1vs1(self._mmr_object,
                                        trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486,
                                                        beta=0.05, tau=0.00164, draw_probability=0).Rating())
                self._mmr_object_agg_field[question_field - 1], _temp = \
                    trueskill.rate_1vs1(self._mmr_object_agg_field[question_field - 1],
                                        trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486,
                                                        beta=0.05,
                                                        tau=0.00164, draw_probability=0).Rating())
            else:
                _temp, self._mmr_object = \
                    trueskill.rate_1vs1(trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486,
                                                        beta=0.05, tau=0.00164, draw_probability=0).Rating(),
                                        self._mmr_object)

                _temp, self._mmr_object_agg_field[question_field - 1] = \
                    trueskill.rate_1vs1(trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486,
                                                        beta=0.05,
                                                        tau=0.00164, draw_probability=0).Rating(),
                                        self._mmr_object_agg_field[question_field - 1])



        else:
            # Content Type !=0,User are watching a  lecture

            self.lessons_overall = self.lessons_overall + 1
            lesson_field = int(data['content_field'])
            self.lessons_overall_agg_field[lesson_field - 1] = self.lessons_overall_agg_field[lesson_field - 1] + 1
    def get_user_dict(self):
        return {'question_answered_num':self.question_answered_num,
               'question_answered_mean_accuracy':self.question_answered_mean_accuracy,
               'max_solved_difficulty':self.max_solved_difficulty,
                'min_wrong_difficulty':self.min_wrong_difficulty,
                'lessons_overall':self.lessons_overall,
                'session_time':self.session_time,
                'time_to_last_session':self.since_last_session_time,
                'mmr_overall':self._mmr_object.mu,
                'mmr_confidence':self._mmr_object.sigma,
                'question_answered_mean_difficulty_weighted_accuracy':self.question_answered_mean_difficulty_weighted_accuracy,
                
               }, {'question_answered_num_agg_field':self.question_answered_num_agg_field,
                'question_answered_mean_accuracy_agg_field':self.question_answered_mean_accuracy_agg_field,
                 'question_answered_mean_difficulty_weighted_accuracy_agg_field':self.question_answered_mean_difficulty_weighted_accuracy_agg_field,
                 'max_solved_difficulty_agg_field':self.max_solved_difficulty_agg_field,
                 'min_wrong_difficulty_agg_field':self.min_wrong_difficulty_agg_field,
                 'lessons_overall_agg_field':self.lessons_overall_agg_field,
                 'mmr_overall_agg_field':self._mmr_object_agg_field

                }
    def get_mmr_obj(self):
        return self._mmr_object
    def get_mmr_object_agg_field(self):
        return self._mmr_object_agg_field
    def get_question_num_dict(self):
        return self._question_num_dict
    def get_most_liked_guess(self):
        return self._most_liked_guess
    def process_output(self, data):
        '''
        
         Output data according to user's existing attributes
        :param data: One row of dataset
        :return: output_dict dict data for training/predicting
        '''
        output_dict = {}

        # Counting Part
        output_dict['question_answered_num'] = self.question_answered_num
        output_dict['question_answered_num_agg_field'] = self.question_answered_num_agg_field[
            int(data['content_field']) - 1]

        # Average Correct Rate
        output_dict['question_answered_mean_accuracy'] = self.question_answered_mean_accuracy

        output_dict['question_answered_mean_accuracy_agg_field'] = self.question_answered_mean_accuracy_agg_field[
            int(data['content_field']) - 1]
        output_dict[
            'question_answered_mean_difficulty_weighted_accuracy'] = self.question_answered_mean_difficulty_weighted_accuracy
        output_dict['question_answered_mean_difficulty_weighted_accuracy_agg_field'] = \
            self.question_answered_mean_difficulty_weighted_accuracy_agg_field[int(data['content_field']) - 1]

        #  Min/Max Part

        output_dict['max_solved_difficulty'] = self.max_solved_difficulty
        output_dict['max_solved_difficulty_agg_field'] = self.max_solved_difficulty_agg_field[
            int(data['content_field']) - 1]
        output_dict['min_wrong_difficulty'] = self.min_wrong_difficulty
        output_dict['min_wrong_difficulty_agg_field'] = self.min_wrong_difficulty_agg_field[
            int(data['content_field']) - 1]

        # Lesson Learning part
        output_dict['lessons_overall'] = self.lessons_overall
        output_dict['lessons_overall_agg_field'] = self.lessons_overall_agg_field[int(data['content_field']) - 1]
        if output_dict['lessons_overall_agg_field'] > 0:
            output_dict['field_learnt'] = 1
        else:
            output_dict['field_learnt'] = 0
        # Session Timing part
        output_dict['session_time'] = self.session_time
        output_dict['time_to_last_session'] = self.since_last_session_time

        output_dict['task_id'] = data['task_container_id']
        output_dict['prior_time'] = data['prior_question_elapsed_time']
        # Question Statics part
        output_dict['mean_question_accuracy'] = data['mean_question_accuracy']
        output_dict['std_question_accuracy'] = data['std_accuracy']
        output_dict['question_id'] = data['content_id']
        # TrueSkill part
        output_dict['mmr_overall'] = self._mmr_object.mu
        output_dict['mmr_overall_agg_field'] = self._mmr_object_agg_field[int(data['content_field']) - 1].mu
        output_dict['mmr_confidence'] = self._mmr_object.sigma

        output_dict['mmr_overall_agg_field'] = self._mmr_object_agg_field[int(data['content_field']) - 1].sigma
        output_dict['mmr_win_prob'] = win_probability(self._mmr_object,
                                                      trueskill.setup(mu=1 - data['mean_question_accuracy'],
                                                                      sigma=0.164486,
                                                                      beta=0.05, tau=0.00164,
                                                                      draw_probability=0).Rating())
        output_dict['mmr_win_prob_agg_field'] = win_probability(
            self._mmr_object_agg_field[int(data['content_field']) - 1],
            trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486, beta=0.05,
                            tau=0.00164, draw_probability=0).Rating())
        output_dict['user_id'] = data['user_id']
        output_dict['tag_1'] = data['tag_1']
        output_dict['tag_2'] = data['tag_2']

        output_dict['tags_encoded'] = data['tags_encoded']
        # Other features

        if not pd.isna(['prior_question_had_explanation']):
            output_dict['previous_explained'] = data['prior_question_had_explanation']
        else:
            output_dict['previous_explained'] = False

        if str(data['content_id']) in self._question_num_dict:
            output_dict['question_seen'] = 1
        else:
            output_dict['question_seen'] = 0

        # Guessing part
        max_choice = 0
        max_choice_num = 0
        i = 0
        for item in self._most_liked_guess:
            if item > max_choice_num:
                max_choice_num = item
                max_choice = i
            i = i + 1

        if output_dict['mmr_win_prob'] <= 0.4:
            if max_choice == data['correct_answer']:
                output_dict['most_liked_guess_correct'] = True
            else:
                output_dict['most_liked_guess_correct'] = False
        else:
            output_dict['most_liked_guess_correct'] = True

        # Target
        #output_dict['answered_correctly'] = data['answered_correctly']

        return output_dict
# %%
# Import Metadata
question_metadata = pd.read_csv(question_metadata_dir)
lesson_metadata = pd.read_csv(lesson_metadata_dir)
print(f"{str(datetime.datetime.now())} Metadata Imported")
#Indexing Metadata
question_metadata = question_metadata.set_index(keys=['content_id'])
lesson_metadata = lesson_metadata.set_index(keys=['content_id'])
print(f"{str(datetime.datetime.now())} Metadata Indexed")
# %%
with open(pickle_dir, 'rb') as f:
    user_pickle = pickle.load(f)

print(f"{str(datetime.datetime.now())} Pickle Object Imported")
# %%
for user_id, user_info in tqdm(user_pickle.items()):
    user_pickle[user_id]._mmr_object = trueskill.setup(mu=user_pickle[user_id]._mmr_object[0],
                                                       sigma=user_pickle[user_id]._mmr_object[1],
                                                       beta=0.05, tau=0.00164,
                                                       draw_probability=0).Rating()
    for i in range(0, 7):
        # 1+1
        user_pickle[user_id]._mmr_object_agg_field[i] = trueskill.setup(mu=user_pickle[user_id]._mmr_object_agg_field[i][0],
                                                       sigma=user_pickle[user_id]._mmr_object_agg_field[i][1],
                                                       beta=0.05, tau=0.00164,
                                                       draw_probability=0).Rating()

print(f"{str(datetime.datetime.now())} Pickle Trueskill Rebuilt")
# %%
rows_accum = 0 #Row Counter
first_submission = True 
model_prd = [0]
true_value = []
last_df = pd.DataFrame()
# %%

# %%
def process_data(df):    
    result_df = pd.DataFrame()
    df['answered_correctly'] = 0.6524
    st = float(time.time())
    # Merging and Concating
#     try:
    sub_1 = df[df['content_type_id'] == False]
    sub_2 = df[df['content_type_id'] == True]
    del df
    sub_1 = sub_1.merge(question_metadata, on="content_id", how="left")
    sub_2 = sub_2.merge(lesson_metadata, on="content_id", how="left")
    df = pd.DataFrame()
    df = pd.concat([sub_1,sub_2])
    return df

# %%
df = process_data(train_df.copy())
# %%
users_dict = {}
for user in tqdm(user_pickle):
    users_dict[user] = user_pickle[user].get_user_dict()[0]

users_df = pd.DataFrame(users_dict)
users_df = users_df.T.reset_index()
df = df.merge(users_df,left_on='user_id',right_on='index')
# %%
agg_dict = {}
for user in user_pickle:
    agg_dict[user] = user_pickle[user].get_user_dict()[1]
    
agg_df = pd.DataFrame(agg_dict)
agg_df = agg_df.T.reset_index()
df = df.merge(agg_df,left_on='user_id',right_on='index')

# %%
for col in agg_df.columns:
    if col != 'index':
        df_np = df[['content_field',col]].values 
        df[col] = [elem [ind-1]for ind,elem in zip(df_np[:,0],df_np[:,1])]
df['field_learnt'] = (df['lessons_overall_agg_field'] >0).astype('int')
df['mmr_overall_agg_field']=[elem.sigma for elem in df['mmr_overall_agg_field'].values]
# %%
df['mmr_win_prob'] = 0
df['mmr_win_prob_agg_field'] = 0
df['most_liked_guess_correct'] = 1

k=0

users = df.user_id.unique()
l = len(users)
for user in tqdm(users):
    k+=1
    mmr_object = user_pickle[user].get_mmr_obj()
    user_df = df[df['user_id']==user]
    # mmr_win_prob
    df.loc[user_df.index,'mmr_win_prob'] = win_probability(mmr_object,
                                                      trueskill.setup(mu=1 - np.array(user_df['mean_question_accuracy']),
                                                                      sigma=0.164486,
                                                                      beta=0.05, tau=0.00164,draw_probability=0).Rating())
    # most_liked_guess_correct
    most_liked_guess = user_pickle[user].get_most_liked_guess()
    i = np.argmax(most_liked_guess)
    
    user_df_filtered = user_df[user_df['mmr_win_prob'] <= 0.4]
    user_df_filtered = user_df_filtered[user_df_filtered['correct_answer']==i]
    df.loc[user_df_filtered.index,'most_liked_guess_correct'] = 0
                                                                      
    # mmr_win_prob_agg_field and question_seen
    mmr_object_agg_field = user_pickle[user].get_mmr_object_agg_field()
    user_df_content = user_df['content_field']
    a = user_df_content.values
    mmr_object_agg_field_arr = [mmr_object_agg_field[i-1] for i in a]

    user_df = df[df['user_id']==user]
    df.loc[user_df.index,'mmr_win_prob_agg_field'] =  [win_probability_orig(b,trueskill.setup(mu=1 - a,
                                      sigma=0.164486,
                                      beta=0.05, tau=0.00164,
                                    draw_probability=0).Rating())for a,b in zip(user_df['mean_question_accuracy'],mmr_object_agg_field_arr)]
    
    question_num_dict = user_pickle[user].get_question_num_dict()
    df.loc[user_df.index,'question_seen'] = (user_df['content_id'].isin(question_num_dict) ).astype('int')
# %%
df['answered_correctly']=train_df['answered_correctly']
# %%
