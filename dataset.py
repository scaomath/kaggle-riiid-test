import datatable as dt
from time import time
'''
Download Kaggle data using kaggle API:
kaggle competitions download riiid-test-answer-prediction --path ./data
'''
DATA_DIR = '/home/scao/Documents/kaggle-riiid-test/data/'
start = time()
data = dt.fread(DATA_DIR+"train.csv")
print(f"Readding train.csv in {time()-start} seconds")