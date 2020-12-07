#%%
import pandas as pd
import dask.dataframe as dd
import datatable as dt
from time import time
# %%
'''
Download Kaggle data using kaggle API:
kaggle competitions download riiid-test-answer-prediction --path ./data
'''
dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "boolean"
}
start = time()
data = dt.fread("./data/train.csv")
print(f"Readding train.csv in {time()-start} seconds")
# %%
df_train = data.to_pandas()
