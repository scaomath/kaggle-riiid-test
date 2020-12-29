import gc
import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random as rd
from contextlib import contextmanager
from time import time
from datetime import date
import math
import numpy as np
import pandas as pd
import psutil
import torch
import seaborn as sns
sns.set()
from sklearn.metrics import roc_auc_score


SEED = 1127 

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_system():
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Physical cores    :", psutil.cpu_count(logical=False))
    print("Total cores       :", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency    : {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency    : {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    print("="*40, "Memory Info", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total     : {get_size(svmem.total)}")
    print(f"Available : {get_size(svmem.available)}")
    print(f"Used      : {get_size(svmem.used)}")


    print("="*40, "Software Info", "="*40)
    print('Python     : ' + sys.version.split('\n')[0])
    print('Numpy      : ' + np.__version__)
    print('Pandas     : ' + pd.__version__)
    print('PyTorch    : ' + torch.__version__)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    if device.type == 'cuda':
        print("="*40, "GPU Info", "="*40)
        print(f'Device     : {device}')
        print(torch.cuda.get_device_name(0))
        print(f"{'Mem allocated': <15}: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(f"{'Mem cached': <15}: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
    

def get_seed(s):
    rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# @contextmanager
# def timer(title):
#     t0 = time()
#     yield
#     print("{} - done in {:.1f} seconds.\n".format(title, time() - t0))

class Colors:
    """Defining Color Codes to color the text displayed on terminal.
    """

    blue = "\033[94m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


@contextmanager
def timer(label: str) -> None:
    '''
    https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/203020#1111022
    '''
    """compute the time the code block takes to run.
    """
    p = psutil.Process(os.getpid())
    start = time()  # Setup - __enter__
    m0 = p.memory_info()[0] / 2. ** 30
    print(color(f"{label}: start at {start};"))
    print(color(f"RAM USAGE AT START: {m0:.2f} GB"))
    try:
        yield  # yield to body of `with` statement
    finally:  # Teardown - __exit__
        m1 = p.memory_info()[0] / 2. ** 30
        delta = m1 - m0
        sign = '+' if delta >= 0 else '-'
        delta = math.fabs(delta)
        end = time()
        print(color(f"{label}:  done at {end} ({end - start:.6f} secs elapsed);", color=Colors.red))
        print(color(f"RAM USAGE AT END:   {m1:.2f}GB ({sign}{delta:.2f}GB)", color=Colors.red))

@contextmanager
def trace(title: str):
    t0 = time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB ({sign}{delta:.3f}GB): {time() - t0:.2f}sec] {title} ", file=sys.stderr)


def get_date():
    today = date.today()
    return today.strftime("%b-%d-%Y")

def roc_auc_compute_fn(y_targets, y_preds):
    '''
    roc_auc func for torch tensors
    '''
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred)

def find_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for _file in files:
            if name in _file:
                result.append(os.path.join(root, _file))
    return result

def argmax(lst):
  return lst.index(max(lst))

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df


if __name__ == "__main__":
    get_system()
    