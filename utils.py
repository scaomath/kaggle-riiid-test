import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score



def get_seed(s):
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

SEED = 1127 
get_seed(SEED)


def roc_auc_compute_fn(y_targets, y_preds):
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred)