import os
import numpy as np
from sklearn.metrics import roc_auc_score


def mse(label, pred):
    return np.mean((pred - label)**2)

def auc(label, pred):
    return roc_auc_score(label, pred)
