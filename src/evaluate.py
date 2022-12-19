import numpy as np
from src.log import log_scores


def mae(y_pred, y_target):
    return np.mean(np.abs(y_pred - y_target), axis=0)


def rmse(y_pred, y_target):
    return np.sqrt(np.mean(y_pred - y_target, axis=0) ** 2)


def evaluate(y_pred, y_target, output_set):
    mae_score = mae(y_pred, y_target)
    rmse_score = rmse(y_pred, y_target)
    
    log_scores(scores_dict={
        'MAE': mae_score,
        'RMSE': rmse_score
    }, output_set=output_set)
