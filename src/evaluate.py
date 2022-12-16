import numpy as np


def evaluate(y_pred, y_target):
    mae = np.mean(np.abs(y_pred - y_target))
    rmse = np.sqrt(np.mean(y_pred - y_target) ** 2)
