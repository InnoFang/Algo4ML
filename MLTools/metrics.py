import numpy as np
from math import sqrt

def accuracy_rate(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0],\
        "The size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """
    :param y_true:    the real result
    :param y_predict: the predict result
    :return: MSE
           1
    MSE = --- · Σ (y_true - y_predict)^2
           m

    """
    assert len(y_true) == len(y_predict),\
        'The size of y_true must be equal to the size of y_predict'

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_meas_squared_error(y_true, y_predict):
    """
    :param y_true:    the real result
    :param y_predict: the predict result
    :return: RMSE
                 1
    RMSE = sqrt(--- · Σ (y_true - y_predict)^2) = sqrt(MSE)
                 m

    """

    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """
    :param y_true:    the real result
    :param y_predict: the predict result
    :return: MAE
           1
    MAE = --- · Σ| y_true - y_predict |
           m

    """
    assert len(y_true) == len(y_predict), \
        'The size of y_true must be equal to the size of y_predict'

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """
    :param y_true:    the real result
    :param y_predict: the predict result
    :return: R^2
                Σ (y_hat - y_i)^2            Σ (y_hat - y_i)^2  / m           MSE(y_hat, y)
    R^2 = 1 - ---------------------- = 1 - --------------------------- = 1 - ---------------
                Σ (y_mean - y_i)^2           Σ (y_mean - y_i)^2 / m              VAR(y)

    """
    assert len(y_true) == len(y_predict), \
        'The size of y_true must be equal to the size of y_predict'

    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
