import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0],\
        "The size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0,\
        "test_ration must be between 0.0 and 1.0"

    if seed:
        np.random.seed(seed)

    shuffle_indexes = np.random.permutation(len(X))

    test_size = len(X) * test_ratio
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, y_train, X_test, y_test