import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:

    def __init__(self, k_neighbours):
        assert k_neighbours >= 1,\
            "K neighbours must be bigger than 1"

        self._X_train = None
        self._y_train = None
        self._k_neighbours = k_neighbours

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0],\
            "The size of X_train must be equal to the size of y_train"
        assert X_train.shape[0] >= self._k_neighbours,\
            "The size of X_train must be at least the `k_neighbours` you set"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None,\
            "The X_train and y_train cannot be None, please fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1],\
            "The number of feature of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1],\
            "The number of feature of x must be equal to X_train"

        distances = [sqrt(sum(x - x_train) ** 2) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK = [self._y_train[i] for i in nearest[: self._k_neighbours]]
        votes = Counter(topK)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k_neighbours={})".format(self._k_neighbours)
