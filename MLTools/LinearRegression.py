import numpy as np
from .metrics import r2_score


class LinearRegression:

    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """

        θ = ((X_b)T·X_b)^-1 · (X_b)T · y

        :param X_train:
        :param y_train:
        :return: self
        """
        assert X_train.shape[0] == y_train.shape[0],\
            'The size of X_train must be equal to the size of y_train'
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coefficients_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        assert self.coefficients_ is not None and self.intercept_ is not None,\
            'Please fit_normal before predict'
        assert X_predict.shape[0] == len(self.coefficients_),\
            'The number of features of X_test must be equal to X_train'

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression(conffiecietns={}, intercept={}"\
            .format(self.coefficients_, self.intercept_)