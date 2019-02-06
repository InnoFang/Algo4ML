import numpy as np
from .metrics import r2_score


class SimpleLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, 'The dimension of X_train must be 1'
        assert len(x_train) == len(y_train),\
            'The length of X_train must be equal to the size of y_train'

        """
             Σ(x_i - x_mean)(y_i - y_mean)
        a = -----------------------------
             Σ(x_i - x_mean)^2
             
        b = y_mean - a·x_mean
        
        """

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # molecule = 0.0
        # denominator = 0.0
        #
        # for x_i, y_i in zip(x_train, y_train):
        #     molecule += (x_i - x_mean) * (y_i - y_mean)
        #     denominator += (x_i - x_mean) ** 2

        # Vectorization
        molecule = (x_train - x_mean).dot(y_train - y_mean)
        denominator = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = molecule / denominator
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, 'The dimension of X_predict must be 1'
        assert self.a_ is not None and self.b_ is not None,\
            'Please fit before transform'
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return 'SimpleLinearRegression41D(a={}, b={})'.format(self.a_, self.b_)
