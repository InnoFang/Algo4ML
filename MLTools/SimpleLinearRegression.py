import numpy as np


class SimpleLinearRegression41D:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, X_train, y_train):
        assert X_train.ndim == 1, 'The dimension of X_train must be 1'
        assert len(X_train) == len(y_train),\
            'The length of X_train must be equal to the size of y_train'

        """
             Σ(x_i - x_hat)(y_i - y_hat)
        a = -----------------------------
             Σ(x_i - x_hat)^2
             
        b = y_hat - a·x_hat
        
        """

        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        molecule = 0.0
        denominator = 0.0

        for x_i, y_i in zip(X_train, y_train):
            molecule += (x_i - x_hat) * (y_i - y_hat)
            denominator += (x_i - x_hat) ** 2

        self.a_ = molecule / denominator
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def transform(self, X_predict):
        assert X_predict.ndim == 1, 'The dimension of X_predict must be 1'
        return np.array([self._predict(x) for x in X_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return 'SimpleLinearRegression41D(a={}, b={})'.format(self.a_, self.b_)
