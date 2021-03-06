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
        assert X_predict.shape[1] == len(self.coefficients_),\
            'The number of features of X_test must be equal to X_train'

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def fit_gradient_descent(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0],\
            'The size of X must be equal to the size of y'

        def J(y, X_b, theta):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(y, X_b, theta):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            """

            2                         2
            - · (X_b·θ - y)^T · X_b = - · X_b^T · (X_b·θ - y)
            m                         m
            """
            return 2.0 * X_b.T.dot(X_b.dot(theta) - y) / len(X_b)

        def dJ_debug(y, X_b, theta, epsilon=0.01):
            res = np.empty(len(theta))
            for i in range(len(theta)):
                theta_1 = theta.copy()
                theta_2 = theta.copy()
                theta_1 += epsilon
                theta_2 -= epsilon
                res[i] = (J(y, X_b, theta_1) - J(y, X_b, theta_2)) / (w * epsilon)
            return res

        def gradient_descent(y, X_b, initial_theta, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = dJ(y, X_b, theta)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(J(y, X_b, last_theta) - J(y, X_b, theta)) < epsilon:
                    break
                i_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(y_train, X_b, initial_theta)

        self.intercept_ = self._theta[0]
        self.coefficients_ = self._theta[1:]
        return self

    def fit_stochastic_gradient_descent(self, X_train, y_train, n_iters=5, t0=5, t1=50):

        assert X_train.shape[0] == y_train.shape[0],\
            'The size of X_train must be equal to the size of y_train'
        assert n_iters >= 1, 'n_iter must bigger than one'

        def dJ_sgd(y_i, X_b_i, theta):
            return 2.0 * X_b_i.T.dot(X_b_i.dot(theta) - y_i)

        def stochastic_gradient_descent(y, X_b, initial_theta, n_iters, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t1 + t)

            theta = initial_theta
            size = len(X_b)

            # 至少每个标签都要跑一遍
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(size)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(size):
                    gradient = dJ_sgd(y_new[i], X_b_new[i], theta)
                    theta = theta - learning_rate(cur_iter * size + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = stochastic_gradient_descent(y_train, X_b, initial_theta, n_iters, t0, t1)

        self.intercept_ = self._theta[0]
        self.coefficients_ = self._theta[1:]
        return self

    def __repr__(self):
        return "LinearRegression(conffiecietns={}, intercept={}"\
            .format(self.coefficients_, self.intercept_)
