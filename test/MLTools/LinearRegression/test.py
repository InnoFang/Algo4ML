from MLTools.SimpleLinearRegression import SimpleLinearRegression
from MLTools.metrics import mean_squared_error, root_meas_squared_error, mean_absolute_error
from MLTools.model_selection import train_test_split
from MLTools.LinearRegression import LinearRegression
from MLTools.preprocessing import StandardScaler
import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class LinearRegressionTest(unittest.TestCase):

    def test_SimpleLinearRegression(self):
        size = 100
        x = np.random.random(size=size)
        y = 3.0 * x + 5.0 + np.random.normal(size=size)

        regression = SimpleLinearRegression()
        print(regression.fit(x, y))
        y_predict = regression.predict(x)

        plt.scatter(x, y)
        plt.plot(x, y_predict, color='r')
        plt.show()

    def test_MSE_RMSE_MAE_R2(self):
        boston = datasets.load_boston()
        print(boston['feature_names'])
        X = boston['data'][:, 5]  # Get the `RM`(room) data
        y = boston['target']

        # clean data
        X = X[y < 50.0]
        y = y[y < 50.0]

        x_train, y_train, x_test, y_test = train_test_split(X, y)

        regression = SimpleLinearRegression()
        print(regression.fit(x_train, y_train))
        y_predict = regression.predict(x_test)

        print(mean_squared_error(y_test, y_predict))
        print(root_meas_squared_error(y_test, y_predict))
        print(mean_absolute_error(y_test, y_predict))
        print(regression.score(x_test, y_test))

        plt.scatter(x_train, y_train)
        plt.plot(x_train, regression.predict(x_train), color='r')
        plt.show()

    def test_LinearRegerssion(self):
        boston = datasets.load_boston()
        X = boston['data']
        y = boston['target']

        X = X[y < 50.0]
        y = y[y < 50.0]

        X_train, y_train, X_test, y_test = train_test_split(X, y)

        regression = LinearRegression()
        print(regression.fit_normal(X_train, y_train))
        print('score={}'.format(regression.score(X_test, y_test)))

    def test_gradient_descent(self):
        boston = datasets.load_boston()
        X = boston['data']
        y = boston['target']

        X = X[y < 50.0]
        y = y[y < 50.0]

        X_train, y_train, X_test, y_test = train_test_split(X, y)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_standard = scaler.transform(X_train)
        X_test_standard = scaler.transform(X_test)

        lin_reg = LinearRegression()
        lin_reg.fit_gradient_descent(X_train_standard, y_train)
        print(lin_reg.score(X_test_standard, y_test))
