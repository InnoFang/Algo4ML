from MLTools.SimpleLinearRegression import SimpleLinearRegression41D
from MLTools.metrics import mean_squared_error, root_meas_squared_error, mean_absolute_error
from MLTools.model_selection import train_test_split
import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class LinearRegressionTest(unittest.TestCase):

    def test_SimpleLinearRegression41D(self):
        size = 100
        x = np.random.random(size=size)
        y = 3.0 * x + 5.0 + np.random.normal(size=size)

        regression = SimpleLinearRegression41D()
        print(regression.fit(x, y))
        y_predict = regression.predict(x)

        plt.scatter(x, y)
        plt.plot(x, y_predict, color='r')
        plt.show()

    def test_MSE_RMSE_MAE(self):
        boston = datasets.load_boston()
        print(boston['feature_names'])
        x = boston['data'][:, 5]  # Get the `RM`(room) data
        y = boston['target']

        # clean data
        x = x[y < 50.0]
        y = y[y < 50.0]

        x_train, y_train, x_test, y_test = train_test_split(x, y)

        regression = SimpleLinearRegression41D()
        print(regression.fit(x_train, y_train))
        y_predict = regression.predict(x_test)

        print(mean_squared_error(y_test, y_predict))
        print(root_meas_squared_error(y_test, y_predict))
        print(mean_absolute_error(y_test, y_predict))

        plt.scatter(x_train, y_train)
        plt.plot(x_train, regression.predict(x_train), color='r')
        plt.show()



