from MLTools.SimpleLinearRegression import SimpleLinearRegression41D
import unittest
import numpy as np
import matplotlib.pyplot as plt


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
