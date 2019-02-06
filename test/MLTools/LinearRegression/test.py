from MLTools.SimpleLinearRegression import SimpleLinearRegression41D
import unittest
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionTest(unittest.TestCase):

    def test_SimpleLinearRegression41D(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 3, 2, 3, 5], dtype=float)

        regression = SimpleLinearRegression41D()
        print(regression.fit(x, y))
        y_predict = regression.transform(x)

        plt.scatter(x, y)
        plt.plot(x, y_predict, color='r')
        plt.show()
