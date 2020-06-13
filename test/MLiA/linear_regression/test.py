import numpy as np
import unittest
from MLiA.linear_regresssion import dataset, regression


class TestLinearRegression(unittest.TestCase):
    def test_loadDataSet(self):
        x_arr, y_arr = dataset.load_ex0()
        print(x_arr[0: 2])
        ws = regression.standRegress(x_arr, y_arr)
        print(ws)

    def test_drawRegression(self):
        import matplotlib.pyplot as plt
        x_arr, y_arr = dataset.load_ex0()
        ws = regression.standRegress(x_arr, y_arr)
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])

        x_copy = x_mat.copy()
        x_copy.sort(0)
        y_hat = x_copy * ws
        ax.plot(x_copy[:, 1], y_hat)
        plt.show()
