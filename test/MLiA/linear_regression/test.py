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

    def test_lwlr(self):
        import matplotlib.pyplot as plt

        x_arr, y_arr = dataset.load_ex0()
        # 对单点进行估计
        print(regression.lwlr(x_arr[0], x_arr, y_arr, 1.0))
        print(regression.lwlr(x_arr[0], x_arr, y_arr, 0.001))

        # 对所有点进行估计
        y_hat = regression.lwlrTest(x_arr, x_arr, y_arr, 0.003)

        # 查看 y_hat 的拟合效果
        x_mat = np.mat(x_arr)
        srt_ind = x_mat[:, 1].argsort(0)
        x_sort = x_mat[srt_ind][:, 0, :]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_sort[:, 1], y_hat[srt_ind])
        ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0], s=2, c='red')
        plt.show()



