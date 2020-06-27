import numpy as np
import unittest
import matplotlib.pyplot as plt
from MLiA.linear_regresssion import regression


class TestLinearRegression(unittest.TestCase):
    def test_loadDataSet(self):
        x_arr, y_arr = regression.loadDataSet('data/ex0.txt')
        print(x_arr[0: 2])
        ws = regression.standRegress(x_arr, y_arr)
        print(ws)

    def test_drawRegression(self):
        import matplotlib.pyplot as plt
        x_arr, y_arr = regression.loadDataSet('data/ex0.txt')
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
        x_arr, y_arr = regression.loadDataSet('data/ex0.txt')
        # 对单点进行估计
        print(regression.lwlr(x_arr[0], x_arr, y_arr, 1.0))
        print(regression.lwlr(x_arr[0], x_arr, y_arr, 0.001))

        # 对所有点进行估计（依据不同 k 值），并查看对应 y_hat 的拟合效果
        x_mat = np.mat(x_arr)
        srt_ind = x_mat[:, 1].argsort(0)
        x_sort = x_mat[srt_ind][:, 0, :]

        fig = plt.figure()
        ax = fig.add_subplot(221)
        # k=1.0，拟合效果与最小二乘的拟合效果差不多
        y_hat = regression.lwlrTest(x_arr, x_arr, y_arr, 1.0)
        ax.plot(x_sort[:, 1], y_hat[srt_ind])
        ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0], s=2, c='red')

        ax = fig.add_subplot(222)
        # k=0.01，根据拟合效果可以挖出数据的潜在规律
        y_hat = regression.lwlrTest(x_arr, x_arr, y_arr, 0.01)
        ax.plot(x_sort[:, 1], y_hat[srt_ind])
        ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0], s=2, c='red')

        ax = fig.add_subplot(223)
        # k=0.003，考虑了过多的噪声，进而导致出现过拟合现象
        y_hat = regression.lwlrTest(x_arr, x_arr, y_arr, 0.003)
        ax.plot(x_sort[:, 1], y_hat[srt_ind])
        ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0], s=2, c='red')

        plt.show()

    def test_predictTheAgeOfAnAbalone(self):
        """
        预测鲍鱼(abalone)的年龄
        abalone 数据集：0-99训练集，100-199测试集
        :return:
        """
        ab_x, ab_y = regression.loadDataSet('data/abalone.txt')
        y_hat01 = regression.lwlrTest(ab_x[0:99], ab_x[0:99], ab_y[0:99], 0.1)
        y_hat1 = regression.lwlrTest(ab_x[0:99], ab_x[0:99], ab_y[0:99], 1)
        y_hat10 = regression.lwlrTest(ab_x[0:99], ab_x[0:99], ab_y[0:99], 10)

        # 为了分析预测误差的大小，使用 rssError() 计算出这一指标
        print(regression.rssError(ab_y[0:99], y_hat01.T))
        print(regression.rssError(ab_y[0:99], y_hat1.T))
        print(regression.rssError(ab_y[0:99], y_hat10.T))
        # 根据以上输出可以发现较小的核将得到较低的误差，但使用过小的核会造成过拟合，对新数据不一定能达到最好的预测效果

        # 测试在新数据上的表现
        y_hat01 = regression.lwlrTest(ab_x[100:199], ab_x[0:99], ab_y[0:99], 0.1)
        print(regression.rssError(ab_y[0:99], y_hat01.T))
        y_hat1 = regression.lwlrTest(ab_x[100:199], ab_x[0:99], ab_y[0:99], 1)
        print(regression.rssError(ab_y[0:99], y_hat1.T))
        y_hat10 = regression.lwlrTest(ab_x[100:199], ab_x[0:99], ab_y[0:99], 10)
        print(regression.rssError(ab_y[0:99], y_hat10.T))
        # 根据以上输出会发现和大小等于10时的测试误差最小，但在训练集上的误差确实最大的

        # 接下来再与简单的线性回归做个比较
        ws = regression.standRegress(ab_x[0:99], ab_y[0:99])
        y_hat = np.mat(ab_x[100:199]) * ws
        print(regression.rssError(ab_y[100:199], y_hat.T.A))
        # 简单线性回归达到了与局部加权线性回归类似的效果。这表明一点，必须在未知数据上比较效果才能选取到最佳模型
        # 如果要得到更好的效果，应该对多个不同的数据集做多次测试来比较结果

    def test_RidgeRegress(self):
        ab_x, ab_y = regression.loadDataSet('data/abalone.txt')
        ridge_weights = regression.ridgeTest(ab_x, ab_y)

        # 对岭回归的回归系数变化可视化
        # λ非常小时，系数与普通回归一样；λ非常大时，所有回归系数缩减为 0。
        # 可以在中间某处找到使得预测的结果最好的 λ 值
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ridge_weights)
        plt.show()

    def test_stageWise(self):
        x_arr, y_arr = regression.loadDataSet('data/abalone.txt')
        print(regression.stageWise(x_arr, y_arr, 0.001, 5000))

