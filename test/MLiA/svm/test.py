from MLiA.svm import svm
import numpy as np
import unittest


class TestSVM(unittest.TestCase):
    def test_loadData(self):
        data_arr, label_arr = svm.loadDataSet('data/testSet.txt')
        print(label_arr)

    def test_smoSimple(self):
        data_arr, label_arr = svm.loadDataSet('data/testSet.txt')
        b, alphas = svm.smoSimple(data_arr, label_arr, 0.6, 0.001, 40)
        print(b)
        print(alphas[alphas > 0])
        print("支持向量的个数：", np.shape(alphas[alphas > 0]))
        print("哪些数据点是支持向量：", "data", "label")
        for i in range(100):
            if alphas[i] > 0.0:
                print(data_arr[i], label_arr[i])

    def test_smoPlatt(self):
        data_arr, label_arr = svm.loadDataSet("data/testSet.txt")
        b, alphas = svm.smoPlatt(data_arr, label_arr, 0.6, 0.001, 40)
        ws = svm.calcWs(alphas, data_arr, label_arr)
        print(ws)
        data_mat = np.mat(data_arr)
        print(data_mat[0] * np.mat(ws) + b)
        print(label_arr[0])
