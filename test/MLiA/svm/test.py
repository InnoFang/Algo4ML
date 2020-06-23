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

    def test_Rbf(self, k1=1.3):
        data_arr, label_arr = svm.loadDataSet("data/testSetRBF.txt")
        b, alphas = svm.smoPlatt(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))
        data_mat, label_mat = np.mat(data_arr), np.mat(label_arr).transpose()
        sv_ind = np.nonzero(alphas.A > 0)[0]
        sVs = data_mat[sv_ind]
        label_sv = label_mat[sv_ind]
        print("there are %d Support Vectors" % np.shape(sVs)[0])
        m, n = np.shape(data_mat)
        error_count = 0
        for i in range(m):
            kernel_eval = svm.kernelTrans(sVs, data_mat[i, :], ('rbf', k1))
            predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_ind]) + b
            if np.sign(predict) != np.sign(label_arr[i]):
                error_count += 1
        print("the training error rate is: %f" % (float(error_count) / m))
        data_arr, label_arr = svm.loadDataSet("data/testSetRBF2.txt")
        error_count = 0
        data_mat, label_mat = np.mat(data_arr), np.mat(label_arr).transpose()
        m, n = np.shape(data_mat)
        for i in range(m):
            kernel_eval = svm.kernelTrans(sVs, data_mat[i, :], ('rbf', k1))
            predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_ind]) + b
            if np.sign(predict) != np.sign(label_arr[i]):
                error_count += 1
        print("the test error rate is: %f" % (float(error_count) / m))
