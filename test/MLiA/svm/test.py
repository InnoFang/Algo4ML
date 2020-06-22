from MLiA.svm import svm
import numpy as np
import unittest


class TestSVM(unittest.TestCase):
    def test_loadData(self):
        data_arr, label_arr = svm.loadDataSet('data/testSet.txt')
        print(label_arr)

        # b, alphas = svm.smoSimple(data_arr, label_arr, 0.6, 0.001, 40)
        # print(b)
