from MLiA.cart import regTrees, dataset
import numpy as np
import unittest


class TestCART(unittest.TestCase):
    def test_binSplitDataSet(self):
        test_mat = np.mat(np.eye(4))
        print(test_mat)
        mat0, mat1 = regTrees.binSplitDataSet(test_mat, 1, 0.5)
        print(mat0)
        print(mat1)

    def test_createTree(self):
        my_data = dataset.load_ex00()
        my_mat = np.mat(my_data)
        my_tree = regTrees.createTree(my_mat)
        print(my_tree)
