from MLiA.cart import regTrees
import numpy as np
import unittest


class TestCART(unittest.TestCase):
    def test_binSplitDataSet(self):
        test_mat = np.mat(np.eye(4))
        print(test_mat)
        mat0, mat1 = regTrees.binSplitDataSet(test_mat, 1, 0.5)
        print(mat0)
        print(mat1)
