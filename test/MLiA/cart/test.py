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

    def test_prepruning(self):
        my_data = dataset.load_ex2()
        my_mat = np.mat(my_data)
        my_tree = regTrees.createTree(my_mat)
        # 输出很多叶节点
        print(my_tree)

        print("prepruning")

        # 修改停止条件 tol_s，tol_s 对误差的数量级十分敏感
        # 在选项中耗费时间并对误差容忍度去平方值，也能得到仅有两个叶节点组成的树
        my_tree = regTrees.createTree(my_mat, ops=(10000, 4))
        print(my_tree)