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

    def test_postpruning(self):
        # 创建一棵尽可能大的树
        my_data = dataset.load_ex2()
        my_mat = np.mat(my_data)
        my_tree = regTrees.createTree(my_mat)

        # 导入测试数据
        my_data_test = dataset.load_ex2test()
        my_mat_test = np.mat(my_data_test)
        new_tree = regTrees.prune(my_tree, my_mat_test)
        print(new_tree)

    def test_modelTree(self):
        my_mat = np.mat(dataset.load_exp2())
        my_tree = regTrees.createTree(my_mat, regTrees.modelLeaf, regTrees.modelErr, (1, 10))
        print(my_tree)

    def test_comparing(self):
        train_mat = np.mat(dataset.load_bikeSpeedVsIq_train())
        test_mat = np.mat(dataset.load_bikeSpeedVsIq_test())

        # 回归树
        my_tree = regTrees.createTree(train_mat, ops=(1, 20))
        y_hat = regTrees.createForecast(my_tree, test_mat[:, 0])
        # 输出相关系数，y_hat为预测值，后者为真实值
        print(np.corrcoef(y_hat, test_mat[:, 1], rowvar=False)[0, 1])

        # 模型树
        ws, X, Y = regTrees.linearSolve(train_mat)
        for i in range(np.shape(test_mat)[0]):
            y_hat[i] = test_mat[i, 0] * ws[1, 0] + ws[0, 0]
        print(np.corrcoef(y_hat, test_mat[:, 1], rowvar=False)[0, 1])
