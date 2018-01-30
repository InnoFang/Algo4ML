import numpy as np


def loadDataSet():
    """
    打开数据文件读取数据
    :return: 
    """
    data_mat = []
    label_mat = []
    with open('data/testSet.txt') as fr:
        for line in fr.readlines():
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(inX):
    """
    Sigmoid 函数
    ---------------------
                     1
    Sigmoid(x) = ----------
                  1+e^(-x)
    :param inX: 
    :return: 
    """
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabelLs):
    """
    梯度上升算法
    ----------------------
    伪代码如下：
    每个回归系数初始化为 1
    重复 R 次：
        计算整个数据集的梯度
        使用 alpha * gradient 更新回归系数的向量
        返回回归系数
    ----------------------
    公式：w = w + α▽f(w)
    :param dataMatIn: 数据矩阵
    :param classLabelLs: 类别向量
    :return: 得到最终的回归系数
    """
    # 转换为 NumPy 矩阵数据类型
    data_matrix = np.mat(dataMatIn)

    # classLabelLs 原本是行向量，为了计算，转化成 NumPy 矩阵后，还要对其进行转置
    label_mat = np.mat(classLabelLs).transpose()

    # 取得新的数据矩阵的行和列
    m, n = data_matrix.shape

    # 步长 α
    alpha = 0.001

    # 最大循环次数
    max_cycles = 500

    # 初始化回归系数的行为数据矩阵的列，列为1，便于之后矩阵相乘
    weights = np.ones((n, 1))

    # 重复 500 次
    for k in range(max_cycles):
        # 计算整个数据集的梯度，h 是列向量
        h = sigmoid(data_matrix * weights)

        # 计算真实类别与预测类别的差值
        error = (label_mat - h)

        # 按照差值方向调整回归系数，data_matrix 是行向量，error 是列向量，为了矩阵相乘，需要将 data_matrix 进行转置
        # 公式：w = w + α▽f(w)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights
