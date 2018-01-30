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
    公式：W = w + α▽f(w)
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


def plotBestFit(weights):
    """
    画出决策边界
    :param weights: 
    :return: 
    """
    import matplotlib.pyplot as plt
    data_mat, label_mat = loadDataSet()
    # 将 data_mat 类型转化为 NumPy 的 array 类型
    data_arr = np.array(data_mat)

    # 取得 data_arr 的行数
    n = data_arr.shape[0]

    # 分别记录 X1 类别和 X2 类别每个点的横纵坐标
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1]);ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1]);ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)

    # 设 sigmoid 函数为 0，0 是两个分类 (0 类和 1 类) 的分界处
    # 因此，我们设定 0 = W0X0 + W1X1 + W2X2, W0,W1,W2 已知，且 X0=1 (原因见 @loadDataSet)
    # 我们需要解出 X2 和 X1 的关系式，即分隔线的方程
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
