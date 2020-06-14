import numpy as np


def standRegress(xArr, yArr):
    """
    计算最佳拟合直线
    :param xArr: 特征向量
    :param yArr: 标签向量
    :return: 回归系数
    """
    x_mat, y_mat = np.mat(xArr), np.mat(yArr).T
    xTx = x_mat.T * x_mat
    # 若 xTx 行列式为 0，则表示不存在逆矩阵，直接退出
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    局部加权线性回归（Locally Weighted Linear Regression，LWLR）
    :return:
    """
    x_mat, y_mat = np.mat(xArr), np.mat(yArr).T
    m = np.shape(x_mat)[0]
    # 创建对角权重矩阵 weights
    weights = np.mat(np.eye((m)))
    # 计算每个样本点对应的权重值：随着样本点与待预测点距离的递增，权重将以指数级衰减，输入参数 k 控制衰减的速度
    for j in range(m):
        diff_mat = testPoint - x_mat[j, :]
        # 高斯核对应的权重计算公式：w(j, j) = exp{ |x^j - x| / (-2 * k^2) }
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    xTx = x_mat.T * (weights * x_mat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 局部加权线性回归系数：w^ = (X^T·W·X)^-1·X^T·W·y
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    用于为数据集中每个点调用 lwlr()，有助于求解 k 的大小
    :param testArr:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    m = np.shape(testArr)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(testArr[i], xArr, yArr, k)
    return y_hat
