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


def rssError(yArr, yHatArr):
    """
    用于分析预测误差的大小
    :param yArr:
    :param yHatArr:
    :return:
    """
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegress(xMat, yMat, lam=0.2):
    """
    岭回归计算公式：w^ = (X^T·X + λI)^-1 · X^T · y
    :param xMat:
    :param yMat:
    :param lam: λ
    :return:
    """
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    # 按理说岭回归不会导致行列式为 0 的情况，但仍需要检查，因为当 lam=0 时，岭回归就变成了普通回归
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    x_mat, y_mat = np.mat(xArr), np.mat(yArr)
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    # 所有特征均减去各自的均值并除以方差
    x_mat = (x_mat - x_means) / x_var
    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridgeRegress(x_mat, y_mat, np.exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat
