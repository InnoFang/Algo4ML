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
    ws = xTx.I * (x_mat * y_mat)
    return ws
