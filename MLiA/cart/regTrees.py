import numpy as np


def binSplitDataSet(dataSet, feature, value):
    """
    通过数组过滤方式将数据集合切分得到两个子集并返回
    :param dataSet: 数据集合
    :param feature: 待切分的特征
    :param value:   该特征的某个值
    :return: 两个切分后的数据子集
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    生成叶节点
    :param dataSet: 数据集
    :return:
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    在给定数据上计算目标变量的平方误差
    :param dataSet: 数据集
    :return: 总方差：均方误差 * 数据集中样本个数
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    用最佳方式切分数据集和生成相应的叶节点
    :param dataSet: 数据集合
    :param leafType: 对创建叶节点的函数的引用
    :param errType: 对总方差计算函数的引用
    :param ops: 用户自定义的参数构成的元组，用以完成树的构建
    :return:   切分特征、特征值
    """
    # 用户指定的参数，用于控制函数的停止时机，其中 tol_s 是容许的误差下降值，tol_n 是切分的最少样本数
    tol_s, tol_n = ops[0], ops[1]
    # 统计不同剩余特征值的数目，若为 1，则不需要再切分而直接返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    s = errType(dataSet)
    best_s, best_index, best_value = np.inf, 0, 0
    for feat_index in range(n - 1):
        for split_val in set(dataSet[:, feat_index].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, feat_index, split_val)
            if np.shape(mat0)[0] < tol_n or np.shape(mat1)[0] < tol_n:
                continue
            new_s = errType(mat0) + errType(mat1)
            if new_s < best_s:
                best_index = feat_index
                best_value = split_val
                best_s = new_s
    # 如果切分数据集后效果提升不够大，那么就不应该进行切分操作而直接创建叶节点
    if (s - best_s) < tol_s:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, best_index, best_value)
    # 检验两个切分后的子集大小，如果某个子集的大小小于用户定义的参数 tol_n，那么也不应切分，直接退出
    if np.shape(mat0)[0] < tol_n or np.shape(mat1)[0] < tol_n:
        return None, leafType(dataSet)
    return best_index, best_value


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 满足条件时，返回 none 和某类模型的值：如果是回归树，该模型是一个常数；如果是模型树，其模型则是一个线性方程
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    ret_tree = {'spInd': feat, 'spVal': val}
    l_set, r_set = binSplitDataSet(dataSet, feat, val)
    ret_tree['left'] = createTree(l_set, leafType, errType, ops)
    ret_tree['right'] = createTree(r_set, leafType, errType, ops)
    return ret_tree


def isTree(obj):
    """
    测试输入是否是一棵树，换句话说，也是用于判断当前处理的节点是否是叶节点
    :param obj:
    :return:
    """
    return type(obj).__name__ == 'dict'


def getMean(tree):
    """
    从上往下遍历直到叶节点为止，如果找到两个叶节点则计算它们的平均值
    :param tree:
    :return:
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """

    :param tree: 待剪枝的树
    :param testData: 剪枝所需的测试数据
    :return:
    """
    # 没有测试数据时对树进行塌陷处理（即返回树平均值）
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    # 树只要非空，则反复递归调用函数 prune() 对测试数据进行切分
    l_set, r_set = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 叶子，即不是子树，则对两个分支进行合并
    if not isTree(tree['right']) and not isTree(tree['left']):
        error_non_merge = np.sum(np.power(l_set[:, -1] - tree['left'], 2)) + \
                          np.sum(np.power(r_set[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = np.sum(np.power(testData[:, -1] - tree_mean, 2))
        # 对合并前后的误差进行比较，如果合并后的误差比不合并的误差小就进行合并操作，反之不合并并返回
        if error_merge < error_non_merge:
            print("merging")
            return tree_mean
        else:
            return tree
    # 如果是子树，则递归进行剪枝
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)
    return tree


def linearSolve(dataSet):
    """
    将数据集格式化成目标变量 Y 和自变量 X
    X 和 Y 用于执行简单的线性回归
    :param dataSet:
    :return:
    """
    m, n = np.shape(dataSet)
    X, Y = np.mat(np.ones((m, n))), np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    # 若矩阵的逆不存在则会出现异常
    if np.linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singular, cannot do inverse, try increasing the second value of ops")
    ws = xTx.I * (X.T * X)
    return ws, X, Y


def modelLeaf(dataSet):
    """
    当数据不再需要切分的时候负责生成叶节点的模型
    :param dataSet:
    :return:
    """
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """
    在给定的数据集上计算误差
    :param dataSet:
    :return: y_hat 和 Y 之间的平方误差
    """
    ws, X, Y = linearSolve(dataSet)
    y_hat = X * ws
    return np.sum(np.power(Y - y_hat, 2))
