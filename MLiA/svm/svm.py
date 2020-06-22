import random
import numpy as np


def loadDataSet(fileName):
    data_mat = []
    label_mat = []

    # 打开文件
    with open(fileName) as fr:
        # 逐行分析
        for line in fr.readlines():
            # 得到每行的类标签和整个数据矩阵
            line_arr = line.strip().split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def selectJRand(i, m):
    """
    用于在某个区间范围内随机选择一个整数
    :param i: 第一个 alpha 的下标
    :param m: 所有 alpha 的数目
    :return: 返回一个 0 到 m 且不等于 i 的值
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """
    用于调整大于 H 或小于 L 的 alpha 值
    :param aj: 
    :param H: 
    :param L: 
    :return: 
    """
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


"""
简单 SMO 函数的伪代码大致如下：

创建一个 alpha 向量并将其初始化为 0 向量
当迭代次数小于最大迭代次数时（外循环）：
    对数据集中的每个数据向量（内循环）：
        如果该数据向量可以被优化：
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有向量都没被优化，增加迭代数目，继续下一次循环
"""


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    SMO 算法的简单实现
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数 C
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :return: 
    """
    # 转化为矩阵，可以简化很多数学操作
    data_matrix = np.mat(dataMatIn)
    label_mat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(data_matrix)

    # alpha 列矩阵，矩阵中元素初始化为 0
    alphas = np.mat(np.zeros((m, 1)))

    # 在没有任何 alpha 改变的情况下遍历数据集的次数
    iter = 0

    # 当 iter 达到输入值 maxIter 时，函数结束运行并退出
    while iter < maxIter:

        # 每次该值设为 0，然后再对整个集合顺序遍历。变量 alpha_pairs_changed 用于记录 alpha 是否已经进行优化
        alpha_pairs_changed = 0
        for i in range(m):

            # 我们预测的类别
            fXi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b

            # 将预测结果与实际结果进行对比可以得到误差 Ei
            Ei = fXi - float(label_mat[i])

            # 如果误差很大，可以对该数据实例所对应的 alpha 值进行优化
            # 在该 if 语句下，无论是正间隔还是负间隔都会被测试，同时还要检查 alpha 值，以保证其不能等于 0 或 C
            if (label_mat[i] * Ei < -toler and alphas[i] < C) or (label_mat[i] * Ei > toler and alphas[i] > 0):

                # 随机选择第二个 alpha 值，即 alpha[j]
                j = selectJRand(i, m)
                fXj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                Ej = fXj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # 保证 alpha 在 0 与 C 之间，
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # eta 是 alpha[j] 的最优修改值，有下面公式计算得出
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
                      data_matrix[i, :] * data_matrix[i, :].T - \
                      data_matrix[j, :] * data_matrix[j, :].T

                # 大于等于 0 则退出本次 for 循环，这里简化了 eta == 0 的操作
                if eta >= 0:
                    print("eta>=0")
                    continue

                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                # 检查 alpha[j] 是否有轻微改变，若是则退出循环
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue
                # 对 i 进行修改，修改量与 j 相同，但方向相反
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

                # 在对 alpha[i] 和 alpha[j] 进行优化之后，给这两个 alpha 值设置一个常数项 b
                b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * \
                     data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * \
                     data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * \
                     data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * \
                     data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 每成功改变一对 alpha，同时可以增加 alpha_pairs_changed 的值
                alpha_pairs_changed += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
        # 在 for 循环之外，需要检查 alpha 值是否做了更新，如果有更新则将 iter 设为 0 后继续运行程序
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


"""
完整版的 Platt SMO 算法，相比于简单版的 SMO 算法，实现 alpha 的更改和代数运算的优化环节一模一样
在优化过程中，唯一的不同就是选择 alpha 的方式
完整版的 Platt SMO 算法应用了一些能够提速的启发方法，该算法是通过一个外循环来选择第一个 alpha 值的，
并且选择过程会在两种方式之间交替：
    一种方式是在所有数据集上进行单遍扫描；
    另一种方式则是在非边界 alpha （不等于边界 0 或 C 的 alpha 值） 中实现单遍扫描
"""


class optStruct:
    """
    使用一个数据结构老保存所有的重要值，可以省掉很多手工输入的麻烦
    """

    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.label_mat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zero((self.m, 1)))
        self.b = 0
        # 误差缓存
        self.e_cache = np.mat(np.zero((self.m, 2)))


def calcEk(oS, k):
    """
    计算误差值
    :param oS:
    :param k:
    :return:
    """
    fXk = float(np.multiply(oS.alphas, oS.label_mat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.label_mat[k])
    return Ek


def selectJ(i, oS, Ei):
    """
    用于选择第二个 alpha 值（或内循环的 alpha）
    :param i:
    :param oS:
    :param Ei:
    :return:
    """
    max_k, max_delta_e, Ej = -1, 0, 0
    oS.e_cache[i] = [1, Ei]
    valid_e_cache_list = np.nonzero(oS.e_cache[:, 0].A)[0]
    if len(valid_e_cache_list) > 1:
        for k in valid_e_cache_list:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            delta_e = abs(Ei - Ek)
            if delta_e > max_delta_e:
                max_k, max_delta_e, Ej = k, delta_e, Ek
        return max_k, Ej
    else:
        j = selectJRand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """
    计算误差值后放入误差缓存
    :param oS:
    :param k:
    :return:
    """
    Ek = calcEk(oS, k)
    oS.e_cache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(i, oS)
    if (oS.label_mat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (oS.label_mat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alpha_i_old, alpha_j_old = oS.alphas[i].copy(), oS.alphas[j].copy()
        if oS.label_mat[i] != oS.label_mat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.label_mat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alpha_j_old) < 0.00001:
            print("j not moving enough:")
            return 0
        oS.alphas[i] += oS.label_mat[j] * oS.label_mat[i] * (alpha_j_old - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.label_mat[i] * (oS.alphas[i] - alpha_i_old) * oS.X[i, :] * oS.X[i, :].T - oS.label_mat[j] * \
             (oS.alphas[j] - alpha_j_old) * oS.X[i, j] * oS.X[j, j].T
        b2 = oS.b - Ej - oS.label_mat[i] * (oS.alphas[i] - alpha_i_old) * oS.X[i, :] * oS.X[j, :].T - oS.label_mat[j] * \
             (oS.alphas[j] - alpha_j_old) * oS.X[j, j] * oS.X[j, j].T
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoPlatt(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    完整的 Platt SMO 的外循环代码
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :param kTup:
    :return:
    """
    oS = optStruct(
        np.mat(dataMatIn),
        np.mat(classLabels).transpose(),
        C,
        toler
    )
    iter = 0
    entire_set, alpha_pairs_changed = True, 0
    while iter < maxIter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(oS.m):
                alpha_pairs_changed += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
            iter += 1
        else:
            non_bound_is = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print("iteration number: %d" % iter)
    return oS.B, oS.alphas
