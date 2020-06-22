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


def selectJrand(i, m):
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
SMO 函数的伪代码大致如下：

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
                j = selectJrand(i, m)
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
