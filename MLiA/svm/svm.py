import random
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    labelMat = []

    # 打开文件
    fr = open(fileName)

    # 逐行分析
    for line in fr.readlines():
        # 得到每行的类标签和整个数据矩阵
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
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
    
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数 C
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :return: 
    """
    # 转化为矩阵，可以简化很多数学操作
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)

    # alpha 列矩阵，矩阵中元素初始化为 0
    alphas = np.mat(np.zeros((m, 1)))

    # 在没有任何 alpha 改变的情况下遍历数据集的次数
    iter = 0

    # 当 iter 达到输入值 maxIter 时，函数结束运行并退出
    while iter < maxIter:

        # 每次该值设为 0，然后再对整个集合顺序遍历。变量 alphaPairsChanged 用于记录 alpha 是否已经进行优化
        alphaPairsChanged = 0
        for i in range(m):

            # 我们预测的类别
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b

            # 将预测结果与实际结果进行对比可以得到误差 Ei
            Ei = fXi - float(labelMat[i])

            # 如果误差很大，可以对该数据实例所对应的 alpha 值进行优化
            # 在该 if 语句下，无论是正间隔还是负间隔都会被测试，同时还要检查 alpha 值，以保证其不能等于 0 或 C
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):

                # 随机选择第二个 alpha 值，即 alpha[j]
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 保证 alpha 在 0 与 C 之间，
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # eta 是 alpha[j] 的最优修改值，有下面公式计算得出
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T

                # 大于等于 0 则退出本次 for 循环，这里简化了 eta == 0 的操作
                if eta >= 0:
                    print("eta>=0")
                    continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                # 检查 alpha[j] 是否有轻微改变，若是则退出循环
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 对 i 进行修改，修改量与 j 相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                # 在对 alpha[i] 和 alpha[j] 进行优化之后，给这两个 alpha 值设置一个常数项 b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 每成功改变一对 alpha，同时可以增加 alphaPairsChanged 的值
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 在 for 循环之外，需要检查 alpha 值是否做了更新，如果有更新则将 iter 设为 0 后继续运行程序
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
