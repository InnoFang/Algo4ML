import random


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
