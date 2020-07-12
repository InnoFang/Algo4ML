import numpy as np


def loadDataSet(fileName):
    data_mat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            fit_line = list(map(float, cur_line))
            data_mat.append(fit_line)
        return data_mat


def distEclud(vecA, vecB):
    """
    计算欧式距离
    :param vecA:
    :param vecB:
    :return:
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    构建一个包含 k 个随机质心的集合
    :param dataSet:
    :param k:
    :return:
    """
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = min(dataSet[:, j])
        range_j = float(max(dataSet[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids
