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


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    k-均值算法
    :param dataSet: 数据集
    :param k: 簇的数目
    :param distMeas: 距离计算
    :param createCent: 创建质心
    :return: 所有的类质心，所有点的分配结果
    """
    m = np.shape(dataSet)[0]
    # 簇分配结果矩阵包含两列：第一列记录簇索引值，第二列存储误差（误差指当前点到簇质心的距离，后边会使用该误差来评价聚类的效果）
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    cluster_changed = True
    # 先创建 k 个质心，让后将点都分配到最近的质心，再重新计算质心，重复多次直到簇分配结果不再改变为止
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist, min_index = np.inf, -1
            # 寻找最近的质心
            for j in range(k):
                dist_ji = distMeas(centroids[j, :], dataSet[i, :])
                if dist_ji < min_dist:
                    min_dist, min_index = dist_ji, j
            # 只要任一点的簇分配结果发生改变，都更新 cluster_changed
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2
        print(centroids)
        # 更新质心的距离
        for cent in range(k):
            # 通过数组过滤来获得给定簇的所有点
            pts_in_clust = dataSet[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            # 计算所有点的均值，axis=0 表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = np.mean(pts_in_clust, axis=0)
    return centroids, cluster_assment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    # 创建一个初始簇
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    cent_list = [centroid0]
    # 遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m):
        cluster_assment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
    # 不停的对簇进行划分，直到得到想要的簇数目为止
    while len(cent_list) < k:
        lowest_sse = np.inf
        for i in range(len(cent_list)):
            # 尝试划分每一簇
            pts_in_curr_cluster = dataSet[np.nonzero(cluster_assment[:, 0].A == i)[0], :]
            centroid_mat, split_clust_ass = kMeans(pts_in_curr_cluster, 2)
            sse_split = split_clust_ass[:, 1].sum()
            sse_not_split = cluster_assment[np.nonzero(cluster_assment[:, 0].A != i)[0], 1].sum()
            print('sse_split:', sse_split, '; not_split:', sse_not_split)
            # 如果当前划分的 SSE 值最小，则本次划分被保存
            if (sse_split + sse_not_split) < lowest_sse:
                best_cent_to_split = i
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass.copy()
                lowest_sse = sse_split + sse_not_split
        # 更新簇的分配结果
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split
        print('the best_cent_to_split is:', best_cent_to_split)
        print('the len of bnest_clust_ass is:', len(best_clust_ass))
        cent_list[best_cent_to_split] = best_new_cents[0, :]
        cent_list.append(best_new_cents[1, :].tolist()[0])
        cluster_assment[np.nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_clust_ass
    return np.mat(cent_list), cluster_assment

