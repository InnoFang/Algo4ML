import numpy as np


def loadDataSet(fileName, delim='\t'):
    with open(fileName) as fr:
        string_arr = [line.strip().split(delim) for line in fr.readlines()]
        data_arr = [list(map(float, line)) for line in string_arr]
        return np.mat(data_arr)


def pca(dataMat, topNfeat=9999999):
    # 去除平均值
    mean_vals = np.mean(dataMat, axis=0)
    mean_removed = dataMat - mean_vals
    # 计算协方差矩阵
    cov_mat = np.cov(mean_removed, rowvar=False)
    # 计算协方差矩阵的特征值和特征向量
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    # 将特征值从大到小排序
    eig_val_ind = np.argsort(eig_vals)
    # 保留最上面的 N 个特征
    eig_val_ind = eig_val_ind[-1:-(topNfeat + 1):-1]
    red_eig_vects = eig_vects[:, eig_val_ind]
    # 将数据转换到上述 N 个特征向量构建的新空间中
    low_D_data_mat = mean_removed * red_eig_vects
    recon_mat = (low_D_data_mat * red_eig_vects.T) + mean_vals
    return low_D_data_mat, recon_mat
