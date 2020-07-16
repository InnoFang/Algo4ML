from MLiA.pca import pca
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import unittest


class TestPCA(unittest.TestCase):
    def test_PCA(self):
        data_mat = pca.loadDataSet('data/testSet.txt')
        low_data, recon_mat = pca.pca(data_mat, 1)
        print(np.shape(low_data))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
        ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], marker='o', s=50, c='red')
        plt.show()
