from MLiA.k_means import kMeans
import numpy as np
import unittest


class TestKMeans(unittest.TestCase):
    def test_randCent(self):
        data_mat = np.mat(kMeans.loadDataSet('data/testSet.txt'))
        print('min(data_mat[:, 0]) =', min(data_mat[:, 0]))
        print('min(data_mat[:, 1]) =', min(data_mat[:, 1]))
        print('max(data_mat[:, 0]) =', max(data_mat[:, 0]))
        print('max(data_mat[:, 1]) =', max(data_mat[:, 1]))
        centroids = kMeans.randCent(data_mat, 2)
        print('centroids:', centroids)
        print('kMeans.distEclud(data_mat[0], data_mat[1]):', kMeans.distEclud(data_mat[0], data_mat[1]))

    def test_kMeans(self):
        data_mat = np.mat(kMeans.loadDataSet('data/testSet.txt'))
        kMeans.kMeans(data_mat, 4)

    def test_biKmeans(self):
        data_mat = np.mat(kMeans.loadDataSet('data/testSet2.txt'))
        cent_list, my_new_assments = kMeans.biKmeans(data_mat, 3)
        print('cent_list:', cent_list)