from MLiA.logistic_regression import logRegres, dataset
import unittest
import numpy as np


class TestLogisticRegression(unittest.TestCase):

    def test_gradAscent(self):
        data_arr, label_mat = logRegres.loadDataSet()
        print(logRegres.gradAscent(data_arr, label_mat))

    def test_plotBestFit(self):
        data_arr, label_mat = logRegres.loadDataSet()
        weights = logRegres.gradAscent(data_arr, label_mat)
        # getA equivalent to ``np.asarray(self)``.
        # Change the weights' type into array type
        logRegres.plotBestFit(weights.getA())

    def test_stocGradAscent1(self):
        data_arr, label_mat = logRegres.loadDataSet()
        weights = logRegres.stocGradAscent1(np.array(data_arr), label_mat)
        logRegres.plotBestFit(weights)
        weights = logRegres.stocGradAscent1(np.array(data_arr), label_mat, 500)
        logRegres.plotBestFit(weights)

    def test_predict(self):
        # Predict the death rate of a sick horse from a hernia (or colic) disease
        logRegres.multiTest()