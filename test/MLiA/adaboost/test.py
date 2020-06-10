from MLiA.adaboost import adaboost
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestAdaBoost(unittest.TestCase):
    def test_SimpleData(self):
        data_matrix, class_labels = adaboost.loadSimpleData()
        class_labels = np.array(class_labels)
        positive_axis = data_matrix[class_labels == 1.0]
        negative_axis = data_matrix[class_labels == -1.0]

        plt.title("Single layer decision tree test data")
        plt.xlim(xmin=0.8, xmax=2.2)
        plt.ylim(ymin=0.8, ymax=2.2)
        plt.scatter(positive_axis[:, 0], positive_axis[:, 1], s=150, marker='.', c="blue")
        plt.scatter(negative_axis[:, 0], negative_axis[:, 1], s=150, marker='^', c="red")
        plt.plot()
        plt.show()
