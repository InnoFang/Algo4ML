from MLiA.adaboost import adaboost
import matplotlib.pyplot as plt
import unittest


class TestAdaBoost(unittest.TestCase):
    def test_SimpleData(self):
        data_matrix, class_labels = adaboost.loadSimpleData()
        assert len(data_matrix) == len(class_labels)
        data_len = len(data_matrix)
        positive_X, positive_Y = [], []
        negative_X, negative_Y = [], []
        for i in range(data_len):
            # label is positive
            if class_labels[i] == 1.0:
                positive_X.append(data_matrix[i][0])
                positive_Y.append(data_matrix[i][1])
            # label is negative
            elif class_labels[i] == -1.0:
                negative_X.append(data_matrix[i][0])
                negative_Y.append(data_matrix[i][1])

        plt.title("Single layer decision tree test data")
        plt.xlim(xmin=0.8, xmax=2.2)
        plt.ylim(ymin=0.8, ymax=2.2)
        plt.scatter(positive_X, positive_Y, s=150, marker='.', c="blue")
        plt.scatter(negative_X, negative_Y, s=150, marker='^', c="red")
        plt.show()