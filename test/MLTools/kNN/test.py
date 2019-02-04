from MLTools.kNN import KNNClassifier
import unittest
import numpy as np
import matplotlib.pyplot as plt


class KNNClassifierTest(unittest.TestCase):

    def test_kNNClassifier(self):
        raw_data_X = [[3.393533211, 2.331273381],
                      [3.110073483, 1.781539638],
                      [1.343808831, 3.368360954],
                      [3.582294042, 4.679179110],
                      [2.280362439, 2.866990263],
                      [7.423436942, 4.696522875],
                      [5.745051997, 3.533989803],
                      [9.172168622, 2.511101045],
                      [7.792783481, 3.424088941],
                      [7.939820817, 0.791637231]
                      ]

        raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        X_train = np.array(raw_data_X)
        y_train = np.array(raw_data_y)

        knn_clf = KNNClassifier(k_neighbours=3)
        fit_result = knn_clf.fit(X_train, y_train)
        print(fit_result)

        x = np.array([8.093607318, 3.365731514])
        X_predict = np.array(x).reshape(1, -1)
        y_predict = knn_clf.predict(X_predict)
        print(y_predict)

        plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
        plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='b')
        plt.scatter(x[0], x[1], color='r')
        plt.show()
