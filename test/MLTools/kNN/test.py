from MLTools.kNN import KNNClassifier
from MLTools.model_selection import train_test_split
from MLTools.metrics import accuracy_rate
import unittest
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets


class KNNClassifierTest(unittest.TestCase):

    def test_kNNClassifier(self):
        iris = datasets.load_iris()
        X = iris['data']
        y = iris['target']

        X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)

        knn_clf = KNNClassifier(k_neighbours=3)
        fit_result = knn_clf.fit(X_train, y_train)
        print(fit_result)

        # x = np.array([8.093607318, 3.365731514])
        # X_predict = np.array(x).reshape(1, -1)
        y_predict = knn_clf.predict(X_test)
        print(y_predict)
        print(y_test)

        # score = accuracy_rate(y_test, y_predict)
        # print(score)
        print(knn_clf.score(X_test, y_test))

        plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
        plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='b')
        plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], color='r')
        plt.show()
