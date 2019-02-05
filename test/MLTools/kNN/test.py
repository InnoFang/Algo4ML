from MLTools.kNN import KNNClassifier
from MLTools.model_selection import train_test_split
from MLTools.metrics import accuracy_rate
from MLTools.preprocessing import StandardScaler
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

        score = accuracy_rate(y_test, y_predict)
        print(score)
        print(knn_clf.score(X_test, y_test))

        plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
        plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='b')
        plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], color='r')
        plt.show()

    def test_find_the_best_k(self):
        iris = datasets.load_iris()
        X = iris['data']
        y = iris['target']

        X_train, y_train, X_test, y_test = train_test_split(X, y)

        best_score = 0.0
        best_k = -1

        for k in range(1, 11):
            knn_clf = KNNClassifier(k_neighbours=k)
            knn_clf.fit(X_train, y_train)
            score = knn_clf.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_k = k

        print('The best K is {} and the best score is {}'.format(best_k, best_score))

    def test_StandardScaler(self):
        iris = datasets.load_iris()
        X = iris['data']
        y = iris['target']

        X_train, y_train, X_test, y_test = train_test_split(X, y)

        # print(X_train)
        std_scaler = StandardScaler()
        fit = std_scaler.fit(X_train)
        print(fit)
        X_train = std_scaler.transform(X_train)
        print(X_train)

        # print(X_test)
        X_test = std_scaler.transform(X_test)
        print(X_test)
