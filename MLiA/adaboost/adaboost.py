import numpy as np


def loadSimpleData():
    """
    简单数据集
    :return: 坐标，类别
    """
    data_matrix = [[1., 2.1],
                   [2., 1.1],
                   [1.3, 1.],
                   [1., 1.],
                   [2., 1.]]
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_matrix, class_labels
