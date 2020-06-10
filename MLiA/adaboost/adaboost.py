import numpy as np


def loadSimpleData():
    """
    简单数据集
    :return: 坐标，类别
    """
    data_matrix = np.array([[1., 2.1],
                   [2., 1.1],
                   [1.3, 1.],
                   [1., 1.],
                   [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_matrix, class_labels


def stumpClassify(data_matrix, dimen, threshVal, threshIneq):
    ret_array = np.ones((np.shape(data_matrix)[0]), 1)
    if threshIneq == 'lt':
        ret_array[data_matrix[:,]]
