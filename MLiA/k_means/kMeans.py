import numpy as np


def loadDataSet(fileName):
    data_mat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            fit_line = map(float, cur_line)
            data_mat.append(fit_line)
        return data_mat
