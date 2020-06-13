import numpy as np
import unittest
from MLiA.linear_regresssion import dataset, regression


class TestLinearRegression(unittest.TestCase):
    def test_loadDataSet(self):
        x_arr, y_arr = dataset.load_ex0()
        print(x_arr[0: 2])
        ws = regression.standRegress(x_arr, y_arr)
        print(ws)
