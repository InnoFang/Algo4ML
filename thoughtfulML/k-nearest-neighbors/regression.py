import random
import sys

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 使用 KDTree 需要多层递归，需要提高递归限制以防止抛出错误
sys.setrecursionlimit(10000)


class Regression:
    def __init__(self):
        self.k = 5
        self.metric = np.mean
        self.kdtree = None
        self.houses = None
        self.values = None

    def set_data(self, houses, values):
        """
        设置房屋和价格数据
        :param houses: 带有房屋参数的 pandas.DataFrame
        :param values: 带有房屋价格的 pandas.Series
        :return: 
        """
        self.houses = houses
        self.values = values
        self.kdtree = KDTree(self.houses)

    def regress(self, query_point):
        """
        使用特定参数计算房屋的预测价格
        :param query_point: 带有房屋参数的 pandas.Series
        :return: 房屋价格
        """
        _, indexes = self.kdtree.query(query_point, self.k)
        value = self.metric(self.values.iloc[indexes])
        if np.isnan(value):
            raise Exception('Unexpected resultt')
        else:
            return value


class RegressionTest(object):
    """
    取出 King 县的房屋数据，计算并画出 kNN 回归错误率
    """

    def __init__(self):
        self.houses = None
        self.values = None

    def load_csv_file(self, csv_file, limit=None):
        """
        加载带有房屋数据的 CSV 文件
        :param csv_file: CSV 文件名
        :param limit: 需要读取的文件的行数
        :return: 
        """
        houses = pd.read_csv(csv_file, nrows=limit)
        self.values = houses['AppraisedValue']
        houses = houses.drop('AppraisedValue', 1)
        houses = (houses - houses.mean()) / (houses.max() - houses.min())
        self.houses = houses
        self.houses = self.houses[['lat', 'long', 'SqFtLot']]

    def plot_error_rates(self):
        folds_range = range(2, 11)
        errors_df = pd.DataFrame({'max': 0, 'min': 0}, index=folds_range)
        for folds in folds_range:
            errors = self.tests(folds)
            errors_df['max'][folds] = max(errors)
            errors_df['min'][folds] = min(errors)
        errors_df.plot(title='Mean Absolute Error of kNN over different folds_range')
        plt.xlabel('#folds_range')
        plt.ylabel('MAE')
        plt.show()

    def tests(self, folds):
        """
        计算一系列测试数据的平均绝对误差
        :param folds: 分割多少次数据
        :return: 错误值列表
        """
        holdout = 1 / float(folds)
        errors = []
        for _ in range(folds):
            values_regress, values_actual = self.test_regression(holdout)
            errors.append(mean_absolute_error(values_actual, values_regress))

        return errors

    def test_regression(self, holdout):
        """
        计算超出样本数据的回归
        :param holdout: 用于测试[0,1]的部分数据
        :return: tuple(y_regression, values_actual)
        """
        test_rows = random.sample(self.houses.index.tolist(),
                                  int(round(len(self.houses) * holdout)))
        train_rows = set(range(len(self.houses))) - set(test_rows)
        df_test = self.houses.ix[test_rows]
        df_train = self.houses.drop(test_rows)

        train_values = self.values.ix[train_rows]
        regression = Regression()
        regression.set_data(houses=df_train, values=train_values)

        values_regr = []
        values_actual = []

        for idx, row in df_test.iterrows():
            values_regr.append(regression.regress(row))
            values_actual.append(self.values[idx])

        return values_regr, values_actual


def main():
    regression_test = RegressionTest()
    regression_test.load_csv_file('data/king_county_data_geocoded.csv', 100)
    regression_test.plot_error_rates()


if __name__ == '__main__':
    main()
