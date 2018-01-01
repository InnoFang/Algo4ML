"""
not important
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from knn import kNN

# dating_data_mat, dating_labels = kNN.file2matrix('data/datingTestSet2.txt')

# # print(dating_data_mat)
# # print(dating_labels[0:20])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
# ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],
#            s=15.0 * np.array(dating_labels), c=15.0 * np.array(dating_labels))  # s:scalar, c:color
#
# plt.xlabel('Percentage of Time Spent Playing Video Games')
# plt.ylabel('Liters of Ice Cream Consumed Per Week')
#
# # label = ['didntLike', 'smallDoses', 'largeDoses']
# # # loc 设置显示的位置，0是自适应
# # # ncol 设置显示的列数
# # # ax.legend(label, loc=0)
# # handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles[::-1], labels[::-1])
# plt.show()


# norm_mat, ranges, min_vals = kNN.autoNorm(dating_data_mat)
# print(norm_mat)
# print(ranges)
# print(min_vals)


# kNN.datingClassTest()

kNN.classifyPerson()
