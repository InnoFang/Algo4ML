from MLiA.knn import kNN, dataset
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestKNN(unittest.TestCase):

    def test_load_data(self):
        dating_data_mat, dating_labels = dataset.load_datingTestSet2()
        print(dating_data_mat)
        print(dating_labels[0:20])

    def test_data_plot(self):
        dating_data_mat, dating_labels = dataset.load_datingTestSet2()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
        ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],
                   s=15.0 * np.array(dating_labels), c=15.0 * np.array(dating_labels))  # s:scalar, c:color

        plt.xlabel('Percentage of Time Spent Playing Video Games')
        plt.ylabel('Liters of Ice Cream Consumed Per Week')

        label = ['didntLike', 'smallDoses', 'largeDoses']
        # loc 设置显示的位置，0是自适应
        # ncol 设置显示的列数
        ax.legend(label, loc=0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        plt.show()

    def test_autoNorm(self):
        dating_data_mat, dating_labels = dataset.load_datingTestSet2()
        norm_mat, ranges, min_vals = kNN.autoNorm(dating_data_mat)
        print(norm_mat)
        print(ranges)
        print(min_vals)

    def test_datingClassTest(self):
        kNN.datingClassTest()
        # """
        # 测试代码
        # :return:
        # """
        # # 用 10% 的数据做是测试样本
        # ho_ratio = 0.10
        #
        # # 获得数据集及标签
        # dating_data_mat, dating_labels = load_data.load_datingTestSet2()
        #
        # # 归一化处理，拿到归一化后的数据集，数据变化范围和特征值的最小值向量
        # norm_mat, ranges, min_vals = kNN.autoNorm(dating_data_mat)
        #
        # # 获得归一化后的数据集的行数
        # m = norm_mat.shape[0]
        #
        # # 对总个数的 10% 进行测试
        # num_test_vecs = int(m * ho_ratio)
        #
        # # 错误个数
        # error_count = 0.0
        # for i in range(num_test_vecs):
        #
        #     # 获得识别结果
        #     classifier_result = kNN.classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
        #                                   dating_labels[num_test_vecs:m], 3)
        #     print('the classifier came back with: %d, the real answer is: %d'
        #           % (classifier_result, dating_labels[i]))
        #
        #     # 识别结果与真实值对比
        #     if classifier_result != dating_labels[i]:
        #         error_count += 1.0
        #
        # # 打印输出错误率
        # print('the total error rate is: %f' % (error_count / float(num_test_vecs)))

    def test_classifyPersson(self):
        kNN.classifyPerson()

    def test_classifyPersson(self):
        kNN.classifyPerson()

    def test_img2verctor(self):
        import pkg_resources
        filename = pkg_resources.resource_filename(kNN.__name__, 'data/testDigits/0_0.txt')

        test_vector = kNN.img2vector(filename)
        for i in range(32):
            print(test_vector[0, 0 + i * 32:31 + i * 32])

    def test_handwritingClassTest(self):
        kNN.handwritingClassTest()


if __name__ == '__main__':
    unittest.main()
