import numpy as np
import operator
from os import listdir

"""
simple data set
"""


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
dating match
"""


def classify0(inX, dataSet, labels, k):
    """
    :param inX: 用于分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签向量
    :param k: 选择最近的邻居数目
    :return: 拿到出现数量最多的标签
    """
    # dataSet 的行号，即 dataSet 中的个数
    data_set_size = dataSet.shape[0]

    # 将输入向量构成与 dataSet 维度一样的向量，并且减去 dataSet 中对应位置的值
    diff_mat = np.tile(inX, (data_set_size, 1)) - dataSet

    # 对每个元素位置上的差值取平方
    sq_diff_mat = diff_mat ** 2

    # 按行求平方和，axis=1 表示按行求和，axis=0 表示按列求和
    sq_distances = sq_diff_mat.sum(axis=1)

    # 根据欧式距离公式，求还需对每个值开根号
    distances = sq_distances ** 0.5

    # 对 distances 这个 array 从小到大排序，返回排序后的索引，从 0 开始
    sorted_dist_indices = np.argsort(distances)
    class_count = {}
    for i in range(k):
        # 拿到排序后索引对应位置的 label
        vote_in_label = labels[sorted_dist_indices[i]]

        # 对拿到的 label 进行计数
        class_count[vote_in_label] = class_count.get(vote_in_label, 0) + 1

    # 对 class_count 按数量逆序排序，大的在前，返回一个二维数组，
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    # 拿到标签数量最多的标签
    return sorted_class_count[0][0]


def file2matrix(filename):
    """
    将文件中的数据转到矩阵中，并返回对应矩阵和类别标签向量
    :param filename: 数据文件名
    :return: 数据矩阵和类别标签向量
    """
    # 打开数据文件
    with open(filename) as fr:
        # 读取文件中的每一行
        array_of_lines = fr.readlines()

        # 获取读取的行数
        number_of_lines = len(array_of_lines)

        # 生成一个与数据文件中数据个数相同，列数为 3 的 0 矩阵
        return_mat = np.zeros((number_of_lines, 3))

        # 类别标签向量
        class_label_vector = []

        # 当前行数，即对应行的下标
        index = 0
        for line in array_of_lines:
            # 去掉每一行中的 \n
            line = line.strip()

            # 将数据以 \t 为分隔符，分割成多个部分
            list_from_line = line.split('\t')

            # 将当前行的数据的前三个部分的内容赋给返回矩阵的对应行的位置
            return_mat[index, :] = list_from_line[0:3]

            # 每一行最后一个内容为标签（label） ，将其复制给类别标签向量
            class_label_vector.append(int(list_from_line[-1]))

            # 当前行数加一
            index += 1

    # 返回矩阵和类别标签向量
    return return_mat, class_label_vector


def autoNorm(dataSet):
    """
    归一化
    在处理不同取值范围的特征值时，采用将数值归一化的方式，下面的公式可以将任意取值范围的特征值转化为 0 到 1 区间内的值
    -----------------------------------------------
    公式：new_value = (old - min) / (max - min)
    -----------------------------------------------
    :param dataSet: 需要归一化的数据集
    :return: 归一化后的数据集，变化范围和数据集中每列最小值
    """
    # 取得数据集中每列的最小值
    min_vals = dataSet.min(axis=0)
    # 取得数据集中每列的最大值
    max_vals = dataSet.max(axis=0)

    # 变化范围，由最大值减去最小值得到
    ranges = max_vals - min_vals

    # 归一化后的数据集，先生成一个与原数据集相同大小的 0 数据集
    norm_data_set = np.zeros(np.shape(dataSet))

    # 原数据的个数，即行数
    m = dataSet.shape[0]

    # 对数据集中每个元素减去数据集中的最小值
    norm_data_set = dataSet - np.tile(min_vals, (m, 1))

    # 归一化处理，特征值相除 。**这里不是矩阵相除，矩阵相处函数为 np.linalg.solve(matA, matB)**
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))

    # 返回归一化后的数据集，变化范围和数据集中每列最小值
    return norm_data_set, ranges, min_vals


def datingClassTest():
    """
    测试代码
    :return: 
    """
    import pkg_resources
    filename = pkg_resources.resource_filename(__name__, 'data/datingTestSet2.txt')
    # 用 10% 的数据做是测试样本
    ho_ratio = 0.10

    # 获得数据集及标签
    dating_data_mat, dating_labels = file2matrix(filename)

    # 归一化处理，拿到归一化后的数据集，数据变化范围和特征值的最小值向量
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)

    # 获得归一化后的数据集的行数
    m = norm_mat.shape[0]

    # 对总个数的 10% 进行测试
    num_test_vecs = int(m * ho_ratio)

    # 错误个数
    error_count = 0.0
    for i in range(num_test_vecs):

        # 获得识别结果
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d'
              % (classifier_result, dating_labels[i]))

        # 识别结果与真实值对比
        if classifier_result != dating_labels[i]:
            error_count += 1.0

    # 打印输出错误率
    print('the total error rate is: %f' % (error_count / float(num_test_vecs)))


def classifyPerson():
    import pkg_resources
    filename = pkg_resources.resource_filename(__name__, 'data/datingTestSet2.txt')

    result_list = ['not at all', 'in small does', 'in large does']
    percent_time_ats = float(input('percentage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    # 获得数据集及标签
    dating_data_mat, dating_labels = file2matrix(filename)
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_time_ats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print('You will probably like this person: ', result_list[classifier_result - 1])


"""
handwriting recognition
"""


def img2vector(filename):
    """
    将图像文本转化为向量
    图像文本为 32 * 32，转化为向量长度为 1024
    :param filename: 图像文本文件名
    :return: 图像文本的向量
    """
    return_vect = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line_string = fr.readline()
            for j in range(32):
                return_vect[0, 32 * i + j] = int(line_string[j])
    return return_vect


def handwritingClassTest():
    """
    测试手写字识别
    [注] 这里的手写字指的是图片文本，数字从 0 到 9
    :return: 
    """
    import pkg_resources

    # 手写字类别
    hw_labels = []

    # 训练样本目录
    training_file_list = pkg_resources.resource_listdir(__name__, 'data/trainingDigits')

    # 训练样本个数
    m = len(training_file_list)

    # 训练样本向量
    training_mat = np.zeros((m, 1024))

    # 对每个训练样本进行处理
    for i in range(m):
        # 训练样本文件名
        file_name_str = training_file_list[i]

        # 分割文件名， '_' 前的字符为类别
        file_str = file_name_str.split('_')[0]
        class_number_str = int(file_str.split('_')[0])

        # 添加训练样本的类别
        hw_labels.append(class_number_str)

        # 将训练样本转化为向量
        training_mat[i, :] = img2vector(
            pkg_resources.resource_filename(__name__, 'data/trainingDigits/%s' % file_name_str))

    # 测试样本
    test_file_list = pkg_resources.resource_listdir(__name__, 'data/testDigits')

    # 错误率
    error_count = 0.0
    # 测试样本个数
    test = len(test_file_list)

    # 对每个测试样本处理
    for i in range(test):
        # 测试样本文件名
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('_')[0]
        class_number_str = int(file_name_str.split('_')[0])

        # 将测试样本转化为向量
        vector_under_test = img2vector(pkg_resources.resource_filename(__name__, 'data/testDigits/%s' % file_name_str))

        # 识别结果
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, class_number_str))

        # 识别结果与真值对比
        if classifier_result != class_number_str:
            error_count += 1.0
    print('\nthe total number of errors is : %d' % error_count)
    print('\nthe total error rate is: %f' % (error_count / float(test)))
