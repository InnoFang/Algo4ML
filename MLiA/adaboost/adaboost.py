import numpy as np


def loadDataSet(fileName):
    num_feat = len(open(fileName).readline().split('\t'))
    data_mat, label_mat = [], []
    fr = open(fileName)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


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
    """
    检测是否有某个值小于或者大于正在测试的阈值，通过阈值比较对数据进行分类
    :param data_matrix: 输入数据，坐标矩阵
    :param dimen: 列数
    :param threshVal: 阈值
    :param threshIneq: "lt" 或 "gt" 小于或大于
    :return:
    """
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if threshIneq == 'lt':
        ret_array[data_matrix[:, dimen] <= threshVal] = 1.0
    else:
        ret_array[data_matrix[:, dimen] > threshVal] = -1.0
    return ret_array


def buildStump(dataArr, classLabels, D):
    """
    在加权数据集中循环，并找到具有最低错误率的单层决策树
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    """
    data_matrix, label_mat = np.mat(dataArr), np.mat(classLabels).T
    m, n = np.shape(data_matrix)
    # 用于在特征的所有可能值上进行遍历
    num_steps = 10.0
    # 用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    # 将最小错误率 min_error 设为 +∞
    min_error = np.inf
    # 对数据集中的每一个特征（第一层循环）
    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps
        # 对每个步长（第二层循环）
        for j in range(-1, int(num_steps) + 1):
            # 对每个不等号（第三层循环）
            for inequal in ['lt', 'gt']:
                # 建立一棵单层决策树并利用加权数据集对它进行测试
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stumpClassify(data_matrix, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                # 计算加权错误率
                weighted_error = D.T * err_arr
                # 如果错误率低于 minError，则将当前单层决策树设为最佳单层决策树
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                    i, thresh_val, inequal, weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    # 返回最佳单层决策树
    return best_stump, min_error, best_class_est


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weak_class_arr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        best_stump, error, class_est = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # 计算alpha，用于告诉总分类器本次单层决策树输出结果的权重
        # max(error, 1e-16) 用于确保在没有错误时不会发生除零溢出
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        print("class_est:", class_est.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, class_est)
        # 为下一次迭代计算 D
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 错误率累加计算，错误率为 0 时中止循环
        agg_class_est += alpha * class_est
        print("agg_class_est:", agg_class_est.T)
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(classLabels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print("total error:", error_rate)
        if error_rate == 0.0:
            break
    return weak_class_arr


def adaClassify(datToClass, classifierArr):
    data_matrix = np.mat(datToClass)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        class_est = stumpClassify(data_matrix, classifierArr[i]['dim'],
                                  classifierArr[i]['thresh'],
                                  classifierArr[i]['ineq'])
        agg_class_est += classifierArr[i]['alpha'] * class_est
        print(agg_class_est)
    return np.sign(agg_class_est)
