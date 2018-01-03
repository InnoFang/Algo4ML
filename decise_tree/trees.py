from math import log


def createDataSet():
    """
    获取简单数据集
    :return: 
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calcShannonEnt(dataSet):
    """
    计算信息熵
    :param dataSet: 输入数据集
    :return: 当前数据集的信息熵
    """
    # 数据集个数
    num_entries = len(dataSet)

    # 类别计数
    label_counts = {}

    # 取得数据集中的每一个特征向量
    for feat_vec in dataSet:
        # 取得当前特征向量的类别
        current_label = feat_vec[-1]

        # 如果还没有该类别的记录，则添加
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0

        # 对当前类别计数加一
        label_counts[current_label] += 1

    # 信息熵
    shannon_ent = 0.0

    # 对每个类别求信息熵
    for key in label_counts:
        # 当前类别占总个数的比例
        prob = float(label_counts[key]) / num_entries

        # 信息熵
        shannon_ent -= prob * log(prob, 2)

    # 返回信息熵
    return shannon_ent


def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    :param dataSet: 带划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return: 符合要求的数据集
    """
    # 返回符合要求的数据集。为了不改变原来的 dataSet，故创建一个新的列表
    ret_data_set = []
    for feat_vec in dataSet:
        # 当选择的特征与需要返回的特征的值一致时，进行下列操作
        if feat_vec[axis] == value:
            # 将 [0, axis) 这一段的特征值添加到 ret_data_set
            reduced_feat_vec = feat_vec[:axis]

            # 将 [axis + 1, 末尾] 这一段的特征值添加到 ret_data_set。[注]使用 extend 会将原列表中的值添加进目的列表
            reduced_feat_vec.extend(feat_vec[axis + 1:])

            # 将符合要求的列表添加到返回数据集中。[注]使用 append 会把整个列表当作元素添加进目的列表
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set
