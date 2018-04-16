from math import log
import operator


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
    --------------------------------------
    公式：Entropy(s) = ∑-P(x)log_2(P(x))
    --------------------------------------
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


def chooseBestFeatureToSplit(dataSet):
    """
    对每个特征计算条件熵，并比较出那个特征的信息增益最大，选出最佳的特征进行决策树划分
    :param dataSet: 数据集
    :return: 具有最大信息增益的特征
    """
    # 获得除目标变量外的特征数量
    num_features = len(dataSet[0]) - 1

    # 获得整个数据集的信息熵
    base_entropy = calcShannonEnt(dataSet)

    # 最大信息增益
    best_info_gain = 0.0

    # 进行划分的最佳特征
    best_feature = -1

    # 对每个特征计算信息增益并取得最大值
    for i in range(num_features):
        # 取得数据集下的当前特征的所有值
        feat_list = [example[i] for example in dataSet]
        # 去除重复元素，取得对当前特征而言的所有类别
        unique_vals = set(feat_list)
        new_entropy = 0.0
        # 计算每种划分方式的信息熵
        for value in unique_vals:
            # 获得对当前特征而言，值为 value 的数据子集
            sub_data_set = splitDataSet(dataSet, i, value)

            # 获得符合要求的数据子集占中数据集个数的比重
            prob = len(sub_data_set) / float(len(dataSet))

            # 计算在特征属性 i 的条件下样本的条件熵：∑(当前特征值的比重 * 当前特征值的信息熵)
            new_entropy += prob * calcShannonEnt(sub_data_set)

        # 特征属性 i 的信息增益 = 信息熵 - 特征属性 i 的条件熵
        info_gain = base_entropy - new_entropy

        # 判断是否为最佳信息增益，若是，则更新最大信息增益和最佳特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majorityCnt(classList):
    """
    获得出现频次最高的分类名称
    :param classList: 分类名称列表
    :return: 出现频次最高的分类名称
    """
    class_count = {}
    for vote in classList:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 按出现次数对类别进行排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 出现频次最高的分类名称，sorted_class_count 是一个列表，列表内包含若干个元组，元组组成为(类别,数量)
    return sorted_class_count[0][0]


def createTree(dataSet, labels):
    """
    递归构建决策树
    -----------------
    终止条件一：数据(子)集的类别是一样的时候，直接返回该类别
    终止条件二：如果特征都分完了，那么返回现在类别列表中出现次数最多的类别
    -----------------
    构建子树时，每个特征为一个结点，每个特征值为一个子树
    
    :param dataSet: 数据集或者经过划分的数据子集
    :param labels: 类别向量
    :return: 构建好的决策树
    """
    # 获得所有类别
    class_list = [example[-1] for example in dataSet]

    # 递归终止条件一
    # 类别完全相同则停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 递归终止条件二
    # 遍历完所有特征值时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(class_list)

    # 取得最佳特征
    best_feat = chooseBestFeatureToSplit(dataSet)

    # 取得最佳特征的类别
    best_feat_label = labels[best_feat]

    # 构建树
    my_tree = {best_feat_label: {}}

    # 删除最佳特征
    del (labels[best_feat])

    # 获得所有特征值
    feat_values = [example[best_feat] for example in dataSet]

    # 特征值去重
    unique_vals = set(feat_values)
    for value in unique_vals:
        # 这是去掉当前最佳特征值后的类别向量
        sub_labels = labels[:]
        # 继续构建子树，每一个特征值为一棵子树
        # 这里开始递归，传入不同特征值划分后的数据子集(即best_feat = value的情况)和类别向量
        my_tree[best_feat_label][value] = createTree(splitDataSet(dataSet, best_feat, value), sub_labels)
    return my_tree


def classify(inputTree, featLabels, testVec):
    """
    分类
    :param inputTree: 构建好的决策树
    :param featLabels: 特征类别
    :param testVec: 测试向量
    :return: 类别
    """
    # 获得决策(子)树根节点，因为是字典存储，所以拿到第一个元素即可
    first_str = list(inputTree)[0]

    # 拿到决策(子)树的子树，即决策树的第二层
    second_dict = inputTree[first_str]

    # 取得特征类别的下标，即该下标下的所有值都是该特征的特征值
    feat_index = featLabels.index(first_str)

    # 开始对子树进行操作
    for key in second_dict.keys():
        # key 是该特征下的所有特征值，当匹配到与一个特征对应的情况下，将继续往下操作
        # 如果没有对应上，说明构建决策树时数据量不够，树构建的不完整
        if testVec[feat_index] == key:
            # 找到一个分支过后，判断子树的结点是什么类型
            # 如果为字典类型，则说明还可以继续判断(判断模块)，则继续递归
            # 否则说明判断终止(终止模块)，直接输出终止模块的类型即可
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], featLabels, testVec)
            else:
                class_label = second_dict[key]
    return class_label


def storeTree(inputTree, filename):
    """
    将构建好的二叉树存起来，防止每次分类都构建
    :param inputTree: 要存储的决策树
    :param filename: 存储决策树的文件名
    :return: 
    """
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    """
    获取在磁盘中存好的决策树
    :param filename: 存储决策树的文件名
    :return: 存储的决策树
    """
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)
