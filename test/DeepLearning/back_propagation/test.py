from DeepLearning.back_propagation.bp import *
from functools import reduce
from numpy import *


def mean_square_error(vec1, vec2):
    """
    计算平均平方误差
    :param vec1: 第一个数
    :param vec2: 第二个数
    :return: 返回 1 / 2 * (x-y)^2 计算得到的值
    """
    return 0.5 * reduce(lambda a, b: a + b, map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2)))


def gradient_check(network, sample_feature, sample_label):
    """
    梯度检查
    :param network: 神经网络对象
    :param sample_feature: 样本的特征
    :param sample_label: 样本的标签
    """
    # 计算网络误差
    network_error = lambda vec1, vec2: 0.5 * reduce(lambda a, b: a + b,
                                                    map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2)))
    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)

        # 根据式 f(w + ε) - f(w - ε) / 2ε 计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (expected_gradient, actual_gradient))


def train_data_set():
    """
    获取训练数据集
    :return: labels: 训练数据集每条数据对应的标签
    """
    # 调用 Normalizer 类
    normalizer = Normalizer()
    # 初始化一个 list，用来存储后面的数据
    data_set = []
    labels = []
    # 0 到 256，其中以 8 为步长
    for i in range(0, 256, 8):
        # 调用 normalizer 对象的 norm 方法
        n = normalizer.norm(int(random.uniform(0, 256)))
        # 在 data_set 中 append n
        data_set.append(n)
        # 在 labels 中 append n
        labels.append(n)
    # 将它们返回
    return labels, data_set


def train(network):
    """
    使用我们的神经网络进行训练
    :param network: 神经网络对象
    """
    # 获取训练数据集
    labels, data_set = train_data_set()
    labels = list(labels)
    data_set = list(labels)
    # 调用 network 中的 train 方法来训练我们的神经网络
    network.train(labels, data_set, 0.3, 50)


def test(network, data):
    """
    对我们的全连接神经网络进行测试
    :param network: 神经网络对象 
    :param data: 测试数据集
    """
    # 调用 Normalizer 类
    normalizer = Normalizer()
    # 调用 norm 方法，对数据进行规范化
    norm_data = normalizer.norm(data)
    norm_data = list(norm_data)
    # 对测试数据进行预测
    predict_data = network.predict(norm_data)
    # 将结果打印出来
    print('\ttestdata(%u)\tpredict(%u)' % (data, normalizer.denorm(predict_data)))


def correct_ratio(network):
    """
    计算我们的神经网络的正确率
    :param network: 神经网络对象
    :return: 
    """
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def gradient_check_test():
    """
    梯度检查测试
    """
    # 创建一个有 3 层的网络，每层有 2 个节点
    net = Network([2, 2, 2])
    # 样本的特征
    sample_feature = [0.9, 0.1]
    # 样本对应的标签
    sample_label = [0.9, 0.1]
    # 使用梯度检查来查看是否正确
    gradient_check(net, sample_feature, sample_label)


def main():
    # 初始化一个神经网络，输入层 8 个节点，隐藏层 3 个节点，输出层 8 个节点
    network = Network([8, 3, 8])
    # 训练我们的神经网络
    train(network)
    # 将我们的神经网络的信息打印出来
    network.dump()
    # 打印出神经网络的正确率
    correct_ratio(network)


if __name__ == '__main__':
    main()
