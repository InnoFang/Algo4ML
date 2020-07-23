class Perceptron:
    def __init__(self, input_num, activator):
        """
        初始化感知机，设置输入参数的个数，以及激活函数
        激活函数的类型 double -> double
        """
        self.activator = activator
        # 权重向量初始化为 0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏执项初始化为 0
        self.bias = 0.0

    def __str__(self):
        """
        打印学习到的权重、偏置项
        """
        return 'weights\t:%s\nbias:\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        输入向量，输出感知器的计算结果
        -----------------------------
        把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        变成[(x1,w1),(x2,w2),(x3,w3),...]
        对input_vec和weights加权求和
        """
        return self.activator(sum([x * w for (x, w) in zip(input_vec, self.weights)]) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        """
        :param input_vecs: 特征向量
        :param labels: 标签
        :param iteration: 迭代次数
        :param rate: 学习率
        :return:
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        """
        一次迭代，把所有的训练数据过一遍
        """
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        """
        按照感知器规则更新权重
        """
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = list(map(lambda tp: tp[1] + rate * delta * tp[0], zip(input_vec, self.weights)))
        # 更新 bias
        self.bias += rate * delta
