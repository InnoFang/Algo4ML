import numpy as np


class _FullConnectedLayer:
    def __init__(self, input_size, output_size, activator):
        """
        全连接层构造函数
        :param input_size: 本层输入向量的维度
        :param output_size: 本层输出向量的维度
        :param activator: 激活函数
        """
        self.activator = activator
        # 权重数组 Weight
        self.W = np.random.randn(input_size, output_size)
        # 偏置项 bias
        self.b = np.zeros(output_size)
        # 输入向量
        self.input = None
        # 输出向量
        self.output = None
        # 反向传播时的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        output = x · W + b
        :param x: 输入向量，维度与 input_size 同
        :return:
        """
        self.input = x
        self.output = self.activator.forward(x)
        return self.output

    def backward(self, delta_array):
        """
        反向计算 W 和 b 的梯度
        :param delta_array: 从上一层传递过来的误差项
        :return:
        """
        delta = self.activator.backward(self.input)
        self.dW = np.dot(delta_array, self.input.T)
        self.db = np.sum(delta_array, axis=0)
        return delta

    def update(self, learning_rate):
        """
        使用梯度下降算法更新权重
        :param learning_rate:
        :return:
        """
        self.W -= self.dW * learning_rate
        self.b -= self.db * learning_rate


class _SigmoidActivator:
    def __init__(self):
        self.out = None

    def forward(self, X):
        out = 1.0 / (1.0 + np.exp(-X))
        self.out = out

        return out

    def backward(self, dout):
        """
        sigmod 的导数：y * ( 1 - y )
        :param dout: 反向传播的上一层的导数
        :return:
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Network:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                _FullConnectedLayer(layers[i], layers[i + 1], _SigmoidActivator()))

    def train(self, X_train, y_train, learning_rate, epoch):
        """
        训练函数
        :param X_train: 输入样本
        :param y_train: 样本标签
        :param learning_rate: 学习率
        :param epoch: 训练论数
        """
        for i in range(epoch):
            for x, y in zip(X_train, y_train):
                self.train_one_sample(x, y, learning_rate)

    def train_one_sample(self, x, y, learning_rate):
        self.predict(x)
        self.calc_gradient()
        self.update_weight(learning_rate)

    def predict(self, x):
        """
        使用神经网络实现预测
        :param x: 输入样本
        :return:
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def calc_gradient(self):
        delta = self.layers[-1].activator.backward(self.layers[-1].output)
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

    def update_weight(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def accuracy(self, X, labels):
        y = self.predict(X)
        y = np.argmax(y, axis=1)
        labels = np.argmax(labels, axis=1)

        accuracy = np.sum(y == labels) / float(X.shape[0])
        return accuracy
