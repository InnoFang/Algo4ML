from DeepLearning.perceptron.perceptron import Perceptron


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        """
        初始化线性单元，线性单元模型与感知器模型的不同点就在于激活函数的不同
        :param input_num: 输入参数个数
        """
        Perceptron.__init__(self, input_num, lambda x: x)
