from DL.perceptron.perceptron import Perceptron


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, lambda x: x)
