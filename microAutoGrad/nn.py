import random
from typing import List

from microAutoGrad import Value


class Neuron:
    def __init__(self, n_in: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0)

    def __call__(self, x, *args, **kwargs):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()


class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x, *args, **kwargs):
        outs = [neuron(x) for neuron in self.neurons]
        return outs if len(outs) > 1 else outs[0]


class MLP:
    def __init__(self, n_in: int, n_outs: List[int]):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x
