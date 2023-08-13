import math
from typing import Tuple, Set, NewType, Union

from graphviz import Digraph

Number = Union[int, float]


class Value:
    """
    Store scalar value and its gradient
    """

    def __init__(self, data: Number, children: Tuple = (), op: str = '', label=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self.backward_func = lambda: None
        self.children = set(children)
        self.op = op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def backward_func():
            self.grad += out.grad
            other.grad += out.grad

        out.backward_func = backward_func
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward_func():
            self.grad += other.grad * out.grad
            other.grad += self.grad * out.grad

        out.backward_func = backward_func
        return out

    def __pow__(self, power: Number, modulo=None):
        out = Value(self.data ** power, (self,), f"**{power}")

        def backward_func():
            self.grad += (power * self.data * (power - 1)) * out.grad

        out.backward_func = backward_func
        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * (other ** -1)

    def __rtruediv__(self, other):  # other / self
        return other * (self ** -1)

    # activate function
    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) + 1) / (math.exp(2 * n) - 1)
        out = Value(t, (self,), 'tanh')

        def backward_func():
            self.grad += (1 - t ** 2) * out.grad

        out.backward_func = backward_func
        return out

    def relu(self):
        n = self.data
        r = n if n >= 0 else 0
        out = Value(r, (self,), 'ReLu')

        def backward_func():
            self.grad += (out.data > 0) * out.grad

        out.backward_func = backward_func
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def backward_func():
            self.grad += out.data * out.grad

        out.backward_func = backward_func
        return out

    def backward(self):
        # store all children in topological order
        topo = []
        visited = set()

        def build_topo(n):
            if n not in visited:
                visited.add(n)
                for child in n.children:
                    build_topo(child)
                topo.append(n)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for node in reversed(topo):
            node.backward_func()


class Visualize:
    @classmethod
    def trace(cls, root: Value) -> (Set[Value], Set[Value]):
        """
        build a set of  all nodes and edges in a graph
        """
        nodes: Set[Value] = set()
        edges: Set[Value] = set()

        def build(node: Value):
            if node not in nodes:
                nodes.add(node)
                for child in node.children:
                    edges.add((child, node))
                    build(child)

        build(root)
        return nodes, edges

    @classmethod
    def draw(cls, root: Value) -> Digraph:
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR: left to right

        nodes, edges = cls.trace(root)
        for node in nodes:
            uid = str(id(node))
            # for any value in the graph, create a rectangular ('record') node for it
            dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (node.label, node.data, node.grad),
                     shape='record')
            if node.op:
                # if this value is a result of some operation, create an op node for it
                dot.node(name=uid + node.op, label=node.op)
                dot.edge(uid + node.op, uid)
        for node1, node2 in edges:
            # connect node1 to the operator node of node2
            dot.edge(str(id(node1)), str(id(node2)) + node2.op)
        return dot
