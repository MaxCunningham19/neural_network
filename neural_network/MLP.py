from typing import List

from .Value import Value
from .Neuron import NonLinear
from .Layer import Layer


class MLP:
    """A fully connected MLP"""

    def __init__(self, shape: List[int]):
        self.layers = []
        self.shape = shape
        for i, x in enumerate(shape):
            if i < len(shape) - 2:
                self.layers.append(Layer(x, shape[i + 1], nonlinear=NonLinear.LeReLU))
            elif i < len(shape) - 1:
                self.layers.append(Layer(x, shape[i + 1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x) -> List[Value]:
        return self.__call__(x)

    def params(self):
        return [l.params() for l in self.layers]

    def __str__(self):
        return f"MLP with shape: {self.shape}"

    def __repr__(self):
        return self.__str__()
