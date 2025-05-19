import random
from typing import List, Union
from enum import Enum
from .Value import Value


class NonLinear(Enum):
    """Defines which non-linearity to use"""

    ReLU = 1
    LeReLU = 2


class Neuron:
    """Implements a simple single Neuron for a neural net"""

    def __init__(self, num_inputs: int, nonlinear: Union[NonLinear, None] = None, label=""):
        self.weights = [Value(random.uniform(-1.0, 1.0), label="w") for i in range(num_inputs)]
        self.bias = Value(random.uniform(-1.0, 1.0), label="b")
        self.nonlinear = nonlinear

    def __call__(self, x):
        inp = sum(((wi * xi) for wi, xi in zip(self.weights, x)), self.bias)
        match self.nonlinear:
            case NonLinear.ReLU:
                return inp.relu()
            case NonLinear.LeReLU:
                return inp.leaky_relu()
            case _:
                return inp

    def params(self):
        return self.weights + [self.bias]

    def forward(self, x: List[float | int]):
        return self.__call__(x)
