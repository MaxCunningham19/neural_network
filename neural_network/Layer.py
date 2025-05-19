from typing import List
from neural_network.Value import Value
from .Neuron import Neuron


class Layer:
    def __init__(self, num_inputs, num_nodes, **kwargs):
        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_nodes)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def params(self) -> List[List[Value]]:
        return [n.params() for n in self.neurons]
