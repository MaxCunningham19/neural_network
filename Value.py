import math
from typing import List, Union
from graphviz import Digraph


class Value:
    def __init__(self, data: int | float, label="", _op="", parents=[], _backprop=lambda: None):
        self.data = data
        self._label = label
        self._op = _op
        self.parents: List[Value] = parents
        self.grad = 0.0
        self._backprop = _backprop

    def __str__(self):
        return f"{self._label if self._label != ""  else"Value"}(data={self.data})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = Value(self.data + other.data, _op="+", parents=[self, other])

        def _backprop():
            other.grad += n.grad
            self.grad += n.grad

        n._backprop = _backprop

        return n

    def __radd__(self, other: Union["Value", int, float]):
        return self + other

    def __sub__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = Value(self.data - other.data, _op="-", parents=[self, other])

        def _backprop():
            other.grad += n.grad
            self.grad += n.grad

        n._backprop = _backprop

        return n

    def __rsub__(self, other: Union["Value", int, float]):
        return self - other

    def __mul__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = Value(self.data * other.data, _op="*", parents=[self, other])

        def _backprop():
            print(n.grad)
            other.grad += n.grad * self.data
            self.grad += n.grad * other.data

        n._backprop = _backprop

        return n

    def __rmul__(self, other: Union["Value", int, float]):
        return self * other

    def __truediv__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = Value(self.data / other.data, _op="/", parents=[self, other])

        def _backprop():
            self.grad += n.grad * (1 / other.data)
            other.grad += n.grad * (-self.data / (other.data**2))

        n._backprop = _backprop

        return n

    def __rtruediv__(self, other: Union["Value", int, float]):
        return self / other

    def __pow__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = Value(self.data**other.data, _op="^", parents=[self, other])

        def _backprop():
            print(n.grad, other.data, self.data)
            self.grad += n.grad * other.data * (self.data ** (other.data - 1))
            print(self.grad)
            print(n.grad, self.data, other.data, math.log(self.data))
            other.grad += n.grad * (self.data**other.data) * math.log(self.data)
            print(other.grad)

        n._backprop = _backprop

        return n

    def __rpow__(self, other: Union["Value", int, float]):
        return self**other

    def exp(self):
        n = Value(math.exp(self.data), _op="e^", parents=[self])

        def _backprop():
            self.grad += n.grad * self.data * (math.exp(self.data - 1))

        n._backprop = _backprop

        return n

    def zero_grad(self):
        """Set all the grad to zero for this node and recursivly set the gradient to zero for each parent"""
        self.grad = 0.0
        for p in self.parents:
            p.zero_grad()

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v.parents:
                    build_topo(p)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backprop()

    def label(self, s: str) -> "Value":
        self._label = s
        return self


def trace(root: Value):
    """Trace all nodes and edges starting from the root node."""
    nodes = set()
    edges = []

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for parent in v.parents:
                edges.append((parent, v))
                build(parent)

    build(root)
    return nodes, edges


def draw_graph(root: Value):
    """Draw a graph of the calculation to achieve the value"""
    dot = Digraph(format="png", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)

    for n in nodes:
        label_text = n._label if n._label else ""
        value_text = f"{n.data:.4f}"
        gradient_text = f"{n.grad:.4f}"
        node_label = f"{{{label_text}|V:{value_text}|G:{gradient_text}}}"
        dot.node(str(id(n)), label=node_label, shape="record")

        if n._op:
            op_id = f"{id(n)}_{n._op}"
            dot.node(op_id, n._op, shape="circle", fixedsize="true", width="0.4")
            dot.edge(op_id, str(id(n)))

            for parent in n.parents:
                dot.edge(str(id(parent)), op_id)

    return dot


if __name__ == "__main__":
    # Example usage
    a = Value(1.2, label="a")
    b = Value(0.9, label="b")
    c = (a + b).label("c")
    d = (c * 2).label("d")
    e = (d / a).label("e")
    f = e.exp().label("f")
    g = Value(-1, label="g")
    o = (f**g).label("o")
    o.backward()

    dot = draw_graph(o)
    dot.render("graph", view=True)
