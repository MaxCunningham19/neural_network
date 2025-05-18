from typing import List, Union
from graphviz import Digraph


class Value:
    def __init__(self, data: int | float, label="", _op="", parents=[]):
        self.data = data
        self.label = label
        self._op = _op
        self.parents: List[Value] = parents
        self.grad = 0.0

    def __str__(self):
        return f"{self.label if self.label != ""  else"Value"}(data={self.data})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = self.data + other.data
        return Value(n, _op="+", parents=[self, other])

    def __radd__(self, other: Union["Value", int, float]):
        return self + other

    def __sub__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = self.data - other.data
        return Value(n, _op="-", parents=[self, other])

    def __rsub__(self, other: Union["Value", int, float]):
        return self - other

    def __mul__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = self.data * other.data
        return Value(n, _op="*", parents=[self, other])

    def __rmul__(self, other: Union["Value", int, float]):
        return self * other

    def __truediv__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = self.data / other.data
        return Value(n, _op="/", parents=[self, other])

    def __rtruediv__(self, other: Union["Value", int, float]):
        return self / other

    def __pow__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = self.data**other.data
        return Value(n, _op="^", parents=[self, other])

    def __rpow__(self, other: Union["Value", int, float]):
        return self**other


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
        label_text = n.label if n.label else ""
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
    a = Value(2, label="a")
    b = Value(3, label="b")
    c = a + b
    c.label = "c"
    d = c * 4
    d.label = "d"
    e = d / a
    e.label = "e"

    dot = draw_graph(e)
    dot.render("graph", view=True)
