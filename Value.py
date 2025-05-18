from typing import List, Union


class Value:
    def __init__(self, data: int | float, label="", _op="", parents=[]):
        self.data = data
        self.label = label
        self._op = _op
        self.parents: List[Value] = parents

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


a = Value(1)
b = Value(2)
print(2 + a)
print(2 - a)
print(a - 2)
print(a * 2)
print(a * b)
print(a**b)
print(b / a)
print(a / 2)
print(2 / a)
