from typing import Union


class Value:
    def __init__(self, data: int | float):
        self.data = data

    def __str__(self):
        return f"Value(data={self.data})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = self.data + other.data
        return Value(n)

    def __radd__(self, other: Union["Value", int, float]):
        return self + other

    def __sub__(self, other: Union["Value", int, float]):
        return self + (other * -1)

    def __rsub__(self, other: Union["Value", int, float]):
        return self - other

    def __mul__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = self.data * other.data
        return Value(n)

    def __rmul__(self, other: Union["Value", int, float]):
        return self * other

    def __truediv__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)

    def __rtruediv__(self, other: Union["Value", int, float]):
        return self / other

    def __pow__(self, other: Union["Value", int, float]):
        other = other if isinstance(other, Value) else Value(other)
        n = self.data**other.data
        return Value(n)

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
