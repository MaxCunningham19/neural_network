import random
import numpy as np
import pandas as pd
import collections
from typing import List
from neural_network.MLP import MLP
from neural_network.Value import Value, draw_graph


def flatten(lst, count=-1):
    if count == 0:
        for item in lst:
            yield item

    for item in lst:
        if isinstance(item, list):
            yield from flatten(item, count - 1)
        else:
            yield item


def read_data(path="./data.csv"):
    df = pd.read_csv(path)
    x_cols = [col for col in df.columns if col.startswith("x")]
    y_cols = [col for col in df.columns if col.startswith("y")]

    return df[x_cols].values, df[y_cols].values


def sample(xs, ys, sample_size=-1, with_replacement=True):
    if sample_size <= 0:
        return xs, ys

    if with_replacement:
        sample_indexs = [random.randint(1, len(xs) - 1) for _ in range(sample_size)]
    else:
        random.sample(range(len(xs)), sample_size)

    return xs[sample_indexs], ys[sample_indexs]


# x_data, y_data = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]], [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
x_data, y_data = read_data()
input_shape = len(x_data[0])
output_shape = len(y_data[0])
SHAPE = [input_shape, input_shape * 2, input_shape * 4, output_shape]
train = 1_000
learning_rate = 0.01
learning_decay = 0.995
learning_min = 0.005
sample_size = -1 if len(x_data) < 50 else 32
epsilon = 0.001
mlp = MLP(SHAPE)

for k in range(train):
    x, y = sample(x_data, y_data, sample_size=sample_size)
    yp = list(flatten([mlp.forward(xi) for xi in x], 0))
    y = y

    losses = []
    for yis, ypis in zip(y, yp):
        for i in range(len(yis)):
            losses.append((yis[i] - ypis[i]) ** 2)
    loss = sum(losses)
    # for i in range(len(losses)):
    # losses[i].backward()
    loss.backward()
    params = list(flatten(mlp.params()))
    for param in params:
        param.data += -learning_rate * param.grad
    loss.zero_grad()
    if k % 10 == 0:
        print(k, loss.data)
    learning_rate = max(learning_min, learning_rate * learning_decay)
    if loss.data < epsilon:
        break


x, y = sample(x_data, y_data, sample_size=-1)
yp = list(flatten([mlp.forward(xi) for xi in x], 0))
y = y
for i in range(len(x)):
    print(i, x[i], y[i], yp[i])
