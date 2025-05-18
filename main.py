from neural_network.MLP import MLP
from neural_network.Value import Value, draw_graph


SHAPE = [3, 4, 4, 1]

mlp = MLP(SHAPE)

print(mlp)

inputs = [Value(1.0, label="i1"), Value(2.0, label="i2"), Value(3.0, label="i3")]


o = mlp.forward(inputs)[0].label("out")
y = 0.0

loss = (y - o) ** 2
loss.label("loss").backward()
dot = draw_graph(loss)
dot.render("graph", view=True)
print(o)
