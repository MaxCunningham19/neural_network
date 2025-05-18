from neural_network.MLP import MLP


SHAPE = [3, 4, 4, 1]

mlp = MLP(SHAPE)

print(mlp)

o = mlp.forward([1.0, 2.0, -1.0])[0]
y = 0.0


print(o)
