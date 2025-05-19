import pandas as pd

for i in range(2**4):
    a, b, c, d = 0, 0, 0, 0


def arrays(length):
    if length <= 0:
        return [[]]
    arrs = arrays(length - 1)
    retr = []
    for arr in arrs:
        for i in range(2):
            retr.append([i] + arr)
    return retr


xs = arrays(4)
ys = []
for x in xs:
    y0, y1 = x[0] == 1 and x[1] == 1, x[2] == 1 and x[3] == 1
    y2, y3 = y0 and y1, y0 or y1
    y4 = y3 and not y2

    ys.append([y0, y1, y2, y3, y4])
    for i in range(len(ys[-1])):
        ys[-1][i] = int(ys[-1][i])

cols = []
for i in range(len(xs[0])):
    cols.append(f"x{i}")

for i in range(len(ys[0])):
    cols.append(f"y{i}")

data = []
for ix in range(len(xs)):
    data.append(xs[ix] + ys[ix])

df = pd.DataFrame(data, columns=cols)


df.to_csv("data_tmp.csv", index=False)
