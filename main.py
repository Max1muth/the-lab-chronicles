import random as rn
import pandas as pd
import matplotlib.pyplot as plt

rn.seed(521)
tasks = [rn.randint(1, 11), rn.randint(1, 11), rn.randint(1, 11)] # 2, 1, 5
print(tasks)

df = pd.read_csv("data.csv")
j = df.iloc[0, 0]
print(eval(j)[1])
x11, y11 = [], []
x12, y12 = [], []
x13, y13 = [], []
for i in range(len(df.iloc[0])):
    j1, j2, j3 = df.iloc[0, i], df.iloc[1, i], df.iloc[2, i]
    x11 += [eval(j1)[0]]
    y11 += [eval(j1)[1]]
    x12 += [eval(j2)[0]]
    y12 += [eval(j2)[1]]
    x13 += [eval(j3)[0]]
    y13 += [eval(j3)[1]]
print(x11, y11)

plt.scatter(x11, y11, color="red")
plt.scatter(x12, y12, color="blue")
plt.scatter(x13, y13, color="green")
plt.show()
