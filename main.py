# Напишите программу, строящую график функции. Коэффициенты a,b,c и диапазон задаются с клавиатуры.
# f(x)=a·sin(x)-cos(b·x)

import math
import matplotlib.pyplot as plt

a, b, length = int(input("Введите a: ")), int(input("Введите b: ")), int(input("Введите длину: "))
x, y = [], []

for i in range(length):
    x += [i]
    y += [round(a*math.sin(i) - math.cos(b*i), 3)]  # в радианах ( math.degress(11) - перевод в градусы ) 

# print(x, y)
# print(11, math.sin(11))
# print(math.degrees(11), math.sin(math.degrees(11)))
# print(math.sin(math.radians(30)))

fig, ax = plt.subplots()

ax.plot(x, y, "--o", color="#362c1d")
fig.patch.set_facecolor("#ABCA80")
ax.set_facecolor("#B6DA83")
plt.title("f(x)=a·sin(x)-cos(b·x)")
plt.xlabel("x", color="#4E7B02")
plt.ylabel("y", color="#4E7B02", rotation=0)
plt.grid(True)
plt.show()

# Напишите программу построения графика по имеющемуся дискретному набору известных значений. Для этого:
# Данные по выданному варианту поместите в файл. 
# С помощью программы прочитайте их.
# Постройте график в Python.
# Проведите проверку решения задачи в Excel.

import pandas as pd
import json
# x = [255, 301, 402, 477, 627, 777, 927, 1077, 1227, 1527, 1827]
# y = [11, 7.9, 6, 5.4, 4.5, 4, 5, 8, 10, 2.5, 2]
# data = [{"x": x, "y": y}]
# with open("data.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False)

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)
# print(df.iloc[0, 0], df.iloc[0, 1])

fig, ax = plt.subplots()
plt.plot(list(df.iloc[0, 0]), list(df.iloc[0, 1]), "--o", color="#362c1d")
fig.patch.set_facecolor("#ABCA80")
ax.set_facecolor("#B6DA83")
plt.title("second_graphic")
plt.xlabel("x", color="#4E7B02")
plt.ylabel("y", color="#4E7B02", rotation=0)
plt.grid(True)
plt.show()
