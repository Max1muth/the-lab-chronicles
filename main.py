# Напишите программу, строящую график функции. Коэффициенты a,b,c и диапазон задаются с клавиатуры.
# f(x)=a·sin(x)-cos(b·x)

import math

a, b, length = int(input("Введите a: ")), int(input("Введите b: ")), int(input("Введите длину: "))
x, y = [], []

for i in range(length):
    x += [i]
    y += [round(a*math.sin(i) - math.cos(b*i), 3)]  # в радианах ( math.degress(11) - перевод в градусы ) 

print(x, y)
print(11, math.sin(11))
print(math.degrees(11), math.sin(math.degrees(11)))
print(math.sin(math.radians(30)))
