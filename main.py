import random as rn

rn.seed(221)
print(rn.randint(1, 20))  # => 8 поменять местами максимальный элемент
# массива и минимальный элемент части массива, расположенной после максимального.

# text = "какой-то текст"
# with open("text.txt", "w", encoding="utf-8") as f:
#     f.write(text)

with open("text.txt", "r", encoding="utf-8") as f:
    t3xt = f.read()

j = list(t3xt)
for i in range(len(j)):
    j[i] = ord(j[i])
m1, m1i = max(j), j.index(max(j))
m2, m2i = min(j[j.index(m1):]), j.index(min(j[j.index(m1):]))
# [j.find(m1):]
print(m1, m2)
print(j)
j[m1i], j[m2i] = m2, m1
print(j)

for i in range(len(j)):
    j[i] = chr(j[i])

print("".join(j))
