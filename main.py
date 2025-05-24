import random as rn

rn.seed(221)
print(rn.randint(1, 20))  # => 8 поменять местами максимальный элемент
# массива и минимальный элемент части массива, расположенной после максимального.

# text = "какой-то текст"
# with open("text.txt", "w", encoding="utf-8") as f:
#     f.write(text)

with open("text.txt", "r", encoding="utf-8") as f:
    t3xt = f.read()

def f3ncti0n(text):
    l1st = list(text)
    for i in range(len(l1st)):
        l1st[i] = ord(l1st[i])
    m1, m1i = max(l1st), l1st.index(max(l1st))
    m2, m2i = min(l1st[l1st.index(m1):]), l1st.index(min(l1st[l1st.index(m1):]))
    l1st[m1i], l1st[m2i] = m2, m1
    for i in range(len(l1st)):
        l1st[i] = chr(l1st[i])
    return "".join(l1st)

print(f3ncti0n(t3xt))
