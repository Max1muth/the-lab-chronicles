import random as rn

rn.seed(121)
print(rn.randint(1, 15))  # выпало 2 => задание:
# Ученики зашифровали свои записки, записывая все слова наоборот.
# Составьте программу, зашифровывающую и зашифровывающую сообщение.

note = """
Ash nazg durbatulûk,
Ash nazg gimbatul,
Ash nazg thrakatulûk,
Agh burzum-ishi krimpatul.
"""


def shuffler(text: str):
    n_text = [""]
    w0rd = ""
    for char in text:
        if char.isalpha():
            w0rd += char
        else:
            if w0rd:
                n_text.append(w0rd[::-1])
                w0rd = ""
            n_text.append(char)
    if w0rd:
        n_text.append(w0rd[::-1])
    return "".join(n_text)


coded = shuffler(note)
print(coded)
encoded = shuffler(coded)
print(encoded)
