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
    n_text = ""
    l1st = text.split(" ")  # разделяет текст на список потенциальных слов
    for char1 in l1st:  # перебор по таких слов по списку
        if "\n" in char1:
            position = [i for i, n in enumerate(char1) if n == "\n"]  # список индексов встречи \n
            print(position)
            l2st = char1.split("\n")
            l2st = [x for x in l2st if x != ""]  # удаление лишних ""
            char2 = ""
            char3 = ""
            for i in range(len(position)):
                print(l2st, 99999999)
                char2 += (l2st[i])
                print(char2, 66666)
                for k in range(len(char2)-1, -1, -1):
                    char3 += char2[k]
                print(char3, 77777)
            n_text += "\n" + char3 + " "
        else:
            n_text += char1
        # while char1.index("\n") != ValueError:
        #     print(char1.find("\n"))
        #     print(char1[:int(char1.find("\n"))] + char1[int(char1.find("\n")):])
    print(n_text)
    return l1st


print(shuffler(note))
