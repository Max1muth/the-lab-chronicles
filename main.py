import random as rn
import this

rn.seed(121)
print(rn.randint(1, 15))  # выпало 2 => задание:
# Ученики зашифровали свои записки, записывая все слова наоборот.
# Составьте программу, зашифровывающую и зашифровывающую сообщение.
k = 5
l1st = []
print(ord("A"), ord("Z"), ord("a"), ord("z"))
for i in range(ord("A"), ord("Z")+1):
    print(chr(i))
    l1st += [chr(i)]
for i in range(26):
    print(chr(ord("a") + ((i+k) % 26)))
    l1st += [chr(i)]


class Encryptor:
    def __init__(self, text: str):
        self.letters = []
        self.text = text
        for step in range(26):
            self.letters += [chr(ord("A") + step)]
        for step in range(26):
            self.letters += [chr(ord("a") + step)]

    def offset_coding(self, offset=0):
        coded = ""
        for char in self.text:
            if char in self.letters[0:26]:
                j = (ord(char)-ord("A") + offset) % 26
                coded += chr(ord("A") + j)
            elif char in self.letters[26:]:
                j = (ord(char)-ord("a") + offset) % 26
                coded += chr(ord("a") + j)
            else:
                coded += char
        return coded

    def offset_encoding(self, offset=0):
        encoded = ""
        for char in self.text:
            if char in self.letters[0:26]:
                j = (ord(char) - ord("A") - offset) % 26
                encoded += chr(ord("A") + j)
            elif char in self.letters[26:]:
                j = (ord(char) - ord("a") - offset) % 26
                encoded += chr(ord("a") + j)
            else:
                encoded += char
        return encoded


f = Encryptor("Hello world!")

print(f.letters)

s = f.offset_coding(1)
d = f.offset_encoding(1)
print(f.text)


print(s)
print(d)


# print(chr(97 + (4 % 26)))
#
# j = int(97 % 98)
# print(chr(int(97 % 98)))
# print(chr(j))
# print(chr(97))
# print(l1st, - ord("A") + ord("Z"))

# print(this.s)
# s = ""
# d = ""
# for i in str(this.s):
#     if i != str:
#         s += i
#         d += i
#     else:
#         s += i
#         d += chr(ord(i) + 13)
# print(s, d)

# print(this.d)
