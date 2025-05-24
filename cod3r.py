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


f = Encryptor("Hello world!")  # класс запоминает введенный текст
s1 = f.offset_coding(1)  # задает смещение вправо для букв английского алфавита
s2 = f.offset_encoding(1)  # задает смещение влево для букв английского алфавита
print(s1)  # => Ifmmp xpsme!
ss = Encryptor(s1)
ss1 = ss.offset_encoding(1)
print(ss1)  # => Hello world!
