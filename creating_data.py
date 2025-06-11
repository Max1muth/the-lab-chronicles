import pygame
import pandas as pd
import sys

# Инициализируем Pygame
pygame.init()

a = 4

# Создаем окно
screen = pygame.display.set_mode((200*a, 150*a))

background_color = (39, 74, 32)

# Устанавливаем название окна
pygame.display.set_caption("Нажмите на нужный квадрат")

font = pygame.font.SysFont("Arial", 24)

# Текст, который нужно отобразить
text1 = ["1", "2", "3", "4", "5"]
text2 = ["10", "20", "30", "40", "50"]
cl2ss = ["a", "b", "c", "d", "e"]
colors1 = [(232, 49, 0), (255, 105, 51), (255, 165, 112), (250, 208, 116), (252, 196, 73)]

points = []

flag = 0
j = 0

# Основной цикл программы
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if flag == 2:
                points.append(event.pos)
                # print(points)
                if len(points) <= k:
                    e = list(event.pos) + [f"{cl2ss[j]}"]
                    l1stik[j].append(e)
                    if len(points) == k:
                        # print("where points?")
                        # print(l1stik)
                        j += 1
                        points = []
                        if j == len(l1stik):
                            print(l1stik)
                            # Создание DataFrame
                            df = pd.DataFrame(l1stik, columns=[str(i) for i in range(1, k+1)])
                            # Запись в CSV
                            df.to_csv("data.csv", index=False)
                            pygame.quit()
                            sys.exit()
                            
                pygame.display.set_caption("Кликер")

            else:
                points.append(event.pos)
                print(points)
    screen.fill(background_color)

    for point in points:
        pygame.draw.circle(screen, colors1[j], point, 5)

    if flag == 0:
        text = "Сколько классов нужно в датасете?"
        text_surface = font.render(text, True, (2, 112, 41))
        screen.blit(text_surface, (100, 100))

        for i in range(5):
            pygame.draw.rect(screen, colors1[i], (120+i*80, 160, 50, 50))

            text_ = font.render(text1[i], True, (2, 112, 41))
            screen.blit(text_, (140+i*80, 170))
        for point in points:
            if 160 <= list(point)[1] <= 240:
                if 120 <= list(point)[0] <= 170:
                    l1stik = [[]]
                    flag+=1
                elif 200 <= list(point)[0] <= 250:
                    l1stik = [[], []]
                    flag+=1
                elif 280 <= list(point)[0] <= 330:
                    l1stik = [[], [], []]
                    flag+=1
                elif 360 <= list(point)[0] <= 410:
                    l1stik = [[], [], [], []]
                    flag+=1
                elif 440 <= list(point)[0] <= 490:
                    l1stik = [[], [], [], [], []]
                    flag+=1
            points = []

    if flag == 1:
        text = "Сколько объектов нужно в каждом классе?"
        text_surface = font.render(text, True, (2, 112, 41))
        screen.blit(text_surface, (100, 100))

        for i in range(5):
            pygame.draw.rect(screen, colors1[i], (120+i*80, 160, 50, 50))

            text_ = font.render(text2[i], True, (2, 112, 41))
            screen.blit(text_, (135+i*80, 170))
        for point in points:
            if 160 <= list(point)[1] <= 240:
                if 120 <= list(point)[0] <= 170:
                    k = 10
                    flag+=1
                elif 200 <= list(point)[0] <= 250:
                    k = 20
                    flag+=1
                elif 280 <= list(point)[0] <= 330:
                    k = 30
                    flag+=1
                elif 360 <= list(point)[0] <= 410:
                    k = 40
                    flag+=1
                elif 440 <= list(point)[0] <= 490:
                    k = 50
                    flag+=1
            points = []
    # Обновление экрана
    pygame.display.flip()
    pygame.time.Clock().tick(60)