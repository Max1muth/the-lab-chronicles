import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random

# 1. Инициализация исходных данных
def generate_data(pointsCount1=50, pointsCount2=50):
    # Границы для первого класса
    xMin1, xMax1 = -5, 0
    yMin1, yMax1 = -5, 0
    # Границы для второго класса
    xMin2, xMax2 = 0, 5
    yMin2, yMax2 = 0, 5
    
    # Генерация точек для первого класса
    x1 = [random.uniform(xMin1, xMax1) for _ in range(pointsCount1)]
    y1 = [random.uniform(yMin1, yMax1) for _ in range(pointsCount1)]
    class1 = [[x1[i], y1[i]] for i in range(pointsCount1)]
    
    # Генерация точек для второго класса
    x2 = [random.uniform(xMin2, xMax2) for _ in range(pointsCount2)]
    y2 = [random.uniform(yMin2, yMax2) for _ in range(pointsCount2)]
    class2 = [[x2[i], y2[i]] for i in range(pointsCount2)]
    
    # Объединение данных
    x = np.array(class1 + class2)
    y = np.array([0] * pointsCount1 + [1] * pointsCount2)
    
    return x, y

# 2. Разбиение данных на обучающую и тестовую выборки
def train_test_split(x, y, p=0.8):
    np.random.seed(42)  # Для воспроизводимости
    indices = np.random.permutation(len(x))
    train_size = int(len(x) * p)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return x_train, x_test, y_train, y_test

# 3. Реализация метода k ближайших соседей
def fit(x_train, y_train, x_test, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    return y_predict

# 4. Расчёт метрики accuracy
def computeAccuracy(y_test, y_predict):
    correct = np.sum(y_test == y_predict)
    total = len(y_test)
    accuracy = correct / total
    return accuracy

# 5. Визуализация результатов
def plot_results(x_train, y_train, x_test, y_test, y_predict):
    plt.figure(figsize=(10, 8))
    
    # Точки обучающей выборки
    plt.scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1], 
                c='blue', marker='o', label='Train Class 0', s=50)
    plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], 
                c='blue', marker='x', label='Train Class 1', s=50)
    
    # Точки тестовой выборки
    for i in range(len(x_test)):
        if y_test[i] == y_predict[i]:
            # Верно классифицированные точки
            color = 'green'
            marker = 'o' if y_test[i] == 0 else 'x'
        else:
            # Неверно классифицированные точки
            color = 'red'
            marker = 'o' if y_test[i] == 0 else 'x'
        plt.scatter(x_test[i, 0], x_test[i, 1], c=color, marker=marker, s=50)
    
    # Подписи для тестовых точек
    plt.scatter([], [], c='green', marker='o', label='Test Class 0 (Correct)')
    plt.scatter([], [], c='green', marker='x', label='Test Class 1 (Correct)')
    plt.scatter([], [], c='red', marker='o', label='Test Class 0 (Incorrect)')
    plt.scatter([], [], c='red', marker='x', label='Test Class 1 (Incorrect)')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'KNN Classification (k=3, Accuracy: {computeAccuracy(y_test, y_predict):.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()

# 6. Основной код
if __name__ == "__main__":
    # Генерация данных
    x, y = generate_data()
    
    # Разбиение данных
    x_train, x_test, y_train, y_test = train_test_split(x, y, p=0.8)
    
    # Применение KNN
    y_predict = fit(x_train, y_train, x_test, k=3)
    
    # Вычисление и вывод точности
    accuracy = computeAccuracy(y_test, y_predict)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Визуализация
    plot_results(x_train, y_train, x_test, y_test, y_predict)
