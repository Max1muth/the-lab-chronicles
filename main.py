import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import defaultdict

def generate_data(num_points_per_cluster=20, centers=None, radii=None):
    """
    Генерирует исходные данные для кластеризации.
    Точки генерируются внутри колец с заданными центрами и радиусами.

    Args:
        num_points_per_cluster (int): Количество точек для каждого кластера.
        centers (list of tuples): Список кортежей (x, y) для центров кластеров.
                                  По умолчанию: [(2, 2), (8, 3), (4, 9)].
        radii (list of float): Список радиусов для каждого кластера.
                               По умолчанию: [1.5, 1.0, 1.8].

    Returns:
        tuple: Списки x и y координат всех сгенерированных точек.
    """
    if centers is None:
        centers = [(2, 2), (8, 3), (4, 9)]  # Произвольные центры
    if radii is None:
        radii = [1.5, 1.0, 1.8]  # Произвольные радиусы

    all_x = []
    all_y = []

    for i, (cx, cy) in enumerate(centers):
        r = radii[i]
        for _ in range(num_points_per_cluster):
            angle = 2 * np.pi * np.random.rand()
            # Генерируем точки внутри круга, а не строго на окружности
            current_r = r * np.sqrt(np.random.rand())
            x = cx + current_r * np.cos(angle)
            y = cy + current_r * np.sin(angle)
            all_x.append(x)
            all_y.append(y)
    return all_x, all_y

def euclidean_distance(point1, point2):
    """Вычисляет евклидово расстояние между двумя точками."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def kmeans(x, y, k=3, max_iterations=100):
    """
    Реализует метод кластеризации k-средних.

    Args:
        x (list): Список координат по оси x исходных точек.
        y (list): Список координат по оси y исходных точек.
        k (int): Количество искомых кластеров.
        max_iterations (int): Максимальное количество итераций для алгоритма.

    Returns:
        tuple:
            - all_labels_history (list of lists): История меток кластеров для каждой точки
                                                  на каждой итерации.
            - all_centroids_history (list of lists): История координат центроидов кластеров
                                                     на каждой итерации.
    """
    points = np.array(list(zip(x, y)))
    num_points = len(points)

    # Инициализация центроидов: случайно выбираем k точек из исходных данных
    random_indices = np.random.choice(num_points, k, replace=False)
    centroids = points[random_indices].tolist()

    all_labels_history = []
    all_centroids_history = []

    for iteration in range(max_iterations):
        # Шаг 1: Присвоение точек к ближайшим центроидам
        labels = []
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            labels.append(closest_centroid)
        all_labels_history.append(labels)
        all_centroids_history.append(centroids)

        # Шаг 2: Обновление центроидов
        new_centroids = []
        for i in range(k):
            # Находим все точки, принадлежащие текущему кластеру
            cluster_points = [points[j] for j, label in enumerate(labels) if label == i]
            if cluster_points:
                new_centroids.append(np.mean(cluster_points, axis=0).tolist())
            else:
                # Если кластер пуст, оставляем центроид на прежнем месте или переинициализируем
                new_centroids.append(centroids[i])

        # Проверка на сходимость
        if np.allclose(centroids, new_centroids):
            print(f"Алгоритм сошелся на итерации {iteration + 1}")
            break
        centroids = new_centroids

    # Добавляем финальное состояние после возможного выхода по сходимости
    if iteration + 1 < max_iterations: # Если не достигли max_iterations
        labels = []
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            labels.append(closest_centroid)
        all_labels_history.append(labels)
        all_centroids_history.append(centroids)

    return all_labels_history, all_centroids_history

# Генерация данных
x_data, y_data = generate_data(num_points_per_cluster=20,
                               centers=[(2, 2), (8, 3), (4, 9)],
                               radii=[1.5, 1.0, 1.8])

# Применение k-средних
k_clusters = 3
labels_history, centroids_history = kmeans(x_data, y_data, k=k_clusters)

# Настройка графика
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)

# Начальное отображение (первая итерация)
scatter = ax.scatter(x_data, y_data, c=labels_history[0], cmap='viridis', s=50, alpha=0.8)
centroids_plot, = ax.plot([c[0] for c in centroids_history[0]],
                          [c[1] for c in centroids_history[0]],
                          'X', markersize=10, color='red', markeredgecolor='black', label='Centroids')

ax.set_title(f'K-Means Clustering (Iteration 0)')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.grid(True)
ax.legend()

# Создание ползунка
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Iteration', 0, len(labels_history) - 1, valinit=0, valstep=1)

# Функция обновления графика при изменении ползунка
def update(val):
    iteration = int(slider.val)
    current_labels = labels_history[iteration]
    current_centroids = centroids_history[iteration]

    # Обновление цвета точек
    scatter.set_array(current_labels)

    # Обновление положения центроидов
    centroids_plot.set_data([c[0] for c in current_centroids],
                            [c[1] for c in current_centroids])

    ax.set_title(f'K-Means Clustering (Iteration {iteration})')
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
