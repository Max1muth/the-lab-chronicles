from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def make_circles_data(n_samples=1000, shuffle=True, noise=0.05, random_state=None):
    """Генерирует два концентрических круга."""
    X, y = datasets.make_circles(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)
    return X, y

def make_moons_data(n_samples=1000, shuffle=True, noise=0.05, random_state=None):
    """Генерирует два полумесяца."""
    X, y = datasets.make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)
    return X, y

def make_blobs_data(n_samples=1000, n_features=2, centers=3, cluster_std=1.0, random_state=None):
    """Генерирует несколько сферических кластеров."""
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return X, y

def make_anisotropic_data(n_samples=1000, random_state=None):
    """Генерирует вытянутые (асимметричные) кластеры."""
    np.random.seed(random_state)
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X = np.dot(X, transformation)
    return X, y

def make_varied_data(n_samples=1000, random_state=None):
    """Генерирует кластеры с разной плотностью и размером."""
    np.random.seed(random_state)
    n_samples_1 = int(n_samples * 0.5)
    n_samples_2 = int(n_samples * 0.3)
    n_samples_3 = n_samples - n_samples_1 - n_samples_2

    X1, y1 = datasets.make_blobs(n_samples=n_samples_1, centers=[[0, 0]], cluster_std=[0.5], random_state=random_state)
    X2, y2 = datasets.make_blobs(n_samples=n_samples_2, centers=[[3, 3]], cluster_std=[1.5], random_state=random_state)
    X3, y3 = datasets.make_blobs(n_samples=n_samples_3, centers=[[-3, 3]], cluster_std=[0.8], random_state=random_state)

    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2 + 1, y3 + 2]) # Смещаем метки, чтобы они были уникальными
    return X, y

def make_no_structure_data(n_samples=1000, random_state=None):
    """Генерирует случайные точки без явной структуры кластеров."""
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 2) * 10 # Точки в диапазоне [0, 10)
    y = np.zeros(n_samples) # Нет явных кластеров, метки не имеют значения для генерации
    return X, y

# Список генераторов данных
data_generators = {
    "Circles": make_circles_data,
    "Moons": make_moons_data,
    "Blobs": make_blobs_data,
    "Anisotropy": make_anisotropic_data,
    "Varied": make_varied_data,
    "No Structure": make_no_structure_data,
}

from sklearn.cluster import MiniBatchKMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler # Для масштабирования данных

def apply_minibatch_kmeans(X, n_clusters=3, random_state=None):
    """Применяет MiniBatchKMeans."""
    # MiniBatchKMeans чувствителен к масштабу, поэтому масштабируем данные.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    # n_init=10: запускает алгоритм 10 раз с разными центроидами и выбирает лучший результат
    labels = model.fit_predict(X_scaled)
    return labels

def apply_affinity_propagation(X, damping=0.9, preference=-50, random_state=None):
    """Применяет AffinityPropagation."""
    # AffinityPropagation также чувствителен к масштабу.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Параметры damping и preference часто требуют настройки.
    # preference = -50 - это обычно разумное начальное значение для многих датасетов.
    # Большое отрицательное значение уменьшает количество кластеров.
    model = AffinityPropagation(damping=damping, preference=preference, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    return labels

def apply_ward(X, n_clusters=3, random_state=None):
    """Применяет Ward (Agglomerative Clustering)."""
    # Ward чувствителен к масштабу, но часто используется без явного масштабирования
    # для данных, где евклидово расстояние имеет смысл. Однако для последовательности
    # с другими алгоритмами, масштабирование может быть полезным.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # linkage='ward' - это конкретный метод объединения кластеров в AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X_scaled)
    return labels

# Словарь для легкого доступа к функциям кластеризации
clustering_algorithms = {
    "MiniBatchKMeans": apply_minibatch_kmeans,
    "AffinityPropagation": apply_affinity_propagation,
    "Ward": apply_ward,
}

# --- Демонстрация работы алгоритмов и построение таблицы ---

# Количество кластеров для алгоритмов, которые его требуют
# Для AffinityPropagation количество кластеров определяется автоматически
N_CLUSTERS = 3 # Для MiniBatchKMeans и Ward

# Инициализация для воспроизводимости
RANDOM_STATE = 42

fig, axes = plt.subplots(len(data_generators), len(clustering_algorithms) + 1, figsize=(15, 20))
fig.suptitle('Сравнение алгоритмов кластеризации на разных типах данных', fontsize=16, y=0.95)

# Заголовки столбцов
for j, algo_name in enumerate(["Исходные данные"] + list(clustering_algorithms.keys())):
    axes[0, j].set_title(algo_name, fontsize=12)

# Проходим по каждому типу данных
for i, (data_type, data_gen_func) in enumerate(data_generators.items()):
    # 1. Генерируем исходные данные
    X_data, y_true = data_gen_func(n_samples=1000, random_state=RANDOM_STATE)

    # Отображаем исходные данные в первой колонке
    axes[i, 0].scatter(X_data[:, 0], X_data[:, 1], c=y_true, cmap='viridis', s=10)
    axes[i, 0].set_ylabel(data_type, rotation=90, fontsize=12)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])

    # 2. Применяем каждый алгоритм кластеризации
    for j, (algo_name, algo_func) in enumerate(clustering_algorithms.items()):
        current_ax = axes[i, j + 1] # +1 потому что первая колонка для исходных данных

        labels = None
        try:
            # Для MiniBatchKMeans и Ward используем N_CLUSTERS
            if algo_name in ["MiniBatchKMeans", "Ward"]:
                labels = algo_func(X_data, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
            # Для AffinityPropagation используем другие параметры, или значения по умолчанию
            elif algo_name == "AffinityPropagation":
                # Для AffinityPropagation нужно подобрать параметры, иначе он может создать много кластеров
                # или упасть на некоторых данных.
                # Можно попробовать динамически подбирать preference или использовать очень низкое значение.
                # Для стабильности на разных данных, иногда можно использовать более агрессивные параметры.
                if data_type == "Circles" or data_type == "Moons":
                     labels = algo_func(X_data, damping=0.9, preference=-200, random_state=RANDOM_STATE)
                elif data_type == "Blobs" or data_type == "Varied":
                    labels = algo_func(X_data, damping=0.9, preference=-50, random_state=RANDOM_STATE)
                else: # Anisotropy, No Structure
                    labels = algo_func(X_data, damping=0.9, preference=-100, random_state=RANDOM_STATE)

            current_ax.scatter(X_data[:, 0], X_data[:, 1], c=labels, cmap='viridis', s=10)
            current_ax.set_xticks([])
            current_ax.set_yticks([])
        except Exception as e:
            current_ax.text(0.5, 0.5, f"Ошибка: {e}", horizontalalignment='center', verticalalignment='center', transform=current_ax.transAxes, color='red')
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            print(f"Ошибка при обработке {data_type} с {algo_name}: {e}")

plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.show()

