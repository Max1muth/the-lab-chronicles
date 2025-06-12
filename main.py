import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

def generate_noisy_data(start_x, end_x, num_points, noise_scale=0.1, filename="regression_data.csv"):
    """
    Генерирует исходные точки для регрессии с добавлением экспоненциального шума
    и сохраняет их в файл.

    Args:
        start_x (float): Начальное значение диапазона для x.
        end_x (float): Конечное значение диапазона для x.
        num_points (int): Количество точек для генерации.
        noise_scale (float): Масштаб для генерации экспоненциального шума.
                              Чем больше значение, тем сильнее шум.
        filename (str): Имя файла для сохранения данных.
    """
    # 1. Генерация x значений
    x = np.linspace(start_x, end_x, num_points)

    # 2. Вычисление y значений по исходной функции
    y_true = np.exp(0.3 * x) * np.cos(2 * x)

    # 3. Добавление экспоненциального шума
    # Генерируем случайные числа из экспоненциального распределения.
    # Вычитаем среднее (1/lambda) чтобы центрировать шум вокруг 0,
    # затем масштабируем и добавляем/вычитаем к y_true.
    # Для экспоненциального роста шума, лучше использовать мультипликативный шум.
    # noise = np.random.exponential(scale=noise_scale, size=num_points) - noise_scale
    # y_noisy = y_true + y_true * noise

    # Другой способ добавления мультипликативного экспоненциального шума:
    # np.random.exponential(scale=1.0) создает шум >= 0.
    # Чтобы шум был как положительным, так и отрицательным,
    # мы можем генерировать его вокруг 1.0 (для умножения)
    # Например, noise_factor = 1 + (np.random.exponential(scale=noise_scale, size=num_points) - noise_scale_offset)
    # где noise_scale_offset нужен для смещения среднего экспоненциального распределения
    # чтобы шум был симметричен относительно y_true.
    # Проще всего генерировать случайные числа и применять их к y_true.

    # Генерируем экспоненциальный шум и преобразуем его так,
    # чтобы он имел как положительные, так и отрицательные значения, центрированные около 0.
    # Используем `np.random.normal` для простоты и контроля за распределением шума,
    # но увеличиваем его дисперсию пропорционально abs(y_true)
    # Если вы хотите строго "экспоненциальный" шум в смысле распределения,
    # то это будет сложнее визуально контролировать его влияние.
    # Для демонстрации зависимости шума от y_true, можно использовать:
    # noise = np.random.normal(loc=0, scale=noise_scale * np.abs(y_true))
    # y_noisy = y_true + noise

    # Если мы хотим, чтобы сам шум был с экспоненциальным распределением:
    # np.random.exponential(scale=1) дает значения >=0.
    # Чтобы получить шум около 0, можно сгенерировать 2 набора, вычесть одно из другого.
    # Или просто масштабировать экспоненциальное распределение и вычесть его среднее.
    # Более наглядно будет, если шум увеличивается с ростом |y|.
    # Давайте добавим шум, который увеличивается пропорционально |y_true|
    # и имеет нормальное распределение, но с масштабом, зависящим от y_true.
    # Это распространенный способ добавления "экспоненциального" (в смысле роста) шума.

    noise_magnitude = noise_scale * np.abs(y_true)
    noise = np.random.normal(loc=0, scale=noise_magnitude)
    y_noisy = y_true + noise


    # 4. Сохранение данных в файл
    data = np.column_stack((x, y_noisy))
    np.savetxt(filename, data, delimiter=',', header='x,y', comments='')
    print(f"Данные успешно сохранены в файл: {filename}")

    # Визуализация (для проверки)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label='Исходная функция $y = e^{0.3x} \cdot \cos(2x)$', color='blue')
    plt.scatter(x, y_noisy, label='Данные с экспоненциальным шумом', color='red', s=10, alpha=0.6)
    plt.title('Сгенерированные данные для регрессии с экспоненциальным шумом')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Пример использования ---
generate_noisy_data(start_x=0, end_x=10, num_points=100, noise_scale=0.15, filename="regression_data_with_exponential_noise.csv")


# Suppress warnings that might appear from optimization process
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Load the data generated previously
try:
    data = pd.read_csv("regression_data_with_exponential_noise.csv")
    X = data[['x']].values
    y = data['y'].values
except FileNotFoundError:
    print("Error: 'regression_data_with_exponential_noise.csv' not found.")
    print("Please run the data generation code first.")
    exit()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It's good practice to scale features for many linear models, including PassiveAggressiveRegressor
# as it's sensitive to feature scales.
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# For y, we might scale it if the range is very large, but for regression,
# it's often more interpretable to predict on the original scale.
# However, for some online learning algorithms, scaling y can also help convergence.
# Let's scale y for this example, remember to inverse_transform predictions.
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()


# --- Custom Epoch-based Training Loop for PassiveAggressiveRegressor ---
# While PassiveAggressiveRegressor has max_iter, to truly "track by epochs"
# and perform custom actions per epoch, we need to iterate manually.

def train_passive_aggressive_regressor_by_epochs(X_train, y_train, X_val=None, y_val=None,
                                                 n_epochs=100, shuffle_data=True,
                                                 model_params=None):
    """
    Trains a PassiveAggressiveRegressor model epoch by epoch and tracks performance.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        X_val (np.array, optional): Validation features.
        y_val (np.array, optional): Validation targets.
        n_epochs (int): Number of epochs to train for.
        shuffle_data (bool): Whether to shuffle data before each epoch.
        model_params (dict, optional): Dictionary of parameters for PassiveAggressiveRegressor.

    Returns:
        tuple: A tuple containing:
            - model (PassiveAggressiveRegressor): The trained model.
            - history (dict): Dictionary with 'train_mse', 'val_mse' (if X_val/y_val provided)
                              and 'train_r2', 'val_r2' for each epoch.
    """
    if model_params is None:
        model_params = {}

    # Initialize the model with warm_start=True to allow incremental fitting
    model = PassiveAggressiveRegressor(warm_start=True, random_state=42, **model_params)

    # We need to set max_iter to 1 to force fit() to run only one pass per call
    # And then control the epochs manually.
    model.set_params(max_iter=1, tol=None) # We control stopping manually

    history = {
        'train_mse': [], 'val_mse': [],
        'train_r2': [], 'val_r2': []
    }

    n_samples = X_train.shape[0]

    print(f"Starting epoch-based training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        if shuffle_data:
            # Create a random permutation of indices
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
        else:
            X_shuffled = X_train
            y_shuffled = y_train

        # Fit for one epoch (one pass over the data)
        model.fit(X_shuffled, y_shuffled)

        # Evaluate performance
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        history['train_mse'].append(train_mse)
        history['train_r2'].append(train_r2)

        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            history['val_mse'].append(val_mse)
            history['val_r2'].append(val_r2)
            print(f"Epoch {epoch+1}/{n_epochs}: Train MSE={train_mse:.4f}, Val MSE={val_mse:.4f}")
        else:
            print(f"Epoch {epoch+1}/{n_epochs}: Train MSE={train_mse:.4f}")

    return model, history

# --- Example Usage of Custom Epoch Training ---
# It's better to use a small validation set for tracking during manual epochs
# as it helps prevent overfitting.
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train_scaled, y_train_scaled, test_size=0.15, random_state=42
)

initial_model_params = {'C': 1.0, 'loss': 'epsilon_insensitive', 'max_iter': 1} # max_iter will be overridden

# Let's train for a fixed number of epochs
trained_model, training_history = train_passive_aggressive_regressor_by_epochs(
    X_train=X_train_sub,
    y_train=y_train_sub,
    X_val=X_val_sub,
    y_val=y_val_sub,
    n_epochs=50, # Let's run for 50 epochs
    shuffle_data=True,
    model_params=initial_model_params
)

# Plotting the training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_history['train_mse'], label='Train MSE')
if 'val_mse' in training_history and training_history['val_mse']:
    plt.plot(training_history['val_mse'], label='Validation MSE')
plt.title('Mean Squared Error (MSE) per Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(training_history['train_r2'], label='Train R2')
if 'val_r2' in training_history and training_history['val_r2']:
    plt.plot(training_history['val_r2'], label='Validation R2')
plt.title('R-squared (R2) per Epoch')
plt.xlabel('Epoch')
plt.ylabel('R2 Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final evaluation on the test set
y_pred_scaled = trained_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

final_mse = mean_squared_error(y_test_original, y_pred)
final_r2 = r2_score(y_test_original, y_pred)

print(f"\nFinal Test MSE: {final_mse:.4f}")
print(f"Final Test R2: {final_r2:.4f}")

# Plotting predictions vs actual for the test set
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test_original, label='Actual Test Data', color='blue', s=10, alpha=0.6)
plt.scatter(X_test, y_pred, label='Predicted Test Data', color='red', s=10, alpha=0.6)
plt.title('Actual vs Predicted values on Test Set')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()


from matplotlib.widgets import Slider
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import StandardScaler


# Suppress warnings that might appear from optimization process
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Загрузка данных
try:
    data = pd.read_csv("regression_data_with_exponential_noise.csv")
    X = data[['x']].values
    y = data['y'].values
except FileNotFoundError:
    print("Ошибка: файл 'regression_data_with_exponential_noise.csv' не найден.")
    print("Пожалуйста, сначала запустите код генерации данных.")
    exit()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# --- Модифицированная функция обучения с отслеживанием MSLE ---

def train_passive_aggressive_regressor_by_epochs_with_msle(X_train, y_train, X_val=None, y_val=None,
                                                          n_epochs=100, shuffle_data=True,
                                                          model_params=None, offset=100):
    """
    Обучает модель PassiveAggressiveRegressor по эпохам, отслеживая MSLE.

    Args:
        X_train (np.array): Обучающие признаки.
        y_train (np.array): Целевые значения для обучения.
        X_val (np.array, optional): Признаки для валидации.
        y_val (np.array, optional): Целевые значения для валидации.
        n_epochs (int): Количество эпох для обучения.
        shuffle_data (bool): Перемешивать ли данные перед каждой эпохой.
        model_params (dict, optional): Словарь параметров для PassiveAggressiveRegressor.
        offset (float): Смещение, добавляемое к y для вычисления MSLE,
                        чтобы все значения были положительными.

    Returns:
        tuple: Кортеж, содержащий:
            - model (PassiveAggressiveRegressor): Обученная модель.
            - history (dict): Словарь с 'train_msle', 'val_msle' (если X_val/y_val предоставлены)
                              и 'train_r2', 'val_r2' для каждой эпохи.
            - model_states (list): Список моделей (или их предсказаний на полной X_data)
                                   для каждой эпохи для визуализации.
    """
    if model_params is None:
        model_params = {}

    model = PassiveAggressiveRegressor(warm_start=True, random_state=42, **model_params)
    model.set_params(max_iter=1, tol=None) # Ручное управление эпохами

    history = {
        'train_msle': [], 'val_msle': [],
        'train_r2': [], 'val_r2': []
    }
    model_states = [] # Будем хранить предсказания для визуализации

    n_samples = X_train.shape[0]

    print(f"Начало обучения по эпохам ({n_epochs} эпох) с отслеживанием MSLE...")
    for epoch in range(n_epochs):
        if shuffle_data:
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
        else:
            X_shuffled = X_train
            y_shuffled = y_train

        model.fit(X_shuffled, y_shuffled)

        # Сохраняем копию модели или ее текущие веса/предсказания
        # Для визуализации всей кривой, лучше сохранить предсказания на полном диапазоне
        # или саму модель, если она не слишком большая.
        # Для простоты, сохраним предсказания на всем X для визуализации
        x_full_range_for_viz = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        x_full_range_scaled_for_viz = scaler_X.transform(x_full_range_for_viz)
        y_pred_full_range_scaled_for_viz = model.predict(x_full_range_scaled_for_viz)
        model_states.append(scaler_y.inverse_transform(y_pred_full_range_scaled_for_viz.reshape(-1, 1)).flatten())


        # Оценка производительности
        y_train_pred = model.predict(X_train)
        y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()

        # Вычисление MSLE: важно, чтобы значения были положительными
        # Добавляем смещение, чтобы все y_true и y_pred стали положительными
        # Внимание: это может повлиять на интерпретацию MSLE, но необходимо для его вычисления
        msle_train = mean_squared_log_error(y_train_original + offset, y_train_pred_original + offset)
        train_r2 = r2_score(y_train_original, y_train_pred_original)

        history['train_msle'].append(msle_train)
        history['train_r2'].append(train_r2)

        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_pred_original = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
            y_val_original = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

            msle_val = mean_squared_log_error(y_val_original + offset, y_val_pred_original + offset)
            val_r2 = r2_score(y_val_original, y_val_pred_original)

            history['val_msle'].append(msle_val)
            history['val_r2'].append(val_r2)
            print(f"Эпоха {epoch+1}/{n_epochs}: Train MSLE={msle_train:.4f}, Val MSLE={msle_val:.4f}")
        else:
            print(f"Эпоха {epoch+1}/{n_epochs}: Train MSLE={msle_train:.4f}")

    return model, history, model_states

# --- Пример использования и визуализация ---

# Разделение обучающей выборки на обучающую и валидационную
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train_scaled, y_train_scaled, test_size=0.15, random_state=42
)

initial_model_params = {'C': 1.0, 'loss': 'epsilon_insensitive'}

# Обучаем модель и получаем историю MSLE и состояния модели
trained_model, training_history, model_states_for_viz = train_passive_aggressive_regressor_by_epochs_with_msle(
    X_train=X_train_sub,
    y_train=y_train_sub,
    X_val=X_val_sub,
    y_val=y_val_sub,
    n_epochs=100, # Количество эпох для обучения
    shuffle_data=True,
    model_params=initial_model_params,
    offset=100 # Добавляем смещение для корректного вычисления MSLE
)

# --- График зависимости MSLE от эпохи обучения ---
plt.figure(figsize=(10, 6))
plt.plot(training_history['train_msle'], label='Train MSLE', color='blue')
if 'val_msle' in training_history and training_history['val_msle']:
    plt.plot(training_history['val_msle'], label='Validation MSLE', color='orange')
plt.title('Зависимость средней квадратичной логарифмической ошибки (MSLE) от эпохи обучения')
plt.xlabel('Эпоха')
plt.ylabel('MSLE')
plt.legend()
plt.grid(True)
plt.show()

### График с ползунком для визуализации изменения кривой
# Для визуализации кривой регрессии по эпохам
x_full_range_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

fig_viz, ax_viz = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.25)

# Истинная функция
y_true_full_range = np.exp(0.3 * x_full_range_plot) * np.cos(2 * x_full_range_plot)
ax_viz.plot(x_full_range_plot, y_true_full_range, label='Истинная функция', color='purple', linestyle='--', alpha=0.7)

# Исходные зашумленные точки
ax_viz.scatter(X, y, label='Зашумленные данные', color='gray', s=15, alpha=0.6)

# Начальная кривая регрессии (эпоха 0)
line_plot, = ax_viz.plot(x_full_range_plot, model_states_for_viz[0],
                         label='Предсказание модели (текущая эпоха)', color='red', linewidth=2)

ax_viz.set_title(f'Изменение кривой регрессии PassiveAggressiveRegressor (Эпоха 0)')
ax_viz.set_xlabel('X Координата')
ax_viz.set_ylabel('Y Координата')
ax_viz.legend()
ax_viz.grid(True)

# Создание ползунка
ax_slider_viz = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
slider_viz = Slider(ax_slider_viz, 'Эпоха', 0, len(model_states_for_viz) - 1, valinit=0, valstep=1)

# Функция обновления графика при изменении ползунка
def update_viz(val):
    epoch_idx = int(slider_viz.val)
    current_prediction_curve = model_states_for_viz[epoch_idx]
    line_plot.set_ydata(current_prediction_curve) # Обновляем данные y для кривой
    ax_viz.set_title(f'Изменение кривой регрессии PassiveAggressiveRegressor (Эпоха {epoch_idx})')
    fig_viz.canvas.draw_idle()

slider_viz.on_changed(update_viz)

plt.show()
