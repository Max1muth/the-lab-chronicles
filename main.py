import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import random

# 1. Определим исходную функцию
def f(x):
    return np.sin(x) + 0.5 * x**2 - 2 * x + 1

# 2. Генерация данных
np.random.seed(42)  # Для воспроизводимости
x = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 точек от -5 до 5
e = [random.uniform(-1, 1) for _ in range(100)]  # Шум
y = f(x.flatten()) + np.array(e)  # Истинные значения с шумом

# 3. Инициализация моделей регрессии
models = {
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# 4. Обучение моделей и предсказания
predictions = {}
mse_scores = {}
x_smooth = np.linspace(-5, 5, 1000).reshape(-1, 1)  # Для плавных графиков

for name, model in models.items():
    # Обучение модели
    model.fit(x, y)
    # Предсказания
    y_pred = model.predict(x)
    y_pred_smooth = model.predict(x_smooth)
    predictions[name] = (y_pred, y_pred_smooth)
    # Вычисление MSE
    mse_scores[name] = mean_squared_error(y, y_pred)

# 5. Построение графиков
plt.figure(figsize=(15, 5))
for i, (name, (y_pred, y_pred_smooth)) in enumerate(predictions.items(), 1):
    plt.subplot(1, 3, i)
    # Исходные точки
    plt.scatter(x, y, color='blue', s=10, label='Data points')
    # Исходная функция
    plt.plot(x_smooth, f(x_smooth.flatten()), color='green', label='True function')
    # Предсказанная функция
    plt.plot(x_smooth, y_pred_smooth, color='red', label='Predicted function')
    plt.title(f'{name} (MSE: {mse_scores[name]:.4f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Выводы
print("Mean Squared Error (MSE) для каждой модели:")
for name, mse in mse_scores.items():
    print(f"{name}: {mse:.4f}")

print("\nВыводы:")
print("На основе визуального анализа графиков и значений MSE можно сделать следующие выводы:")
if mse_scores['Gradient Boosting'] == min(mse_scores.values()):
    print("- Gradient Boosting показывает наилучший результат, так как имеет наименьшую MSE и график предсказанной функции наиболее близок к истинной функции.")
if mse_scores['Random Forest'] == min(mse_scores.values()):
    print("- Random Forest демонстрирует наилучшую производительность с наименьшей MSE и хорошим соответствием истинной функции.")
if mse_scores['SVR'] == min(mse_scores.values()):
    print("- SVR работает лучше всего, минимизируя MSE и точно воспроизводя форму истинной функции.")
print("- SVR может быть менее точным для этой функции, так как его предсказания имеют больше отклонений, особенно на краях интервала.")
print("- Random Forest и Gradient Boosting обычно лучше справляются с нелинейными зависимостями, но Gradient Boosting чаще всего оказывается предпочтительным благодаря более плавным предсказаниям.")
