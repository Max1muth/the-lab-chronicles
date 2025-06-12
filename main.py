# f(x) = x**3 − 2x + sin(x) (with gpt:)

import matplotlib.pyplot as plt
import numpy as np

# Исходная функция и её производная
func = lambda x: x**3 - 2*x + np.sin(x)
diffFunc = lambda x: 3*x**2 - 2 + np.cos(x)

# Реализация градиентного спуска
def gradientDescend(func=lambda x: x**2,
                    diffFunc=lambda x: 2*x,
                    x0=3,
                    speed=0.01,
                    epochs=100):
    xList = [x0]
    yList = [func(x0)]

    x = x0
    for i in range(epochs):
        x = x - speed * diffFunc(x)
        xList.append(x)
        yList.append(func(x))

    return xList, yList

# Вызов градиентного спуска для заданной функции
x_vals, y_vals = gradientDescend(func, diffFunc, x0=3, speed=0.01, epochs=100)

# Вывод финального значения минимума
print(f"Минимум достигается при x ≈ {x_vals[-1]:.5f}, f(x) ≈ {y_vals[-1]:.5f}")

# Строим функцию и точки
x_range = np.linspace(-2.5, 2.5, 400)
y_range = func(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='f(x)', color='blue')
plt.scatter(x_vals, y_vals, color='red', label='Gradient Descent Points')
plt.title('Градиентный спуск для функции f(x) = x^3 - 2x + sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод: сошелся ли метод?
print(f"Последнее значение x: {x_vals[-1]:.5f}")
print(f"Значение функции в этой точке: {y_vals[-1]:.5f}")

def test_convergence(speed, x_target=1.2, tol=0.1):
    x_vals, _ = gradientDescend(func, diffFunc, x0=3, speed=speed, epochs=100)
    return abs(x_vals[-1] - x_target) < tol

# Бинарный поиск граничного значения speed
def find_critical_speed(low=0.001, high=1.0, tolerance=1e-4):
    while high - low > tolerance:
        mid = (low + high) / 2
        if test_convergence(mid):
            low = mid  # скорость допустимая, можно выше
        else:
            high = mid  # скорость слишком большая, метод расходится
    return low

critical_speed = find_critical_speed()
print(f"Примерное граничное значение speed: {critical_speed:.5f}")
print("Ниже этого значения метод сходится, выше — расходится.")
