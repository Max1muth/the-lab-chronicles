import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Generate data
np.random.seed(42)  # For reproducibility
x_min, x_max, points = -10, 10, 50
x = np.linspace(x_min, x_max, points)
true_k, true_b = 2, 3  # True parameters for y = kx + b
y = true_k * x + true_b + np.random.uniform(-3, 3, points)  # Add noise

# Mean Squared Error
def mse(x, y, k, b):
    y_pred = k * x + b
    return np.mean((y_pred - y) ** 2)

# Partial derivatives
def get_dk(x, y, k, b):
    return (2 / len(x)) * np.sum(x * (k * x + b - y))

def get_db(x, y, k, b):
    return (2 / len(x)) * np.sum(k * x + b - y)

# Gradient descent
def fit(x, y, speed, epochs, k0, b0):
    k, b = k0, b0
    k_list = [k0]
    b_list = [b0]
    mse_list = [mse(x, y, k0, b0)]
    
    for i in range(epochs):
        k = k - speed * get_dk(x, y, k, b)
        b = b - speed * get_db(x, y, k, b)
        k_list.append(k)
        b_list.append(b)
        mse_list.append(mse(x, y, k, b))
    
    return k_list, b_list, mse_list

# Parameters for gradient descent
speed = 0.001
epochs = 100
k0, b0 = 0, 0  # Initial guesses

# Run gradient descent
k_list, b_list, mse_list = fit(x, y, speed, epochs, k0, b0)

# Visualization with slider
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)

# Plot data points and initial regression line
ax1.scatter(x, y, color='blue', label='Data points')
line, = ax1.plot(x, k_list[0] * x + b_list[0], color='red', label='Regression line')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression')
ax1.legend()
ax1.grid(True)

# Plot MSE
mse_line, = ax2.plot(range(len(mse_list)), mse_list, color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')
ax2.set_title('Mean Squared Error')
ax2.grid(True)

# Slider
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Epoch', 0, epochs, valinit=0, valstep=1)

# Update function for slider
def update(val):
    epoch = int(slider.val)
    line.set_ydata(k_list[epoch] * x + b_list[epoch])
    ax1.set_title(f'Linear Regression (Epoch {epoch}, k={k_list[epoch]:.2f}, b={b_list[epoch]:.2f})')
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()

# Print final parameters
print(f"Final k: {k_list[-1]:.2f}, Final b: {b_list[-1]:.2f}, Final MSE: {mse_list[-1]:.2f}")
