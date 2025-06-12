import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Set random seed for reproducibility
np.random.seed(30)

# Generate the five datasets
n_samples = 500
seed = 30

# 1. Circles
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
# 2. Moons
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
# 3. Varied clusters
cluster_std = [1.0, 0.5]
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=cluster_std, random_state=seed, centers=2)
# 4. Anisotropic
random_state = 170
x, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state, centers=2)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
x_aniso = np.dot(x, transformation)
aniso = (x_aniso, y)
# 5. Blobs
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed, centers=2)

datasets_list = [
    ("Circles", noisy_circles),
    ("Moons", noisy_moons),
    ("Varied", varied),
    ("Anisotropic", aniso),
    ("Blobs", blobs)
]

# Classification models
def create_knn_model():
    return KNeighborsClassifier(n_neighbors=3)

def create_lr_model():
    return LogisticRegression(max_iter=200)

def create_mlp_model(input_shape=(2,)):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

models = [
    ("KNN", create_knn_model),
    ("Logistic Regression", create_lr_model),
    ("MLP (Keras)", create_mlp_model)
]

# Create figure for visualization
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 25))
markers = ['x', 'o']

# Function to plot decision boundary
def plot_decision_boundary(ax, model, x, y, x_train, x_test, y_train, y_test, title):
    # Create mesh grid
    temp_x = np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 100)
    temp_y = np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 100)
    xx, yy = np.meshgrid(temp_x, temp_y)
    
    # Predict for grid points
    if 'Keras' in title:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
        Z = (Z > 0.5).astype(int).reshape(xx.shape)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    # Plot training points
    for i in range(len(x_train)):
        ax.scatter(x_train[i, 0], x_train[i, 1], marker=markers[y_train[i]], c='b', alpha=0.7)
    
    # Plot test points (green for correct, red for incorrect)
    y_pred = model.predict(x_test) if 'Keras' not in title else (model.predict(x_test, verbose=0) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    for i in range(len(x_test)):
        color = 'g' if y_test[i] == y_pred[i] else 'r'
        ax.scatter(x_test[i, 0], x_test[i, 1], marker=markers[y_test[i]], c=color)
    
    ax.set_title(f"{title}\nAccuracy: {accuracy:.2f}")

# Process each dataset and model combination
for i, (dataset_name, (x, y)) in enumerate(datasets_list):
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    for j, (model_name, create_model) in enumerate(models):
        # Create and train model
        model = create_model()
        if 'Keras' in model_name:
            model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                     epochs=15, batch_size=32, verbose=0)
        else:
            model.fit(x_train, y_train)
        
        # Plot results
        plot_decision_boundary(axes[i, j], model, x, y, x_train, x_test, y_train, y_test, 
                            f"{model_name} on {dataset_name}")

# Adjust layout and display
plt.tight_layout()
plt.show()
