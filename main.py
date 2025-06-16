import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch, MeanShift
from sklearn.mixture import GaussianMixture
from collections import defaultdict

def generate_data(num_points_per_cluster=20, centers=None, radii=None):
    """Generate sample data for clustering."""
    if centers is None:
        centers = [(2, 2), (8, 3), (4, 9)]
    if radii is None:
        radii = [1.5, 1.0, 1.8]

    all_x = []
    all_y = []

    for i, (cx, cy) in enumerate(centers):
        r = radii[i]
        for _ in range(num_points_per_cluster):
            angle = 2 * np.pi * np.random.rand()
            current_r = r * np.sqrt(np.random.rand())
            x = cx + current_r * np.cos(angle)
            y = cy + current_r * np.sin(angle)
            all_x.append(x)
            all_y.append(y)
    return all_x, all_y

def apply_clustering(algorithm, X, k=3):
    """Apply selected clustering algorithm to data."""
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=k, init='random', n_init=1, max_iter=1)
        model.fit(X)
        return model.labels_, model.cluster_centers_, True
    
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=1.0, min_samples=5)
        labels = model.fit_predict(X)
        return labels, None, False
    
    elif algorithm == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
        centroids = [X[labels == i].mean(axis=0) for i in range(k) if len(X[labels == i]) > 0]
        return labels, centroids, False
    
    elif algorithm == 'Spectral':
        model = SpectralClustering(n_clusters=k, affinity='rbf', gamma=1.0)
        labels = model.fit_predict(X)
        centroids = [X[labels == i].mean(axis=0) for i in range(k) if len(X[labels == i]) > 0]
        return labels, centroids, False
    
    elif algorithm == 'Birch':
        model = Birch(n_clusters=k)
        labels = model.fit_predict(X)
        centroids = [X[labels == i].mean(axis=0) for i in range(k) if len(X[labels == i]) > 0]
        return labels, centroids, False
    
    elif algorithm == 'MeanShift':
        model = MeanShift(bandwidth=1.5, bin_seeding=True)
        labels = model.fit_predict(X)
        return labels, model.cluster_centers_, False
    
    elif algorithm == 'GaussianMixture':
        model = GaussianMixture(n_components=k)
        labels = model.fit(X).predict(X)
        return labels, model.means_, False

# Generate data
x_data, y_data = generate_data(num_points_per_cluster=20,
                             centers=[(2, 2), (8, 3), (4, 9)],
                             radii=[1.5, 1.0, 1.8])
X = np.array(list(zip(x_data, y_data)))

# Setup plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25, left=0.25)

# Algorithm selection
algorithms = ['KMeans', 'DBSCAN', 'Agglomerative', 'Spectral', 'Birch', 'MeanShift', 'GaussianMixture']
current_algorithm = 'KMeans'

# Radio buttons for algorithm selection
rax = plt.axes([0.025, 0.4, 0.15, 0.3], facecolor='lightgoldenrodyellow')
radio = RadioButtons(rax, algorithms, active=0)

# Slider for iterations (only active for KMeans)
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Iteration', 0, 9, valinit=0, valstep=1)

# Store KMeans history
models_history = defaultdict(list)
current_labels = None
current_centroids = None

def init_kmeans_history():
    """Initialize KMeans history for animation."""
    global models_history
    models_history['KMeans'] = []
    
    model = KMeans(n_clusters=3, init='random', n_init=1, max_iter=1)
    for i in range(10):  # 10 iterations
        if i > 0:
            model.init = model.cluster_centers_
        model.fit(X)
        models_history['KMeans'].append({
            'labels': model.labels_,
            'centroids': model.cluster_centers_
        })

init_kmeans_history()

def update_display(algorithm, iteration=0):
    """Update the display with current clustering."""
    global current_labels, current_centroids
    
    if algorithm == 'KMeans':
        data = models_history['KMeans'][iteration]
        current_labels = data['labels']
        current_centroids = data['centroids']
        slider.set_active(True)
    else:
        current_labels, current_centroids, has_history = apply_clustering(algorithm, X)
        slider.set_active(False)
    
    ax.clear()
    
    # Plot data points
    scatter = ax.scatter(x_data, y_data, c=current_labels, cmap='viridis', s=50, alpha=0.8)
    
    # Plot centroids if they exist
    if current_centroids is not None and len(current_centroids) > 0:
        centroids_plot = ax.scatter([c[0] for c in current_centroids], 
                                  [c[1] for c in current_centroids],
                                  marker='X', s=200, color='red', 
                                  edgecolor='black', label='Cluster Centers')
        ax.legend()
    
    ax.set_title(f'{algorithm} Clustering (Iteration {iteration if algorithm == "KMeans" else "Final"})')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    fig.canvas.draw_idle()

def on_algorithm_change(label):
    """Handle algorithm change event."""
    global current_algorithm
    current_algorithm = label
    update_display(label)

def on_slider_change(val):
    """Handle slider change event."""
    if current_algorithm == 'KMeans':
        update_display(current_algorithm, int(val))

radio.on_clicked(on_algorithm_change)
slider.on_changed(on_slider_change)

# Initial display
update_display(current_algorithm)

plt.show()