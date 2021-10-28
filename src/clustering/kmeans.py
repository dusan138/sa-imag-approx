import numpy as np

def find_closest(point, centroids):
    pass

def perform_kmeans(data_set, k=5):
    # Cluster initialization
    max_vals = np.max(data_set, axis=0)
    min_vals = np.min(data_set, axis=0)

    centroids = min_vals + (max_vals - min_vals)*np.random.rand(k, 3)
    