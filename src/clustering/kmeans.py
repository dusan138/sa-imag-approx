import numpy as np


class KMeansClustering:

    def __init__(
            self, k, max_iter, tol, init_method="random", random_seed=None):
        
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = np.zeros(k, 3)
        self.random_seed = random_seed
        self.init_method = init_method

    def fit(self, data_set):
        
        if self.init_method == "random":
            self.__random_init(data_set=data_set)
        else:
            raise NotImplementedError()

    def __random_init(self, data_set):
        # Random cluster initialization
        min_vals = np.min(data_set, axis=0)
        max_vals = np.min(data_set, axis=0)

        rng = np.random.default_rng(seed=self.random_seed)

        self.centroids = \
            min_vals + (max_vals - min_vals)*rng.random((self.k, 3))

    def __calculate_distances(self, centroid, points):
        pass


if __name__ == "__main__":
    pass