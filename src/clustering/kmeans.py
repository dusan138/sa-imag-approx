import numpy as np


class KMeansClustering:

    def __init__(
            self, 
            k: int, 
            max_iter: int, 
            tol: float, 
            max_repeats: int = 20, 
            init_method: str = "random", 
            random_seed: int = None
    ):
        
        self.k = k
        self.max_iter = max_iter
        self.max_repeats = max_repeats
        self.tol = tol
        self.centroids = np.zeros(k, 3)
        self.random_seed = random_seed
        self.init_method = init_method

    def fit(self, data_set, predict=False):
        # If predict is True, then return the list of clusters        
        
        if self.init_method == "random":
            centroids = self.__random_init(data_set=data_set)
        else:
            raise NotImplementedError()

        n = data_set.shape[0] # number of data points

        cluster_labels = [-1]*n

        n_iter = 0
        repeats = 0

        while n_iter <= self.max_iter:

            new_labels = [-1]*n

            for j, point in enumerate(data_set):
                new_labels[j] = self.__get_cluster(point, centroids)

            # Break if cluster assignments don't change
            if new_labels == cluster_labels:
                break

            cluster_labels = new_labels.copy()

            # Update centroids
            new_centroids = self.__calculate_centroids(
                data_set=data_set, cluster_labels=cluster_labels
            )

            # Calculate the distance between old and new centroids
            diff = ((centroids - new_centroids).sum(axis=1)**2).sum()
            
            # If diff is less than tolerance for max_repeats iterations, break
            if diff < self.tol:
                repeats += 1
                if repeats == self.max_repeats:
                    break

            centroids = new_centroids.copy()
            

    def __random_init(self, data_set):
        # Random cluster initialization
        min_vals = np.min(data_set, axis=0)
        max_vals = np.min(data_set, axis=0)

        rng = np.random.default_rng(seed=self.random_seed)

        centroids = \
            min_vals + (max_vals - min_vals)*rng.random((self.k, 3))

        return centroids

    def __calculate_centroids(self, data_set, cluster_labels):
        
        sums = np.zeros((self.k, 3))
        lens = [0]*self.k

        for j, point in enumerate(data_set):
            label = cluster_labels[j]
            sums[label, :] += point
            lens[label] += 1
        
        centroids = [sums[label, :] / lens[label] for label in range(self.k)]

        return centroids

    def __get_cluster(self, point, centroids):
        # Calculate the distance of point from centroids and return the index
        # of closest one
        distances = ((centroids - point)**2).sum(axis=1)

        return distances.argmin()


if __name__ == "__main__":
    pass