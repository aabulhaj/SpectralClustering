import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def euclid(X, Y):
    """
    Returns the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    return euclidean_distances(X, Y)


def euclidean_centroid(X):
    """
    Returns the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    return np.mean(X, axis=0)


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++. Returns k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    samples_count = X.shape[0]

    centroids = np.array([X[np.random.randint(samples_count)]])

    for _ in range(k - 1):
        w_i = np.min(metric(centroids, X) ** 2, axis=0)
        w_i = w_i / np.sum(w_i)
        centroids = np.concatenate((centroids, [X[np.random.choice(samples_count, p=w_i)]]))
    return centroids


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: a tuple of (clustering, centroids)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    C = list()
    centroids = init(X, k, metric)
    for _ in range(iterations):
        C = np.argmin(metric(X, centroids), axis=1)
        new_centroids = np.array([center(X[C == i]) for i in range(k)])

        if (new_centroids == centroids).all():
            break

        centroids = new_centroids
    return C, centroids
