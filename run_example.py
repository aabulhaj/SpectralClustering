import matplotlib.pyplot as plt

import kmeans
import spectral_clustering
import synthetic_data


def visualize_kmeans(dataset, clusters_count, iters=10000):
    clusters, centroids = kmeans.kmeans(dataset, clusters_count, iterations=iters)

    plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1], c=clusters)
    plt.title('kmeans++ with {} clusters.'.format(clusters_count))
    plt.show()


def visualize_spectral(dataset, clusters_count):
    clusters = spectral_clustering.spectral(dataset, clusters_count, 15, spectral_clustering.mnn)

    plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1], c=clusters)
    plt.title('Spectral clustering with {} clusters (KNN, neighbours={}).'.format(clusters_count, 15))
    plt.show()


if __name__ == '__main__':
    circles = synthetic_data.create_circles()
    blobs = synthetic_data.create_blobs()

    # Analyze blobs with kmean.
    visualize_kmeans(blobs, 5)

    # # Analyze circles with kmean.
    visualize_kmeans(circles, 4)

    # Analyze circles with spectral clustering.
    visualize_spectral(circles, 4)
