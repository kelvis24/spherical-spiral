import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs


# Generate synthetic data with make_blobs
# This function will create a dataset that consists of several "blobs" or clusters of points.
# n_samples: The total number of points equally divided among clusters.
# centers: The number of centers to generate, or the fixed center locations.
# n_features: The number of features for each sample.
# random_state: Determines random number generation for dataset creation. Ensures reproducibility.
# n_samples = 150
# random_state = 123
# X, y = make_blobs(n_samples=n_samples, centers=3, n_features=5, random_state=random_state)
def generate_synthetic(n_samples=4000, n_centers=5, n_features=128, random_state=123):
    # Parameters for the synthetic dataset
    # n_samples = 4000
    # centers = 5
    # n_features = 128
    # random_state = 123

    # Generate synthetic data
    X, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features, random_state=random_state)

    # Create a dictionary where the keys are the sample indices and the values are dictionaries
    # containing the features and the cluster label for each sample
    data_dict = {i: {'features': X[i], 'cluster': y[i]} for i in range(n_samples)}

    print(X, "\n")
    print(y)
    # PCA transformation
    # Initialize PCA, specifying the number of components (n_components) to reduce the dataset to.
    # Here, we reduce the dataset from 5 dimensions (features) to 2 dimensions for visualization.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plotting
    # After applying PCA, we plot the transformed dataset. The points are colored according to their original cluster.
    # This demonstrates how PCA can be used for dimensionality reduction, helping in visualizing high-dimensional data.
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', s=40)
    # plt.title('PCA of Synthetic Dataset')
    # plt.xlabel('Principal Component 1')  # The first principal component
    # plt.ylabel('Principal Component 2')  # The second principal component
    # plt.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1', 'Class 2'])
    # plt.grid(True)
    # plt.show()

    return data_dict, X, y, X_pca, pca


if __name__ == '__main__':
    print('Error: not main file')
