import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from spherical_spiral import SphericalSpiral


def plot_centroids(vectors, num_select=64):
    spirals = SphericalSpiral(num_select=num_select)
    centroids = []

    # Adjust points for each vector and find its centroid
    for vector in vectors:
        spirals.adjust_points(vector)
        centroid = spirals.find_centroid()
        centroids.append(centroid)

    # Setting up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the centroids
    centroids = np.array(centroids)  # Convert to numpy array for easier handling
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2])

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()


def plot_centroids_unique_spheres(vectors, num_select=64):
    # spirals = SphericalSpiral(num_select=64)
    centroids = []

    # Adjust points for each vector and find its centroid
    for vector in vectors:
        spirals = SphericalSpiral(num_select=num_select)
        spirals.adjust_points(vector)
        centroid = spirals.find_centroid()
        centroids.append(centroid)

    # Setting up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the centroids
    centroids = np.array(centroids)  # Convert to numpy array for easier handling
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2])

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()


# def plot_centroids_3d(data_dict, num_select=128):
#     spirals = SphericalSpiral(num_select=num_select)
#     centroids = []
#     labels = []
#
#     # Adjust points for each vector, find its centroid, and store the label
#     for item in data_dict.values():
#         vector = item['features']
#         label = item['cluster']
#         spirals.adjust_points(vector)
#         centroid = spirals.find_centroid()
#         centroids.append(centroid)
#         labels.append(label)
#
#     # Convert to numpy arrays for easier handling
#     centroids = np.array(centroids)
#     labels = np.array(labels)
#
#     print(centroids)
#     print(labels)
#
#     # Setting up the 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Scatter plot for the centroids in 3D, color-coded by cluster
#     scatter = ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=labels, cmap='viridis')
#
#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     ax.set_zlabel('Z Axis')
#     plt.colorbar(scatter, ax=ax, label='Cluster')
#
#     plt.show()


def find_optimal_clusters(centroids, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k).fit(centroids)
        inertias.append(kmeans.inertia_)

    # You could implement an algorithm to find the elbow here.
    # For simplicity, we will assume the elbow is at 3
    optimal_k = 3
    return optimal_k


def plot_centroids_2d(data_dict, num_select=128):
    spirals = SphericalSpiral(num_select=num_select)
    centroids = []

    # Adjust points for each vector and find its centroid
    for item in data_dict.values():
        vector = item['features']
        spirals.adjust_points(vector)
        centroid = spirals.find_centroid()
        centroids.append(centroid)

    # Convert to numpy arrays for easier handling
    centroids = np.array(centroids)

    # Dynamic clustering
    optimal_k = find_optimal_clusters(centroids)
    print(optimal_k)
    kmeans = KMeans(n_clusters=optimal_k)
    labels = kmeans.fit_predict(centroids)

    # Linear regression within each cluster
    lines = []
    for i in range(optimal_k):
        cluster_points = centroids[labels == i]
        lin_reg = LinearRegression()
        # Note: Using only the first two coordinates for 2D regression
        lin_reg.fit(cluster_points[:, :2], cluster_points[:, 1])
        lines.append((lin_reg.coef_[0], lin_reg.intercept_))

    # Scatter plot for the centroids in 2D, color-coded by cluster
    plt.figure(figsize=(8, 6))
    for i in range(optimal_k):
        cluster_points = centroids[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

        # Plotting regression line for each cluster
        lin_reg_coef, lin_reg_intercept = lines[i]
        x_values = np.array([cluster_points[:, 0].min(), cluster_points[:, 0].max()])
        y_values = lin_reg_coef * x_values + lin_reg_intercept
        plt.plot(x_values, y_values, color='black')

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.legend()
    plt.colorbar(label='Cluster')
    plt.show()


def plot_centroids_3d(data_dict, num_select=128):
    spirals = SphericalSpiral(num_select=num_select)
    centroids = []

    # Adjust points for each vector and find its centroid
    for item in data_dict.values():
        vector = item['features']
        spirals.adjust_points(vector)
        centroid = spirals.find_centroid()
        centroids.append(centroid)

    # Convert to numpy arrays for easier handling
    centroids = np.array(centroids)

    # Dynamic clustering
    optimal_k = find_optimal_clusters(centroids)
    print(f'Optimal number of clusters: {optimal_k}')

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=optimal_k)
    labels = kmeans.fit_predict(centroids)  # Using all three coordinates

    # Prepare the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Linear regression and plotting for each cluster in 3D
    for i in range(optimal_k):
        # Select the points that belong to the current cluster
        cluster_points = centroids[labels == i]

        # Perform linear regression on the cluster
        lin_reg = LinearRegression()
        lin_reg.fit(cluster_points[:, :2], cluster_points[:, 2].reshape(-1, 1))

        # Create a meshgrid to plot the regression plane
        X, Y = np.meshgrid(np.linspace(cluster_points[:, 0].min(), cluster_points[:, 0].max(), 10),
                           np.linspace(cluster_points[:, 1].min(), cluster_points[:, 1].max(), 10))
        Z = lin_reg.intercept_ + lin_reg.coef_[0][0] * X + lin_reg.coef_[0][1] * Y

        # Plot the cluster points
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i}')

        # Plot the regression plane
        ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, color='w', edgecolor='k', linewidth=0.5)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title('Clustered Data with Regression Planes')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('Not main file')
