import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
# from mpl_toolkits.mplot3d import Axes3D

from spherical_spiral import SphericalSpiral


def plot_centroids_2d(vectors):
    spirals = SphericalSpiral(num_select=64)
    centroids = []

    # Adjust points for each vector and find its centroid
    for vector in vectors:
        spirals.adjust_points(vector)
        centroid = spirals.find_centroid()
        centroids.append(centroid)

    # Setting up the 2D plot
    fig, ax = plt.subplots()

    # Scatter plot for the centroids in 2D (just using the first two coordinates)
    centroids = np.array(centroids)  # Convert to numpy array for easier handling
    ax.scatter(centroids[:, 0], centroids[:, 1])

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    plt.show()


def plot_centroids_3d(data_dict, num_select=128):
    spirals = SphericalSpiral(num_select=num_select)
    centroids = []
    labels = []

    # Adjust points for each vector, find its centroid, and store the label
    for item in data_dict.values():
        vector = item['features']
        label = item['cluster']
        spirals.adjust_points(vector)
        centroid = spirals.find_centroid()
        centroids.append(centroid)
        labels.append(label)

    # Convert to numpy arrays for easier handling
    centroids = np.array(centroids)
    labels = np.array(labels)

    print(centroids)
    print(labels)

    # Setting up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the centroids in 3D, color-coded by cluster
    scatter = ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=labels, cmap='viridis')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.colorbar(scatter, ax=ax, label='Cluster')

    plt.show()


def plot_embedding(data, labels, title):
    if data is not None:
        fig = px.scatter_3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            color=labels.astype(str), labels={'color': 'Class'},
            title=title
        )
        fig.update_traces(marker=dict(size=2))
        fig.show()
    else:
        print("No data available for plotting.")


if __name__ == '__main__':
    print('Error: not main file')
