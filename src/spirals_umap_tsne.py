import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from spherical_spiral import SphericalSpiral
from neo_sphere.spiral import NeoSpiral


def load_data():
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
    images = np.array(fashion_mnist.data, dtype=np.float32)
    labels = np.array(fashion_mnist.target, dtype=int)
    np.random.seed(42)
    indices = np.random.choice(len(images), 10000, replace=False)
    return images[indices], labels[indices]


def apply_pca(data):
    try:
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(data)
    except Exception as e:
        print(f"An error occured while applying PCA: {e}")
        return None


def apply_tsne(data):
    try:
        tsne = TSNE(n_components=3, random_state=42)
        return tsne.fit_transform(data)
    except Exception as e:
        print(f"An error occurred while applying t-SNE: {e}")
        return None


def apply_umap(data):
    try:
        reducer = UMAP(n_components=3, random_state=42)
        return reducer.fit_transform(data)
    except Exception as e:
        print(f"An error occurred while applying UMAP: {e}")
        return None


def apply_spiral(data):
    try:
        centroids = []
        for vector in data:
            spirals = SphericalSpiral(num_select=len(vector), num_points=1000)
            spirals.adjust_points(vector)
            centroids.append(spirals.find_centroid())
            spirals = None
        return np.array(centroids)
    except Exception as e:
        print(f"An error occurred while applying spherical spirals: {e}")
        return None


def apply_neo(data):
    try:
        centroids = []
        for vector in data:
            spiral = NeoSpiral(vector=vector)
            centroids.append(spiral.get_centroid())
        return np.array(centroids)
    except Exception as e:
        print(f"An error occurred while applying spherical spirals: {e}")
        return None


def manual_apply_neo(data, num_spirals, num_points):
    try:
        centroids = []
        for vector in data:
            spiral = NeoSpiral(vector=vector, manual=True, num_spirals=num_spirals, num_points=num_points)
            centroids.append(spiral.get_centroid())
        return np.array(centroids)
    except Exception as e:
        print(f"An error occurred while applying spherical spirals: {e}")
        return None


def dict_apply_neo(data_dict):
    centroids = []
    labels = []
    for item in data_dict.values():
        vector = item['features']
        label = item['cluster']
        spiral = NeoSpiral(vector=vector)
        centroids.append(spiral.get_centroid())
        labels.append(label)
    return np.array(centroids), np.array(labels)


def dict_apply_neo_meshes(data_dict):
    centroids = []
    labels = []
    for item in data_dict.values():
        vector = item['features']
        label = item['cluster']
        spiral = NeoSpiral(vector=vector, construct_mesh=True, construct_mesh_poisson=True)
        centroids.append(spiral.get_centroid())
        spiral.visualize_mesh()
        spiral.visualize_mesh_matplotlib()
        labels.append(label)
    return np.array(centroids), np.array(labels)


def dict_manual_apply_neo(data_dict, num_spirals, num_points):
    centroids = []
    labels = []
    for item in data_dict.values():
        vector = item['features']
        label = item['cluster']
        spiral = NeoSpiral(vector=vector, manual=True, num_spirals=num_spirals, num_points=num_points)
        centroids.append(spiral.get_centroid())
        labels.append(label)
    return np.array(centroids), np.array(labels)


def plot_mnist(algo_name, result, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(result[:, 0], result[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class Labels')
    plt.title(f"{algo_name} of Fashion MNIST Embeddings")
    plt.xlabel(f"{algo_name} 1")
    plt.ylabel(f"{algo_name} 2")
    plt.show()


def plot_embedding(algo_name, dataset_name, results, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(results[:, 0], results[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class Labels')
    plt.title(f"{algo_name} | {dataset_name} Embeddings")
    plt.xlabel(f"{algo_name} 1")
    plt.ylabel(f"{algo_name} 2")
    plt.show()


if __name__ == '__main__':
    print('Error: not a main file')
