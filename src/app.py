# import os
#
# import numpy as np
# import umap
# from sklearn.manifold import TSNE
# from sklearn.datasets import fetch_openml
# import torch
#
# from spherical_spiral import SphericalSpiral
# from plot import plot_embedding
#
# from pc_processor import PointCloudProcessor
# from pn_autoencoder import PointNetAutoencoder
# from trainer import Train

from synthetic import generate_synthetic
# from test_plot import plot_centroids, plot_centroids_unique_spheres, plot_centroids_3d
import test_plot


# def load_data():
#     fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
#     images = np.array(fashion_mnist.data, dtype=np.float32)
#     labels = np.array(fashion_mnist.target, dtype=int)
#     np.random.seed(42)
#     indices = np.random.choice(len(images), 10000, replace=False)
#     return images[indices], labels[indices]
#
#
# def apply_tsne(data):
#     try:
#         tsne = TSNE(n_components=3, random_state=42)
#         return tsne.fit_transform(data)
#     except Exception as e:
#         print(f"An error occurred while applying t-SNE: {e}")
#         return None
#
#
# def apply_umap(data):
#     try:
#         reducer = umap.UMAP(n_components=3, random_state=42)
#         return reducer.fit_transform(data)
#     except Exception as e:
#         print(f"An error occurred while applying UMAP: {e}")
#         return None
#
#
# def apply_spherical(data):
#     try:
#         centroids = []
#         for vector in data:
#             spirals = SphericalSpiral(num_select=len(vector), num_points=1000)
#             spirals.adjust_points(vector)
#             centroids.append(spirals.find_centroid())
#             spirals = None
#         return np.array(centroids)
#     except Exception as e:
#         print(f"An error occurred while applying spherical spirals: {e}")
#         return None
#
#
# def comparison_main():
#     images, labels = load_data()
#
#     if images is None or labels is None:
#         print("Error, images or labels is None")
#         return
#
#     tsne_result = apply_tsne(images)
#     umap_result = apply_umap(images)
#     spherical_result = apply_spherical(images)
#
#     plot_embedding(tsne_result, labels, 't-SNE on Fashion MNIST')
#     plot_embedding(umap_result, labels, 'UMAP on Fashion MNIST')
#     plot_embedding(spherical_result, labels, 'spherical spiral on Fashion MNIST')
#
#
# def encoder_main(folder_dir='./simple-3d-objects',
#                  pc_path='sphere_point_cloud.ply',
#                  show_pc=False,
#                  train_dim=256,
#                  test_dims=None,
#                  min_points=500,
#                  n_epochs=50,
#                  lr=0.001,
#                  visualize_epochs=10,
#                  visualize_train=False):
#     if test_dims is None:
#         test_dims = [64, 128, 256, 512, 1024, 2048]
#
#     if torch.backends.mps.is_available():
#         device = torch.device('mps')
#     elif torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#
#     # device = torch.device('cpu')
#
#     # Initialize the PointCloudProcessor
#     processor = PointCloudProcessor(folder_dir)
#
#     # Optionally display loaded point clouds
#     if show_pc:
#         for i, point_cloud in enumerate(processor.point_clouds):
#             processor.plot_point_cloud(point_cloud, title=f"Point Cloud {i + 1}")
#
#     # Initialize and train the model on all point clouds
#     print("Training on all point clouds...")
#     trainer = Train()
#
#     # Train the model on all point clouds
#     model = trainer.multiple_point_clouds(processor.point_clouds,
#                                           num_points=min_points,
#                                           latent_size=train_dim,
#                                           epochs=n_epochs,
#                                           lr=lr,
#                                           visualize_every_n_epochs=visualize_epochs,
#                                           condition=visualize_train)
#
#     # After training, map each point cloud file name to its latent vector
#     latent_vectors = {}
#
#     for file_name in os.listdir(folder_dir):
#         if file_name.endswith('.ply'):
#             full_path = os.path.join(folder_dir, file_name)
#             point_cloud = processor.load_point_cloud_from_ply(full_path)
#             point_cloud = processor.resample_point_cloud(point_cloud, min_points)  # Ensure consistency in size
#
#             latent_vector = trainer.encode_point_cloud(model, point_cloud.to(device))
#             latent_vectors[file_name] = latent_vector
#
#     print("Latent vectors for each point cloud file have been saved.")
#     print(latent_vectors)


def test_synthetic_main():
    data_dict, X, y, X_pca, pca = generate_synthetic()
    # test_plot.plot_centroids(X, 5)
    # test_plot.plot_centroids_unique_spheres(X, 5)
    # test_plot.plot_centroids_3d(data_dict)
    # test_plot.plot_centroids_2d(data_dict)
    test_plot.plot_centroids_3d(data_dict, 128)


if __name__ == '__main__':
    # comparison_main()
    # encoder_main()
    test_synthetic_main()
