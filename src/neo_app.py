# import os
import sys

import numpy as np

from spherical_spiral import SphericalSpiral
from neo_sphere.spiral import NeoSpiral, calc_nums

from spirals_umap_tsne import load_data, apply_tsne, apply_umap, apply_spiral, apply_neo
from spirals_umap_tsne import plot_mnist, manual_apply_neo, dict_manual_apply_neo, plot_embedding
from spirals_umap_tsne import apply_pca, dict_apply_neo, dict_apply_neo_meshes

from synthetic import generate_synthetic


def mnist_comparison():
    print('Begin neo_app.py')

    images, labels = load_data()

    tsne_results = apply_tsne(images)
    plot_mnist('tsne', tsne_results, labels)

    umap_results = apply_umap(images)
    plot_mnist('umap', umap_results, labels)

    spiral_results = apply_spiral(images)
    plot_mnist('spiral', spiral_results, labels)

    neo_results = apply_neo(images)
    plot_mnist('neo', neo_results, labels)


def spiral_mnist_test():
    print('Begin spiral_props_loop')

    images, labels = load_data()
    print(f"len(images) = {len(images)}")
    print(f"shape: {images[0].shape}")
    print(f"len(vector): {len(images[0])}")

    # num_spirals = 10

    # for i in range(500, 10000, 100):
    #     neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #     plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    # for i in range(10, 200, 10):
    #     neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #     plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    # for i in range(1, 10, 1):
    #     neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #     plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    # for j in range(1, 10, 1):
    #     num_spirals = j
    #     for i in range(4, 8, 1):
    #         neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #         plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    # for j in range(1, 10, 1):
    #     num_spirals = j
    #     for i in range(4, 8, 1):
    #         neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=i)
    #         plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {i} |", neo_results, labels)

    # for i in range(4, 10, 1):
    #     num_points = i
    #     for j in range(0, 3, 1):
    #         num_spirals = np.power(num_points, j)
    #         neo_results = manual_apply_neo(images, num_spirals=num_spirals, num_points=num_points)
    #         plot_mnist(f"neo | num_spirals: {num_spirals} | num_points: {num_points} |", neo_results, labels)

    pca_results = apply_pca(images)
    plot_mnist(f"pca |",
               pca_results,
               labels)

    calc_num_spirals, calc_num_points = calc_nums(len(images[0]))
    neo_results = apply_neo(images)
    plot_mnist(f"neo | num_spirals: {calc_num_spirals} | num_points: {calc_num_points} |",
               neo_results,
               labels)


def synthetic_test():
    print('Begin synthetic_test')
    random_state = 42
    n_samples = 4000
    n_centers = 5
    n_features = 128

    # num_spirals = 10
    # num_points = 10

    data_dict, X, y, X_pca, pca = generate_synthetic(n_samples=n_samples,
                                                     n_centers=n_centers,
                                                     n_features=n_features,
                                                     random_state=random_state)

    print(f"len(data_dict) = {len(data_dict)}")
    print(f"X.shape = {X.shape}")
    print(f"len(X[0]) = {len(X[0])}")

    # neo_results, labels = dict_manual_apply_neo(data_dict, num_spirals=num_spirals, num_points=num_points)
    # plot_embedding(f"neo | num_spirals: {num_spirals} | num_points: {num_points}",
    #                'sklearn synthetic', neo_results, labels)

    plot_embedding('pca', 'synthetic', X_pca, y)

    # for num_spirals in range(1, 11, 1):
    #     for num_points in range(1, 11, 1):
    #         neo_results, labels = dict_manual_apply_neo(data_dict, num_spirals=num_spirals, num_points=num_points)
    #         plot_embedding(f"neo | num_spirals: {num_spirals} | num_points: {num_points}",
    #                        'sklearn synthetic', neo_results, labels)

    # for num_spirals in range(1, 11, 1):
    #     for num_points in range(15, 101, 5):
    #         neo_results, labels = dict_manual_apply_neo(data_dict, num_spirals=num_spirals, num_points=num_points)
    #         plot_embedding(f"neo | num_spirals: {num_spirals} | num_points: {num_points}",
    #                        'sklearn synthetic', neo_results, labels)

    # num_spirals = int(np.ceil(np.sqrt(len(X[0])) / 2))
    # num_points = np.power(2, num_spirals)
    # neo_results, labels = dict_manual_apply_neo(data_dict, num_spirals=num_spirals, num_points=num_points)
    # plot_embedding(f"neo | num_spirals: {num_spirals} | num_points: {num_points}",
    #                'sklearn synthetic', neo_results, labels)

    num_spirals, num_points = calc_nums(len(X[0]))
    neo_results, labels = dict_apply_neo(data_dict)
    plot_embedding(f"neo | num_spirals: {num_spirals} | num_points: {num_points}",
                   'sklearn synthetic', neo_results, labels)


def mesh_test():
    print('Begin mesh_test')
    random_state = 42
    n_samples = 4000
    n_centers = 5
    n_features = 128
    data_dict, X, y, X_pca, pca = generate_synthetic(n_samples=n_samples,
                                                     n_centers=n_centers,
                                                     n_features=n_features,
                                                     random_state=random_state)
    print(f"len(data_dict) = {len(data_dict)}")
    print(f"X.shape = {X.shape}")
    print(f"len(X[0]) = {len(X[0])}")

    num_spirals, num_points = calc_nums(len(X[0]))
    neo_results, labels = dict_apply_neo_meshes(data_dict)
    plot_embedding(f"neo | num_spirals: {num_spirals} | num_points: {num_points}",
                   'sklearn synthetic', neo_results, labels)


def main(args):
    argc = len(args)
    if argc > 1:
        print('Error: Too many arguments')
        return

    if '--mnist-comparison' in args:
        mnist_comparison()
    elif '--mnist' in args:
        spiral_mnist_test()
    elif '--mesh' in args:
        mesh_test()
    elif '--synthetic' in args or argc == 0:
        synthetic_test()
    else:
        print('Error: Invalid arguments')


if __name__ == '__main__':
    main(sys.argv[1:])
