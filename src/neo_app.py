# import os
import sys
import time

import numpy as np

from spherical_spiral import SphericalSpiral
from neo_sphere.spiral import NeoSpiral, calc_nums

from spirals_umap_tsne import load_data, apply_tsne, apply_umap, apply_spiral, apply_neo, apply_pca_3d
from spirals_umap_tsne import plot_mnist, manual_apply_neo, dict_manual_apply_neo, plot_embedding
from spirals_umap_tsne import apply_pca, dict_apply_neo, dict_apply_neo_meshes, plot_mnist_3d

from synthetic import generate_synthetic


def mnist_comparison():
    print('Begin mnist_comparison')

    images, labels = load_data()

    st_time = time.time()
    pca_results = apply_pca(images)
    pca_time = time.time() - st_time
    plot_mnist('pca', pca_results, labels)

    st_time = time.time()
    tsne_results = apply_tsne(images)
    tsne_time = time.time() - st_time
    plot_mnist('tsne', tsne_results, labels)

    st_time = time.time()
    umap_results = apply_umap(images)
    umap_time = time.time() - st_time
    plot_mnist('umap', umap_results, labels)

    st_time = time.time()
    spiral_results = apply_spiral(images)
    spiral_time = time.time() - st_time
    plot_mnist('spiral', spiral_results, labels)

    st_time = time.time()
    neo_results = apply_neo(images)
    neo_spiral_time = time.time() - st_time
    plot_mnist('neo', neo_results, labels)

    print(f"pca_time:          {pca_time}")
    print(f"tsne_time:         {tsne_time}")
    print(f"umap_time:         {umap_time}")
    print(f"spiral_time:       {spiral_time}")
    print(f"neo_spiral_time:   {neo_spiral_time}")


def mnist_comparison_3d():
    print('Begin mnist_comparison_3d')

    images, labels = load_data()

    st_time = time.time()
    pca_results = apply_pca_3d(images)
    pca_time = time.time() - st_time
    plot_mnist_3d('pca', pca_results, labels)

    st_time = time.time()
    tsne_results = apply_tsne(images)
    tsne_time = time.time() - st_time
    plot_mnist_3d('tsne', tsne_results, labels)

    st_time = time.time()
    umap_results = apply_umap(images)
    umap_time = time.time() - st_time
    plot_mnist_3d('umap', umap_results, labels)

    st_time = time.time()
    spiral_results = apply_spiral(images)
    spiral_time = time.time() - st_time
    plot_mnist_3d('spiral', spiral_results, labels)

    st_time = time.time()
    neo_results = apply_neo(images)
    neo_spiral_time = time.time() - st_time
    plot_mnist_3d('neo', neo_results, labels)

    print(f"pca_time:          {pca_time}")
    print(f"tsne_time:         {tsne_time}")
    print(f"umap_time:         {umap_time}")
    print(f"spiral_time:       {spiral_time}")
    print(f"neo_spiral_time:   {neo_spiral_time}")


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
    elif '--mnist-comparison-3d' in args:
        mnist_comparison_3d()
    else:
        print('Error: Invalid arguments')


if __name__ == '__main__':
    main(sys.argv[1:])
