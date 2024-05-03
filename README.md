# Spherical_Spirals

## Authors

- [Elvis Kimara (ekimara@iastate.edu)](https://github.com/kelvis24)
- [Tara Munjuluri (taram27@iastate.edu)](https://github.com/TaraMunjuluri)
- [Rolf Anderson (rolf@iastate.edu)](https://github.com/asaprolfy)

## About

This repo contains a new method for visualizing and comparing latent vectors.
A `spherical_spiral` is a process for dimension reduction for vectors, comparable to PCA.

## Dependencies
    - python3.9
    - torch
    - numpy
    - open3d
    - matplotlib
    - plotly
    - scipy
    - scikit-learn
    - umap-learn
    - plyfile
    - plotly

## Arguments
*Note: only one test method can be run per execution*
- `--synthetic`
  - Run dimension reduction test on synthetic data, with PCA as comparison
- `--mesh`
  - Visualize Spherical Spiral meshes on synthetic data
- `--mnist`
  - Run dimension reduction test on Fashion MNIST data, with PCA as comparison
- `--mnist-comparison`
  - Run dimension reduction test on Fashion MNIST data, with t-SNE and UMAP as comparisons

## How to Run
- Create Python virtual environment
- `python3.9 -m venv venv`
- Install requirements
- `venv/bin/pip3.9 install -r requirements.txt`
- Run program
- `venv/bin/python3.9 src/neo_app.py <arguments>`