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

## Relevant Classes

### neo_sphere.spiral.NeoSpiral
- **CURRENT**
- Version 0.2 of project concept
- Required Parameters:
  - `vector`
- Optional Parameters:
  - `construct_mesh` (Whether to automatically construct Ball-Pivot mesh)
  - `construct_mesh_poisson` (Whether to automatically construct Poisson Recon mesh)
  - `manual`  (NOTE: this must be set to True to manually specify num_spirals and num_points)
  - `num_spirals`
  - `num_points`
  - `title`  (Note: this is only used for the title of plots)
- Instantiation process:
  - `spiral = NeoSpiral(vector)`
- Accessors:
  - `centroid = spiral.get_centroid()`
  - `rod_len = spiral.get_rod_length()`
- Construct Meshes:
  - Note: this is only necessary if the optional parameters to construct meshes weren't supplied
  - `spiral.reconstruct_surface_mesh()`
  - `spiral.reconstruct_surface_mesh_poisson()`
- Visualize Meshes:
  - `spiral.visualize_mesh()`
  - `spiral.visualize_mesh_matplotlib()`
- Save Meshes:
  - `spiral.save_meshes(output_path, timmed_path)`


### spherical_spiral.SphericalSpiral
- **DEPRECATED**
- Version 0.1 of project concept
- Parameters:
  - `num_spirals`
  - `num_points`
  - `num_select` (NOTE: this must be equal to the vector length, has to be supplied manually)
- Instantiation process:
  - `spiral = SphericalSpiral(num_spirals, num_points, num_select=len(vector))`
  - `spiral.generate_spaced_spherical_spiral()`
  - `spiral.select_evenly_spaced_points()`
  - `spiral.adjust_points(vector)`
  - `centroid = spiral.find_centroid()`
  - `rod_len = spiral.calculate_rod_length()`
- Meshes:
  - Ball Pivoting:
    - `spiral.reconstruct_surface_mesh()`
  - Poisson Surface Reconstruction:
    - `spiral.reconstruct_surface_mesh_poisson()`
  - Visualization:
    - `spiral.visualize_mesh(plot_title)`
    - or
    - `spiral.visualize_mesh_matplotlib()`
  - Save Meshes:
    - `spiral.save_meshes(output_path, trimmed_output_path)`