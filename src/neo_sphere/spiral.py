from hashlib import md5

import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class NeoSpiral(object):
    def __init__(self, vector, manual=False, num_spirals=10, num_points=5000, title='',
                 construct_mesh=False, construct_mesh_poisson=False):
        if title == '':
            self.title = md5(str(vector).encode()).hexdigest()[:10]
        else:
            self.title = title
        self.vector = vector
        if manual:
            self.num_spirals = num_spirals
            self.num_points = num_points
        else:
            self.num_spirals, self.num_points = calc_nums(len(vector))
        self.num_select = len(vector)
        ####################################
        # generate_spaced_spherical_spiral()
        theta = np.linspace(0, self.num_spirals * 2 * np.pi, self.num_points)
        z = np.linspace(-1, 1, self.num_points)
        radius = np.sqrt(1 - z ** 2)
        self.x = radius * np.cos(theta)
        self.y = radius * np.sin(theta)
        self.z = z
        ####################################
        # select_evenly_spaced_points()
        indices = np.linspace(0, len(self.x) - 1, self.num_select, dtype=int)
        self.x_selected, self.y_selected, self.z_selected = self.x[indices], self.y[indices], self.z[indices]
        ####################################
        # adjust_points()
        norms = np.sqrt(self.x_selected ** 2 + self.y_selected ** 2 + self.z_selected ** 2)
        x_unit = self.x_selected / norms
        y_unit = self.y_selected / norms
        z_unit = self.z_selected / norms
        self.x_adjusted = self.x_selected + x_unit * vector
        self.y_adjusted = self.y_selected + y_unit * vector
        self.z_adjusted = self.z_selected + z_unit * vector
        ####################################
        # find_centroid()
        adjusted_points = np.vstack((self.x_adjusted, self.y_adjusted, self.z_adjusted)).T
        self.centroid = np.mean(adjusted_points, axis=0)
        ####################################
        # calculate_rod_length()
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dz = np.diff(self.z)
        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        self.rod_length = np.sum(distances)
        ####################################
        if construct_mesh:
            # reconstruct_surface_mesh()
            points = np.vstack((self.x_adjusted, self.y_adjusted, self.z_adjusted)).T
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2]
            radii = o3d.utility.DoubleVector(radii)
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=pcd, radii=radii)
        ####################################
        if construct_mesh_poisson:
            # reconstruct_surface_mesh_poisson()
            points = np.vstack((self.x_adjusted, self.y_adjusted, self.z_adjusted)).T
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # Poisson surface reconstruction
            self.mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            self.original_mesh_poisson = self.mesh_poisson  # Keep the original mesh
            # Optional: trim the mesh based on the densities
            densities = np.asarray(densities)
            density_threshold = np.percentile(densities, 50)
            self.mesh_poisson = self.mesh_poisson.select_by_index((densities > density_threshold).nonzero()[0])
            # Optionally set a uniform color for both meshes
            self.original_mesh_poisson.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color for original mesh
            self.mesh_poisson.paint_uniform_color([1, 0.706, 0])  # Light orange for the trimmed mesh

    def get_centroid(self):
        return self.centroid

    def get_rod_length(self):
        return self.rod_length

    def reconstruct_surface_mesh(self):
        points = np.vstack((self.x_adjusted, self.y_adjusted, self.z_adjusted)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = o3d.utility.DoubleVector([avg_dist, avg_dist * 2])
        self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=pcd, radii=radii)

    def reconstruct_surface_mesh_poisson(self):
        points = np.vstack((self.x_adjusted, self.y_adjusted, self.z_adjusted)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Poisson surface reconstruction
        self.mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        self.original_mesh_poisson = self.mesh_poisson  # Keep the original mesh

        # Optional: trim the mesh based on the densities
        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, 50)
        self.mesh_poisson = self.mesh_poisson.select_by_index((densities > density_threshold).nonzero()[0])

        # Optionally set a uniform color for both meshes
        self.original_mesh_poisson.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color for original mesh
        self.mesh_poisson.paint_uniform_color([1, 0.706, 0])  # Light orange for the trimmed mesh

    def visualize_mesh(self):
        if hasattr(self, 'mesh'):
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(self.title)
            mesh_collection = Poly3DCollection(vertices[triangles], alpha=0.6, edgecolor='darkblue')
            mesh_collection.set_facecolor('lightblue')
            ax.add_collection3d(mesh_collection)
            ax.set_xlim([min(vertices[:, 0]), max(vertices[:, 0])])
            ax.set_ylim([min(vertices[:, 1]), max(vertices[:, 1])])
            ax.set_zlim([min(vertices[:, 2]), max(vertices[:, 2])])
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')
            plt.tight_layout()
            plt.show()
        else:
            print("Mesh has not been generated yet.")

    def visualize_mesh_matplotlib(self):
        if hasattr(self, 'original_mesh_poisson') and hasattr(self, 'mesh_poisson'):
            # Prepare for two side-by-side plots
            fig = plt.figure(figsize=(20, 10))
            # Visualize original mesh
            ax1 = fig.add_subplot(121, projection='3d')
            vertices = np.asarray(self.original_mesh_poisson.vertices)
            triangles = np.asarray(self.original_mesh_poisson.triangles)
            ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, c='r')
            for triangle in triangles:
                for i in range(3):
                    start_point = vertices[triangle[i]]
                    end_point = vertices[triangle[(i + 1) % 3]]
                    ax1.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                             [start_point[2], end_point[2]], 'k', linewidth=0.5)
            ax1.set_title(f"{self.title} Original Mesh")
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            # Visualize trimmed mesh
            ax2 = fig.add_subplot(122, projection='3d')
            vertices = np.asarray(self.mesh_poisson.vertices)
            triangles = np.asarray(self.mesh_poisson.triangles)
            ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, c='r')
            for triangle in triangles:
                for i in range(3):
                    start_point = vertices[triangle[i]]
                    end_point = vertices[triangle[(i + 1) % 3]]
                    ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                             [start_point[2], end_point[2]], 'k', linewidth=0.5)
            ax2.set_title(f"{self.title} Trimmed Mesh")
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            plt.tight_layout()
            plt.show()
        else:
            print("Poisson meshes have not been generated yet.")


def calc_nums(vector_len):
    # num_spirals = int(np.ceil(np.sqrt(vector_len) / 2))
    num_spirals = int(np.ceil(np.cbrt(vector_len)))
    # num_points = num_spirals * 2
    # num_points = np.power(num_spirals, 2)
    # num_points = int(vector_len / num_spirals)
    num_points = vector_len
    return num_spirals, num_points


if __name__ == "__main__":
    print('Error: not main file')
