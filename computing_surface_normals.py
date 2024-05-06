import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import eigh
from myploty import plot_cloud_with_normals_plotly, plot_cloud_with_normals
from load_point_cloud import load_point_cloud


def normalize_point_cloud(point_cloud):
    """
    Normalize the point cloud so that longitude, latitude, and altitude are on a similar scale.
    :param point_cloud: Nx3 numpy array where columns are longitude, latitude, and altitude.
    :return: Normalized point cloud.
    """
    mean_vals = np.mean(point_cloud, axis=0)
    std_vals = np.std(point_cloud, axis=0)
    normalized_cloud = (point_cloud - mean_vals) / std_vals
    return normalized_cloud


def compute_normals(point_cloud, k=20, r_factor=1.5):
    """
    Compute surface normals for each point in the point cloud using an adaptive neighborhood size.
    :param point_cloud: Nx3 numpy array of 3D points.
    :param k: Initial number of neighbors to consider for the covariance matrix.
    :param r_factor: Factor to adjust the neighborhood radius based on local variance.
    :return: Nx3 numpy array of normals.
    """
    tree = KDTree(point_cloud)
    normals = np.zeros_like(point_cloud)

    for i, point in enumerate(point_cloud):
        # Step 1: Determine initial neighborhood based on k nearest neighbors
        distances, indices = tree.query(point, k=k)
        initial_neighbors = point_cloud[indices]

        # Step 2: Calculate the local variance and adjust the radius
        local_var = np.var(distances)
        adaptive_radius = r_factor * np.sqrt(local_var)

        # Step 3: Redefine neighbors based on the adaptive radius
        indices = tree.query_ball_point(point, r=adaptive_radius)
        adaptive_neighbors = point_cloud[indices]

        # Step 4: Compute covariance matrix from adaptive neighbors
        if len(adaptive_neighbors) > 2:
            cov_matrix = np.cov(
                adaptive_neighbors - adaptive_neighbors.mean(axis=0), rowvar=False
            )
            # Step 5: Compute eigenvalues and eigenvectors; the normal is the eigenvector with the smallest eigenvalue
            eigenvalues, eigenvectors = eigh(cov_matrix)
            normal = eigenvectors[:, np.argmin(eigenvalues)]
            normals[i] = normal * np.sign(normal[2])  # Ensuring the normal is outward
        else:
            normals[i] = [np.nan, np.nan, np.nan]  # Not enough points to define a plane

    return normals


if __name__ == "__main__":
    # Example usage
    # point_cloud = np.random.rand(100, 3)  # Simulate a small random point cloud
    point_cloud = load_point_cloud()
    point_cloud = point_cloud[::17]
    point_cloud = normalize_point_cloud(point_cloud=point_cloud)

    normals = compute_normals(point_cloud)
    print("Computed Normals:\n", normals)

    # plot_cloud_with_normals(point_cloud=point_cloud, normals=normals)
    plot_cloud_with_normals_plotly(point_cloud=point_cloud, normals=normals)
