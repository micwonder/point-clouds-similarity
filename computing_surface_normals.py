import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import eigh
from myploty import plot_cloud_with_normals_plotly, plot_cloud_with_normals
from load_point_cloud import load_point_cloud
import time


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


def compute_normals(point_cloud, k=20):
    """
    Compute normals and curvature for each point in the point cloud.
    Args:
    - point_cloud (np.array): N x 3 numpy array of coordinates.
    - k (int): Number of nearest neighbors to consider for the neighborhood.

    Returns:
    - normals (np.array): N x 3 array of normals.
    - curvatures (np.array): N array of curvature values.
    """
    tree = KDTree(point_cloud)
    normals = np.zeros_like(point_cloud)
    curvatures = np.zeros(point_cloud.shape[0])

    for i, point in enumerate(point_cloud):
        # Find k-nearest neighbors (including the point itself)
        distances, indices = tree.query(point, k=k + 1)

        # Extract the neighborhood points
        neighbors = point_cloud[indices]

        # Compute the mean (centroid) of the neighbors
        mu = np.mean(neighbors, axis=0)

        # Compute the covariance matrix of the neighborhood
        covariance_matrix = np.cov(neighbors - mu, rowvar=False, bias=True)

        # Perform eigen decomposition to find the normals and curvature
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # The normal vector is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, 0]
        curvature = eigenvalues[0] / eigenvalues.sum()

        # Ensuring the normal points towards the viewpoint
        if np.dot(normal, point - mu) > 0:
            normal = -normal

        normals[i] = normal
        curvatures[i] = curvature

    return normals, curvatures


if __name__ == "__main__":
    # Example usage
    # point_cloud = np.random.rand(100, 3)  # Simulate a small random point cloud
    print("Loading ...")

    timestamp = time.time()
    point_cloud = load_point_cloud(file_path=r"dataset\boss\2.txt")
    print("Loaded: {:.3f}".format(time.time() - timestamp))

    point_cloud = point_cloud[:10000]
    print("Filtered: {:.3f}".format(time.time() - timestamp))

    point_cloud = normalize_point_cloud(point_cloud=point_cloud)
    print("Normalized: {:.3f}".format(time.time() - timestamp))

    normals, curvatures = compute_normals(point_cloud)
    print("Normals computed: {:3f}".format(time.time() - timestamp))
    # print("Computed Normals:\n", normals)
    # for normal in normals:
    #     print(normal)

    print("~~~~~~~~~~~~~~~~~~~~~~~")
    print(curvatures)

    plot_cloud_with_normals(point_cloud=point_cloud, normals=normals)
    # plot_cloud_with_normals_plotly(point_cloud=point_cloud, normals=normals)
    # plot_cloud_with_normals_plotly(point_cloud=point_cloud, normals=normals)

    print("Successfully executed: {:3f}".format(time.time() - timestamp))
