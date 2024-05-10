# def find_matching_robots(map, query):
#     rows = len(map)
#     cols = len(map[0])

#     # Preprocess to calculate distances to nearest blockers in all four directions
#     left = [[0] * cols for _ in range(rows)]
#     right = [[0] * cols for _ in range(rows)]
#     up = [[0] * cols for _ in range(rows)]
#     down = [[0] * cols for _ in range(rows)]

#     for i in range(rows):
#         for j in range(cols):
#             if map[i][j] == "X":
#                 left[i][j] = 0
#                 up[i][j] = 0
#             else:
#                 left[i][j] = left[i][j - 1] + 1 if j > 0 else 1
#                 up[i][j] = up[i - 1][j] + 1 if i > 0 else 1

#     for i in range(rows - 1, -1, -1):
#         for j in range(cols - 1, -1, -1):
#             if map[i][j] == "X":
#                 right[i][j] = 0
#                 down[i][j] = 0
#             else:
#                 right[i][j] = right[i][j + 1] + 1 if j < cols - 1 else 1
#                 down[i][j] = down[i + 1][j] + 1 if i < rows - 1 else 1

#     ans = []

#     for i in range(rows):
#         for j in range(cols):
#             if map[i][j] == "O":
#                 # Check if distances match the query
#                 if (
#                     left[i][j] == query[0]
#                     and right[i][j] == query[3]
#                     and up[i][j] == query[1]
#                     and down[i][j] == query[2]
#                 ):
#                     ans.append([i, j])

#     return ans


# map = [
#     ["O", "E", "E", "E", "X"],
#     ["E", "O", "X", "X", "X"],
#     ["E", "E", "E", "E", "E"],
#     ["X", "E", "O", "E", "E"],
#     ["X", "E", "X", "E", "X"],
# ]
# query = [2, 2, 4, 1]
# print(find_matching_robots(map, query))


import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import eigh
from myploty import plot_cloud_with_normals_plotly, plot_cloud_with_normals
from load_point_cloud import load_point_cloud


def normalize_point_cloud(point_cloud):
    mean_vals = np.mean(point_cloud, axis=0)
    std_vals = np.std(point_cloud, axis=0)
    normalized_cloud = (point_cloud - mean_vals) / std_vals
    return normalized_cloud


def compute_normals(point_cloud, k=20, r_factor=1.5):
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
