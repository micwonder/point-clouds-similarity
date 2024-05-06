import numpy as np
from scipy.spatial import KDTree
from computing_surface_normals import compute_normals, normalize_point_cloud
from load_point_cloud import load_point_cloud
from myploty import plot_point_cloud_plotly, plot_clusters_plotly


def region_growing(point_cloud, normals, angle_threshold=-91, distance_threshold=0.1):
    """
    Segment point cloud using region growing based on normals similarity and spatial proximity.
    :param point_cloud: Nx3 numpy array of points.
    :param normals: Nx3 numpy array of corresponding normals.
    :param angle_threshold: Maximum angle (in degrees) between normals to consider them similar.
    :param distance_threshold: Maximum spatial distance for points to be considered in the same region.
    :return: List of clusters, each is a list of indices into the point cloud.
    """
    point_count = point_cloud.shape[0]
    visited = np.zeros(point_count, dtype=bool)
    clusters = []
    tree = KDTree(point_cloud)

    # Convert angle threshold from degrees to cosine of the angle for comparison
    cos_angle_threshold = np.cos(np.radians(angle_threshold))
    print(cos_angle_threshold)
    print(np.radians(angle_threshold))
    print(angle_threshold)

    for i in range(point_count):
        if not visited[i]:
            visited[i] = True
            cluster = [i]
            # Use a list as a queue for the region growing
            queue = [i]
            while queue:
                current_point = queue.pop(0)
                current_normal = normals[current_point]

                # Query for nearby points
                indices = tree.query_ball_point(
                    point_cloud[current_point], distance_threshold
                )
                for idx in indices:
                    if (
                        not visited[idx]
                        and np.dot(normals[idx], current_normal) >= cos_angle_threshold
                    ):
                        visited[idx] = True
                        queue.append(idx)
                        cluster.append(idx)
            clusters.append(cluster)

    return clusters


def simulate_parameters(point_cloud, normals, angle_thresholds, distance_thresholds):
    """
    Simulate different parameter combinations and plot the resulting clusters.
    :param point_cloud: Nx3 numpy array of points.
    :param angle_thresholds: List of angle threshold values to test.
    :param distance_thresholds: List of distance threshold values to test.
    """
    clusters_list = []
    angle_list = []
    distance_list = []
    for angle_threshold in angle_thresholds:
        for distance_threshold in distance_thresholds:
            print(
                f"Simulating with angle threshold={angle_threshold}, distance threshold={distance_threshold}"
            )
            clusters = region_growing(
                point_cloud, normals, angle_threshold, distance_threshold
            )
            # plot_clusters_plotly(point_cloud, clusters)
            print(f"Number of clusters found: {len(clusters)}")
            clusters_list.append(len(clusters))
            angle_list.append(angle_threshold)
            distance_list.append(distance_threshold)
            # print(clusters)

    return np.stack(
        (np.array(angle_list), np.array(distance_list), np.array(clusters_list)), axis=1
    )


if __name__ == "__main__":
    # Example usage:
    # point_cloud = np.random.rand(100, 3)
    point_cloud = load_point_cloud()
    # point_cloud = point_cloud[::17]
    point_cloud = normalize_point_cloud(point_cloud=point_cloud)
    normals = compute_normals(point_cloud)
    clusters = region_growing(point_cloud=point_cloud, normals=normals)

    plot_clusters_plotly(point_cloud=point_cloud, clusters=clusters)

    # # Simulate different parameter combinations and plot the resulting clusters
    # cloud_data = simulate_parameters(
    #     point_cloud=point_cloud,
    #     normals=normals,
    #     angle_thresholds=list(range(90, 101, 1)),
    #     distance_thresholds=[i / 500 for i in range(50, 70, 1)],
    # )

    # plot_point_cloud_plotly(
    #     position="NaN",
    #     similarity=1.0,
    #     cloud_data=cloud_data,
    # )
