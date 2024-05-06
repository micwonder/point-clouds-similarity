import numpy as np
from scipy.spatial import KDTree
from myploty import plot_point_cloud, plot_point_cloud_plotly
import matplotlib.pyplot as plt


def cartesian_to_polar(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    return r, theta, phi


def generate_synthetic_point_cloud(num_points=1000):
    # Initialize the point cloud array
    point_cloud = np.zeros((num_points, 3))
    print(len(point_cloud))

    # Generate flat ground points
    num_ground_points = num_points // 2
    ground_points = np.random.rand(num_ground_points, 3)
    ground_points[:, 2] = 10  # Set z to 0 to make it a flat ground

    # Generate vertical structures (e.g., vertical posts)
    num_vertical_points = num_points // 4
    vertical_points = np.random.rand(num_vertical_points, 3)
    vertical_points[:, 0] = np.random.choice(
        np.linspace(10, 20, 5), num_vertical_points
    )  # Random x-positions for posts
    vertical_points[:, 1] = np.random.choice(
        np.linspace(10, 20, 5), num_vertical_points
    )  # Random y-positions for posts
    vertical_points[:, 2] = np.linspace(
        0, 50, num_vertical_points
    )  # Height of the posts

    # Add random noise
    num_noise_points = num_points - num_ground_points - num_vertical_points
    noise_points = np.random.rand(num_noise_points, 3) * 50  # Random noise all over

    # Combine all parts into one array
    point_cloud[:num_ground_points] = ground_points
    point_cloud[num_ground_points : num_ground_points + num_vertical_points] = (
        vertical_points
    )
    point_cloud[num_ground_points + num_vertical_points :] = noise_points

    return point_cloud


def remove_flat_regions_polar(
    point_cloud, elevation_threshold=0.1, radius=0.5, min_points=10
):
    # Convert to polar coordinate
    polar_points = np.array([cartesian_to_polar(*point) for point in point_cloud])
    plot_point_cloud_plotly(position="NaN", similarity=1.0, cloud_data=polar_points)
    # KD-Tree for efficient neighbor searches
    tree = KDTree(polar_points[:, 1:3])  # Using theta and phi for tree

    non_flat_indices = []

    # Iterate over each point and determine if it's part of a flat region
    for i, (r, theta, phi) in enumerate(polar_points):
        # Query points within a certain angular radius
        neighbors_idx = tree.query_ball_point([theta, phi], r=radius)

        # Check the elevation variance among neighbors
        elevations = polar_points[neighbors_idx, 2]  # phi values of neighbors
        if np.std(elevations) > elevation_threshold:
            non_flat_indices.append(i)

    # Filter the point cloud to remove flat regions
    filtered_point_cloud = point_cloud[non_flat_indices]
    plot_point_cloud_plotly(
        position="NaN", similarity=1.0, cloud_data=filtered_point_cloud
    )

    return filtered_point_cloud


def remove_flat_regions(point_cloud, elevation_threshold=0.05):
    # Dummy function to simulate removal of flat regions
    return point_cloud[point_cloud[:, 2] > elevation_threshold]


if __name__ == "__main__":
    # Random point cloud simulation
    # point_cloud = np.random.rand(1000, 3) * 100
    point_cloud = generate_synthetic_point_cloud()
    plot_point_cloud_plotly(position="NaN", similarity=1.0, cloud_data=point_cloud)
    # next(plot_point_cloud(position="NaN", similarity=1.0, cloud_data=point_cloud))
    # plt.show()

    point_cloud = remove_flat_regions(
        point_cloud=point_cloud, elevation_threshold=10.05
    )
    plot_point_cloud_plotly(position="NaN", similarity=1.0, cloud_data=point_cloud)
