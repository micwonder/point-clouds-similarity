import numpy as np
import os


def load_point_cloud(file_path: str = r"dataset\point_cloud1.txt"):
    """Load point cloud data from a txt file."""
    return np.loadtxt(file_path)


def load_point_cloud_separate(file_path: str):
    """Load point cloud data from a txt file."""
    # Load data, skip the first row (header), and select columns for latitude, longitude, and altitude
    return np.loadtxt(file_path, delimiter="\t", skiprows=1, usecols=(1, 2, 3))


def load_point_cloud_separate_utm(file_path):
    """Load point cloud and convert coordinates to UTM."""
    data = np.loadtxt(file_path, delimiter="\t", skiprows=1, usecols=(1, 2, 3))
    utm_coords = np.array([lat_lon_to_utm(lat, lon) + (alt,) for lat, lon, alt in data])
    return utm_coords


def lat_lon_to_utm(latitude, longitude):
    """Convert latitude and longitude to UTM coordinates."""
    import utm

    return utm.from_latlon(latitude, longitude)[
        :2
    ]  # Returns (easting, northing, zone_number, zone_letter)


def load_target_point_cloud_by_directions(path: str):
    point_cloud_x = np.load(os.path.join(path, "x.npy"))
    point_cloud_z = np.load(os.path.join(path, "y.npy"))
    point_cloud_y = np.load(os.path.join(path, "z.npy"))

    # Combine x, y, z arrays into a single array to construct point cloud
    point_cloud = np.stack((point_cloud_x, point_cloud_y, point_cloud_z), axis=1)

    # Preprocessing: Standardize altitude
    mean_alt = np.mean(point_cloud_z)
    std_alt = np.std(point_cloud_z)
    # point_cloud[:, 2] = (point_cloud[:, 2] - mean_alt) / std_alt
    # point_cloud[:, 2] = (point_cloud[:, 2]) / std_alt

    # point_cloud[:, 2] = -point_cloud[:, 2]

    return point_cloud


if __name__ == "__main__":
    from myploty import plot_point_cloud_plotly

    point_cloud = load_target_point_cloud_by_directions("reconstruction_result")
    # point_cloud = load_target_point_cloud_by_directions("sample_output")
    # point_cloud = load_point_cloud()
    # point_cloud = load_point_cloud(file_path=r"dataset\airport-data\result\merged.txt")

    plot_point_cloud_plotly(position="NaN", similarity=1.0, cloud_data=point_cloud)
