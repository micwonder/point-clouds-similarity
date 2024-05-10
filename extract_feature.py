import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from computing_surface_normals import compute_normals, normalize_point_cloud
from load_point_cloud import load_point_cloud
from myploty import plot_cloud_with_normals_plotly
from segmentation_and_clustering import region_growing


def fit_plane_svd(points):
    """
    Fit a plane to a set of points using Singular Value Decomposition (SVD).
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    _, _, vh = svd(points_centered, full_matrices=False)
    normal = vh[2, :]
    return normal, centroid


def fit_line_svd(points):
    """
    Fit a line to a set of points using Singular Value Decomposition (SVD).
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    _, _, vh = svd(points_centered, full_matrices=False)
    direction = vh[0, :]
    return direction, centroid


def evaluate_fit_quality(points, model, normal=None):
    """
    Evaluate the quality of fit using residual error and explained variance ratio.
    """
    if model == "plane":
        distances = np.abs(np.dot(points - points.mean(axis=0), normal))
        residual_error = np.mean(distances)
        pca = PCA(n_components=2)
        pca.fit(points)
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    else:  # line
        pca = PCA(n_components=1)
        pca.fit(points)
        explained_variance_ratio = pca.explained_variance_ratio_[0]
        residual_error = np.mean(
            np.min(
                cdist(
                    points,
                    np.outer(np.dot(points, pca.components_[0]), pca.components_[0])
                    + points.mean(axis=0),
                ),
                axis=1,
            )
        )

    return explained_variance_ratio, residual_error


def extract_features(point_cloud, clusters):
    """
    Extract features by fitting planes and lines to each cluster and selecting the best model based on quality metrics.
    """
    features = []
    for cluster_indices in clusters:
        cluster_points = point_cloud[cluster_indices]
        if len(cluster_indices) > 3:
            plane_normal, plane_point = fit_plane_svd(cluster_points)
            line_direction, line_point = fit_line_svd(cluster_points)

            # Evaluate fits
            plane_quality, plane_error = evaluate_fit_quality(
                cluster_points, "plane", plane_normal
            )
            line_quality, line_error = evaluate_fit_quality(cluster_points, "line")

            # Decide the best model based on explained variance and error
            if plane_quality > line_quality and plane_error < line_error:
                features.append(
                    {
                        "type": "plane",
                        "normal": plane_normal,
                        "point": plane_point,
                        "error": plane_error,
                    }
                )
            else:
                features.append(
                    {
                        "type": "line",
                        "direction": line_direction,
                        "point": line_point,
                        "error": line_error,
                    }
                )
        else:
            features.append({"type": "sparse", "points": cluster_points})

    return features


if __name__ == "__main__":
    # point_cloud = np.random.rand(100, 3)
    point_cloud = load_point_cloud()
    point_cloud = point_cloud[::117]
    point_cloud = normalize_point_cloud(point_cloud=point_cloud)
    normals, curvatures = compute_normals(point_cloud)
    print("normals:", normals[:100])
    print("curvatures:", curvatures[:100])
    plot_cloud_with_normals_plotly(point_cloud=point_cloud, normals=normals)
    clusters = region_growing(point_cloud, normals)
    print("clusters:", clusters[:100])
    features = extract_features(point_cloud, clusters)

    print("Extracted Features:")
    for feature in features:
        print(feature)
