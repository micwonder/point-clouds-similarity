from computing_surface_normals import compute_normals, normalize_point_cloud
from load_point_cloud import load_target_point_cloud_by_directions
from myploty import (
    plot_cloud_with_normals,
    plot_point_cloud_plotly,
    plot_clusters_plotly,
)
from segmentation_and_clustering import region_growing

if __name__ == "__main__":
    point_cloud = load_target_point_cloud_by_directions("dataset_noise")
    point_cloud = normalize_point_cloud(point_cloud=point_cloud)

    # plot_point_cloud_plotly(cloud_data=point_cloud)

    normals = compute_normals(point_cloud)
    clusters = region_growing(point_cloud=point_cloud, normals=normals)
    print("Computed Normals:\n", normals)
    print(clusters)
    print(len(clusters))
    print(len(point_cloud))

    # plot_cloud_with_normals(point_cloud=point_cloud, normals=normals)
    # plot_cloud_with_normals(point_cloud=point_cloud, normals=normals)

    plot_clusters_plotly(point_cloud=point_cloud, clusters=clusters)
