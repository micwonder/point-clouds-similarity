import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def plot_clusters_plotly(point_cloud, clusters):
    """
    Plot clusters in 3D using Plotly.
    :param point_cloud: Nx3 numpy array of points.
    :param clusters: List of clusters, each is a list of indices into the point cloud.
    """
    fig = go.Figure()

    for i, cluster in enumerate(clusters):
        # Select points of the current cluster
        cluster_points = point_cloud[cluster, :]

        # Add scatter plot for cluster points
        fig.add_trace(
            go.Scatter3d(
                x=cluster_points[:10, 0],
                y=cluster_points[:10, 1],
                z=cluster_points[:10, 2],
                mode="markers",
                marker=dict(
                    size=1.5,  # Adjust point size here
                    opacity=0.8,  # Adjust point opacity here
                ),
                name=f"Cluster {i+1}",  # Name each cluster trace for clarity
            )
        )

    # Set titles and labels
    fig.update_layout(
        title="3D Point Cloud Clusters",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude",
        ),
        legend_title="Clusters",
    )

    # Show the plot
    fig.show()


def plot_cloud_with_normals_plotly(point_cloud, normals, arrow_length=0.1):
    """
    Plot the point cloud with normals as arrows.
    :param point_cloud: Nx3 numpy array of points.
    :param normals: Nx3 numpy array of corresponding normals.
    :param arrow_length: Length of the arrow representing the normal.
    """
    fig = go.Figure()

    # Plotting the points
    fig.add_trace(
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=point_cloud[:, 2],  # Color by altitude
                colorscale="Viridis",  # Choose a colormap
                opacity=0.6,
            ),
            name="Point Cloud",
        )
    )

    # Plotting the normals as arrows
    for i in range(len(point_cloud)):
        point = point_cloud[i]
        normal = normals[i]
        arrow_end = point + arrow_length * normal
        fig.add_trace(
            go.Scatter3d(
                x=[point[0], arrow_end[0]],
                y=[point[1], arrow_end[1]],
                z=[point[2], arrow_end[2]],
                mode="lines",
                line=dict(color="red", width=2),
                hoverinfo="none",
                showlegend=False,
            )
        )

    fig.update_layout(
        # title=f'Position: {position.split(".txt")[0]} - Similarity: {similarity:.2f}',
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude",
            aspectmode="data",
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
        ),
        title="Point Cloud with Normals",
    )

    fig.show()


def plot_cloud_with_trace_plotly(point_cloud):
    fig = go.Figure()

    # Plotting the points
    fig.add_trace(
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode="markers",
            marker=dict(color="blue", size=5),
            name="Point Cloud",
        )
    )

    # Plotting the normals as arrows
    fig.add_trace(
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode="lines",
            line=dict(color="red", width=2),
            hoverinfo="none",
            showlegend=False,
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
        ),
        title="Point Cloud with Normals",
    )

    fig.show()


def plot_cloud_with_normals(point_cloud, normals):
    """
    Plot the point cloud with normals.
    :param point_cloud: Nx3 numpy array of points.
    :param normals: Nx3 numpy array of corresponding normals.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plotting the points
    ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        color="b",
        label="Point Cloud",
    )

    # Plotting the normals
    for i in range(len(point_cloud)):
        if not np.isnan(normals[i]).any():  # Check if normals were computed
            ax.quiver(
                point_cloud[i, 0],
                point_cloud[i, 1],
                point_cloud[i, 2],
                normals[i, 0],
                normals[i, 1],
                normals[i, 2],
                length=0.2,
                color="r",
                normalize=True,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Point Cloud with Normals")
    plt.show()


def plot_point_cloud_by_density_plotly(
    cloud_data, position: str = "NaN", similarity=1.0, density=2
):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=cloud_data[::density, 0],
                y=cloud_data[::density, 1],
                z=cloud_data[::density, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=cloud_data[::density, 2],  # Color by altitude
                    colorscale="Viridis",  # Choose a colormap
                    opacity=0.6,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f'Position: {position.split(".txt")[0]} - Similarity: {similarity:.2f}',
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude",
            # zaxis=dict(range=[0, 1000]),
            # yaxis=dict(range=[125.64, 125.7]),
        ),
    )

    fig.show()


def plot_point_cloud_plotly(cloud_data, position: str = "NaN", similarity=1.0):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=cloud_data[::2, 0],
                y=cloud_data[::2, 1],
                z=cloud_data[::2, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=cloud_data[::2, 2],  # Color by altitude
                    colorscale="Viridis",  # Choose a colormap
                    opacity=0.6,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f'Position: {position.split(".txt")[0]} - Similarity: {similarity:.2f}',
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude",
            # zaxis=dict(range=[0, 1000]),
            # yaxis=dict(range=[125.64, 125.7]),
        ),
    )

    fig.show()


def plot_point_cloud(position: str, similarity, cloud_data, figure_idx: int = 1):
    fig = plt.figure(figure_idx, figsize=(10, 8))
    plt.get_current_fig_manager().set_window_title(
        position.split(".txt")[0]
    )  # * Set the title of the window to position(longitude, latitude)
    fig.suptitle(t=str(similarity))

    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z coordinates from the point cloud data
    x = cloud_data[::2, 0]
    y = cloud_data[::2, 1]
    z = cloud_data[::2, 2]

    # Plot point cloud
    scatter = ax.scatter(x, y, z, c=z, marker="o", s=1, alpha=0.6, cmap="viridis")

    # Colorbar indicating altitude
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    colorbar.set_label("Altitude (m)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude")

    # Set grid and labels more visible
    ax.grid(True)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.zaxis.label.set_size(10)
    yield  # ! It is unnecessary if I don't use next()
