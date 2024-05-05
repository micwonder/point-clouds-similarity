import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_point_cloud_plotly(position: str, similarity, cloud_data):
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
            xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Altitude"
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
