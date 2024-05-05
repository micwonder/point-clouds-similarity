import numpy as np
from scipy.spatial import KDTree
import os
import time
from tqdm import tqdm
from myploty import plot_point_cloud_plotly
from load_point_cloud import (
    load_target_point_cloud_by_directions,
    load_point_cloud_separate,
    load_point_cloud,
)

K = 1


def calculate_similarity(cloud1, cloud2, threshold=np.inf):
    """Calculate similarity between two point clouds using average nearest neighbor distance."""
    timestamp = time.time()
    tree1 = KDTree(cloud1)
    tree2 = KDTree(cloud2)

    # Distances from cloud1 to cloud2
    distances, _ = tree2.query(cloud1, k=K)
    print(len(distances))

    valid_distances1 = distances[distances < threshold]
    average_distance1 = (
        np.mean(valid_distances1) if len(valid_distances1) > 0 else np.inf
    )

    # Distances from cloud2 to cloud1
    distances, _ = tree1.query(cloud2, k=K)
    valid_distances2 = distances[distances < threshold]
    average_distance2 = (
        np.mean(valid_distances2) if len(valid_distances2) > 0 else np.inf
    )
    print("{:.3f} sec".format(time.time() - timestamp))

    # Average the two average distances
    return (average_distance1 + average_distance2) / 2


def test_tdtree():
    """Test TDTree similarity calculation"""
    # Load your point clouds (replace 'path_to_cloud1.txt' and 'path_to_cloud2.txt' with your file paths)
    cloud1 = load_point_cloud(r"dataset\point_cloud1.txt")
    cloud2 = load_point_cloud(r"dataset\point_cloud2.txt")

    # Calculate similarity
    threshold = 50
    while threshold > 49:
        similarity = calculate_similarity(
            cloud1, cloud2, threshold=threshold
        )  # Threshold set to 50 meters
        print(f"Threshold: {threshold}, Similarity measure: {similarity}")
        threshold -= 1


def data_preload(target_dir: str):
    """Dataset preload function"""
    if not os.path.isdir(target_dir):
        print("Error: target dir not exist.")
        return [], []

    timestamp = time.time()
    # ? Cloud data
    cloud_curview = load_point_cloud(r"dataset\point_cloud1.txt")
    # cloud_curview = load_target_point_cloud_by_directions("reconstruction_result")
    print(cloud_curview)
    map_data = []

    # * Load your point clouds in the target directory
    cloud_map_filenames = os.listdir(target_dir)
    for file_name in tqdm(cloud_map_filenames, desc="Loading data", unit="files"):
        file_path = os.path.join(target_dir, file_name)
        if not os.path.isfile(file_path):
            print("Error: {file_name} not available.")
            continue

        # ! Separate file structure is a bit different from the merged one
        map_data.append(
            {
                "pos": file_name,
                "data": load_point_cloud_separate(file_path=file_path),
            }
        )
    print(f"Success: dataset loaded successfully in {time.time() - timestamp} sec")

    return cloud_curview, map_data


def run(target_dir: str, threshold: int, recommend_cnt: int = 4):
    """Main Process"""
    # ? Preload
    cloud_curview, map_data = data_preload(target_dir=target_dir)

    similarities_with_index = []

    for idx, cloud_map_data in enumerate(map_data):
        position, cloud_map = cloud_map_data["pos"], cloud_map_data["data"]

        # Calculate similarity
        similarity = calculate_similarity(
            cloud1=cloud_map, cloud2=cloud_curview, threshold=threshold
        )
        similarities_with_index.append((idx, similarity))
        print(f"Threshold: {threshold}, Similarity measure: {similarity}")

    # * Sort by similarity ascend and then print first 4 pieces of the map data
    similarities_with_index.sort(key=lambda x: x[1], reverse=False)
    for idx, similarity in similarities_with_index[:recommend_cnt]:
        print((idx, similarity))
        plot_gen = plot_point_cloud_plotly(
            position=map_data[idx]["pos"],
            similarity=similarity,
            cloud_data=map_data[idx]["data"],
        )
        # #! Unnecessary to use generator
        # next(plot_gen)  # Start each generator to setup the plot

    # # * After all plots are set up, display them
    # plt.show()


if __name__ == "__main__":
    PATH = r"dataset\airport-data\test-txt"

    timestamp = time.time()
    run(target_dir=PATH, threshold=50, recommend_cnt=4)
    print("Execution succeed: ", time.time() - timestamp)
