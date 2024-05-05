import numpy as np
import os
from load_point_cloud import load_point_cloud_separate

point_cloud = load_point_cloud_separate(
    r"dataset\airport-data\test-txt\39°14'31.3794N_125°40'25.6466E_39°14'14.6753N_125°40'48.17E.txt"
)

os.makedirs("sample_output")

np.save(r"sample_output\x.npy", point_cloud[:, 0])
np.save(r"sample_output\y.npy", point_cloud[:, 1])
np.save(r"sample_output\z.npy", point_cloud[:, 2])

print("Saved successfully!")
