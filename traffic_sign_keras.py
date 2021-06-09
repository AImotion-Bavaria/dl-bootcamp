import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from pathlib import Path
import pandas

"""
Method from https://www.tensorflow.org/tutorials/load_data/images
"""

data_root_path = Path("C:/Arbeit/datasets/traffic_sign")
train_path = data_root_path / "Train.csv"
test_path = data_root_path / "Test.csv"

def convert_path_to_image(path):
    temp = Image.open(path)
    img = temp.copy()
    temp.close()
    img = img.resize((43, 43))
    return np.array(img)



train_data_frame = pandas.read_csv(train_path)

train_labels = train_data_frame.pop('ClassId').to_numpy()
train_img_paths = train_data_frame.pop('Path').to_numpy()
train_images = np.array([convert_path_to_image(data_root_path / p) for p in train_img_paths])

print("Happy debug")
