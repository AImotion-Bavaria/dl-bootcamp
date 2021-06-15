import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from pathlib import Path
import pandas

from tensorflow import keras
from tensorflow.keras import layers

#### Config ####
image_size = 45
batch_size = 64
num_epochs = 25
learning_rate = 0.001
num_classes = 43
#################

def fully_connected_net(input_dim=image_size*image_size, num_classes=num_classes):
    model = keras.Sequential([
        keras.Input(shape=(image_size, image_size, 3)),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="relu"),
    ])

    model.summary()
    return model


def small_conv_net(num_classes=num_classes):
    model = keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, 3)),
            layers.Conv2D(6, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),

            layers.Dense(120, activation="relu"),
            layers.Dense(84, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    return model

def pretrained_model(num_classes=num_classes):
    # from https://keras.io/guides/transfer_learning/

    base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(image_size, image_size, 3)
    )

    # base_model.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=True)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes)(x)

    model = keras.Model(inputs, outputs)
    return model

def convert_path_to_image(path):
    temp = Image.open(path)
    img = temp.copy()
    temp.close()
    # img = img.convert(mode='L') # convert to grayscale
    img = img.resize((image_size, image_size))
    return np.array(img)

def load_csv(csv_path, root_path):
    data_frame = pandas.read_csv(csv_path)

    label_ints = data_frame.pop('ClassId').to_numpy()
    labels = np.zeros((label_ints.size, label_ints.max()+1))
    labels[np.arange(label_ints.size), label_ints] = 1 

    # labels = np.expand_dims(labels, axis=1)
    img_paths = data_frame.pop('Path').to_numpy()
    images = np.array([convert_path_to_image(root_path / p) for p in img_paths])

    return images, labels


"""
Method from https://www.tensorflow.org/tutorials/load_data/images
"""

data_root_path = Path("C:/Arbeit/datasets/traffic_sign")
train_path = data_root_path / "Train.csv"
test_path = data_root_path / "Test.csv"




train_images, train_labels = load_csv(train_path, data_root_path)


opt = keras.optimizers.SGD(learning_rate=learning_rate)
# opt = keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# model = fully_connected_net()
model = small_conv_net()

# model = pretrained_model()

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size)

print("#######################")
print("Evaluating")
print("#######################")

test_images, test_labels = load_csv(test_path, data_root_path)
model.evaluate(test_images, test_labels)

print("Happy debug")
