import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from pathlib import Path
import pandas

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#### Config ####
image_size = 128
batch_size = 64
num_epochs = 150
learning_rate = 0.0001
num_classes = 1
train_path = "C:/Arbeit/datasets/TL/train"
test_path = "C:/Arbeit/datasets/TL/test"
#################

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
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_path, image_size=(image_size, image_size))
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_path, image_size=(image_size, image_size))

# opt = keras.optimizers.SGD(learning_rate=learning_rate)
opt = keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

model = pretrained_model()

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)

print("#######################")
print("Evaluating")
print("#######################")

model.evaluate(test_dataset)

vis_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_path, image_size=(image_size, image_size), batch_size=1)
for i, l in vis_dataset:
    pred = model.predict(i)
    plt.imshow(tf.squeeze(i, axis=0).numpy().astype(np.uint8))
    plt.title(f"Label: {l[0]}; Prediction: {tf.math.round(pred)}")
    plt.show()
