import os
import sys
from typing import List
import tensorflow as tf
import csv
import argparse
import csv
from PIL import Image
import numpy as np
from skimage import transform

#Path to the classifier model
model_path = "PATH TO THE MODEL"

class_names = [
    "boots",
    "flipflops",
    "loafers",
    "noshoe",
    "sandals",
    "sneakers",
    "soccershoes",
]

#Converting image into numpy array - Input ->Filename, Output-> numpy array
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype("float32") / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


#Loading saved model
model = tf.keras.models.load_model(model_path)

#classify_model
def classify_model(images:List[str])-> List[str]:
    result = []
    for image in images:
        image = load(image)
        predicted_batch = model.predict(image)
        predicted_id = np.argmax(predicted_batch, axis=-1)
        id = list(predicted_id)[0]
        result.append(class_names[id])
    return result




