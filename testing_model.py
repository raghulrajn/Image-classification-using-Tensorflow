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


# Sample input:
# python test_model.py -p <IMAGES FOLDER PATH> -m <CLASSFIER MODEL PATH> -f <CSV FILE NAME TO BE SAVED>


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype("float32") / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


class_names = [
    "boots",
    "flipflops",
    "loafers",
    "noshoe",
    "sandals",
    "sneakers",
    "soccershoes",
]

parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument(
    "-p", "--path", help="<DIRECTORY WITH FOLDERS OF IMAGES>", required=True
)
parser.add_argument("-m", "--model", help="<MODEL PATH>", required=True)
parser.add_argument("-f", "--file", help="<FILE_NAME>", required=True)
args = vars(parser.parse_args())

input_path = args["path"]
model_path = args["model"]
file_name = args["file"]

# classify_model
def classify_model(images: List[str]) -> List[str]:
    result = []
    for image in images:
        image = load(image)
        predicted_batch = model.predict(image)
        predicted_id = np.argmax(predicted_batch, axis=-1)
        id = list(predicted_id)[0]
        result.append(class_names[id])
    return result


# Loads model and iterate through each folder to make predictons and saves the CSV file.
if input_path and model_path and file_name:
    model = tf.keras.models.load_model(model_path)
    result = []
    csv_file = open(file_name + ".csv", "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter=",")
    for root, dirs, files in os.walk(input_path):
        for dir in dirs:
            for r, d, file in os.walk(os.path.join(root, dir)):
                for f in file:
                    image = load(dir + "/" + f)
                    predicted_batch = model.predict(image)
                    predicted_id = np.argmax(predicted_batch, axis=-1)
                    id = list(predicted_id)[0]
                    predicted_label_batch = class_names[id]
                    csv_writer.writerow([f, predicted_label_batch])
csv_file.close()


classify_model("<list of images>")

