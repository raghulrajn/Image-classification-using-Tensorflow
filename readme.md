# Shoe classification based on Tensorflow MobileNet

Ref: https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

## Requirements:

---

- Python
- Tensorflow

All dependencies can be installed using PIP

```Python
pip install -r requirements.txt
```

## Training

---

1. All the necessary training images were stored in the respective folders.
   - Train
     - Soccershoes
     - Sandals
     - Sneakers
     - Loafers
     - Flipflops
     - Boots
     - Noshoes
2. Model was trained using
   [Transfer Learning of Tensorflow Mobilenet V2 image classification](https://tfhub.dev/s?module-type=image-classification&q=tf2) model

## Image collection for Training

Images to train the model was downloaded using [Automated Python script](https://github.com/raghulrajn/Image-collection-for-Computer-vision-projects)

## Testing the model

Use `classify_model()` function in testing_model.py with input of list of images and returns the output of predicted lables.

> To get output as CSV files

`python -testing_model.py -p IMAGES FOLDER PATH -m CLASSFIER MODEL PATH -f CSV FILE NAME TO BE SAVED`
