# data_processing.py
#This file contains functions for loading and preprocessing the dataset.

import numpy as np
import cv2
from os import listdir
from sklearn.utils import shuffle

def scale_and_normalize(arr):
    """
    Perform Positive Global Standardization on input array and return it.
    Arguments:
        arr: 2-dimensional image array containing int or float values
    Returns:
        arr: positive globally standardized arr of float values
    """
    arr = arr.astype('float32')
    mean, stand_dev = arr.mean(), arr.std()
    arr = (arr - mean) / stand_dev
    arr = np.clip(arr, -1, 1)
    arr = (arr + 1) / 2
    return arr

def load_data(dir_list, image_size):
    """
    Read images, resize and normalize them.
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size

    for directory in dir_list:
        for filename in listdir(directory):
            # load the image
            image = cv2.imread(directory + '/' + filename)
            # crop the brain and ignore the unnecessary rest part of the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = scale_and_normalize(image)
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory.endswith('yes'):
                y.append(1)
            else:
                y.append(0)

    X = np.array(X)
    y = np.array(y)

    # Shuffle the data
    X, y = shuffle(X, y)
    return X, y
