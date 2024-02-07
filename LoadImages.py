import cv2
import os
import numpy as np

'''
def get_label(directory):
    # add more to add more class
    if directory == "fish_dataset\\abudefduf": return 1
    if directory == "fish_dataset\\amphiprion": return 2
    if directory == "fish_dataset\\chaetodon": return 3
    else: return 0

'''
def get_label(directory):
    # add more to add more class
    if directory == "fish_dataset\\emas": return 1
    if directory == "fish_dataset\\lele": return 2
    if directory == "fish_dataset\\nila": return 3
    else: return 0


def load_images(directory):
    images=[]
    labels=[]
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # yg dikomen ini buat daun 700x524. Ikan pake 350x200
            image = cv2.resize(image, (350, 200))
            # image = cv2.resize(image, (700, 524))
            images.append(image)
            labels.append(get_label(directory))
    return np.array(images), np.array(labels)

def combine_dataset():
    # load datasets from directories
    # add class to get_label first or the class will be considered a negative example
    emas_images, emas_labels = load_images("fish_dataset\\emas")
    lele_images, lele_labels = load_images("fish_dataset\\lele")
    nila_images, nila_labels = load_images("fish_dataset\\nila")
    negatives_images, negatives_labels = load_images("fish_dataset\\negative_examples")

    # combining into a single dataset
    images = np.concatenate((emas_images, lele_images, nila_images, negatives_images), axis = 0)
    labels = np.concatenate((emas_labels, lele_labels, nila_labels, negatives_labels), axis = 0)

    # images = np.concatenate((lele_images, nila_images, negatives_images), axis = 0)
    # labels = np.concatenate((lele_labels, nila_labels, negatives_labels), axis = 0)

    return images, labels
