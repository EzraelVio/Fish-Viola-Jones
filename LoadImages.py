import cv2
import os
import numpy as np

def get_label(directory):
    # add more to add more class
    if directory == "abudefduf": return 1
    if directory == "amphiprion": return 2
    if directory == "chaetodon": return 3
    else: return 0

def load_images(directory):
    images=[]
    labels=[]
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (700, 400))
            images.append(image)
            labels.append(get_label(directory))
    return np.array(images), np.array(labels)

def combine_dataset():
    # load datasets from directories
    # add class to get_label first or the class will be considered a negative example
    abudefduf_images, abudefduf_labels = load_images("fish_dataset\\abudefduf")
    amphiprion_images, amphiprion_labels = load_images("fish_dataset\\amphiprion")
    chaetodon_images, chaetodon_labels = load_images("fish_dataset\\chaetodon")
    negatives_images, negatives_labels = load_images("fish_dataset\\negatives_examples")

    # combining into a single dataset
    images = np.concatenate((abudefduf_images, amphiprion_images, chaetodon_images, negatives_images), axis = 0)
    labels = np.concatenate((abudefduf_labels, amphiprion_labels, chaetodon_labels, negatives_labels), axis = 0)
    # print("Shape of images array:", images.shape)
    # print("Shape of labels array:", labels.shape)

    return images, labels