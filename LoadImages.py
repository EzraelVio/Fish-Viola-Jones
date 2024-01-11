import cv2
import os
import numpy as np

def get_label(directory):
    # add more to add more class
    if directory == "fish_dataset\\abudefduf": return 1
    if directory == "fish_dataset\\amphiprion": return 2
    if directory == "fish_dataset\\chaetodon": return 3
    else: return 0

'''
def get_label(directory):
    # add more to add more class
    if directory == "fish_dataset\\1.jambu": return 1
    if directory == "fish_dataset\\2.sirih_gajah": return 2
    if directory == "fish_dataset\\3.jari5": return 3
    else: return 0
'''

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
    abudefduf_images, abudefduf_labels = load_images("fish_dataset\\abudefduf")
    amphiprion_images, amphiprion_labels = load_images("fish_dataset\\amphiprion")
    chaetodon_images, chaetodon_labels = load_images("fish_dataset\\chaetodon")
    negatives_images, negatives_labels = load_images("fish_dataset\\negative_examples")

    # combining into a single dataset
    images = np.concatenate((abudefduf_images, amphiprion_images, chaetodon_images, negatives_images), axis = 0)
    labels = np.concatenate((abudefduf_labels, amphiprion_labels, chaetodon_labels, negatives_labels), axis = 0)

    return images, labels

# def combine_dataset():
#     # load datasets from directories
#     # add class to get_label first or the class will be considered a negative example
#     jambu_images, jambu_labels = load_images("fish_dataset\\1.jambu")
#     sirih_gajah_images, sirih_gajah_labels = load_images("fish_dataset\\2.sirih_gajah")
#     jari5_images, jari5_labels = load_images("fish_dataset\\3.jari5")
#     negative_images, negative_labels = load_images("fish_dataset\\negative_examples2")

#     # combining into a single dataset
#     images = np.concatenate((jambu_images, sirih_gajah_images, jari5_images, negative_images), axis = 0)
#     labels = np.concatenate((jambu_labels, sirih_gajah_labels, jari5_labels, negative_labels), axis = 0)

#     return images, labels