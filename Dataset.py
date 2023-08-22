import numpy as np

from HaarFeatures import *
from IntegralImage import *

class Dataset:
    class_Window_offset_1 = [
        # order according to label's order in LoadImages
        # for searching mouth feature
        (0, 0), 
        (0, 0),
        (0, 0),
        (0, 0)
    ]

    class_Window_offset_2 = [
        # order according to label's order in LoadImages
        # for searching fin feature
        (0, 0), 
        (0, 0),
        (0, 0),
        (0, 0)
    ]

    class_Window_offset_3 = [
        # order according to label's order in LoadImages
        # for searching tail feature
        (0, 0), 
        (0, 0),
        (0, 0),
        (0, 0)
    ]

    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.blue_Channel, self.green_Channel, self.red_Channel = combine_integral_grb(image)