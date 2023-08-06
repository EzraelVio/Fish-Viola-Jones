import numpy as np
import cv2
import os
import json

class TrainingImage:
    def make_integral_image(self, h, w, ColorChannel, integral_image):
        for i in range(h):
            for j in range(w):
                integral_image[i][j] = ColorChannel[i][j]
        
        integral_image[0][0] = ColorChannel[0][0]
        #Calculating 1st row integral image
        for i in range(h-1):
            integral_image[i+1][0] = ColorChannel[i+1][0] + integral_image[i][0]
        #Calculating 1st coloum integral image
        for i in range(w-1):
            integral_image[0][i+1] = ColorChannel[0][i+1] + integral_image[0][i]
        #Calculating the rest of the integral image
        for i in range(h-1):
            for j in range(w-1):
                integral_image[i+1][j+1] = ColorChannel[i+1][j+1] + integral_image[i][j+1] + integral_image[i+1][j] - integral_image[i][j]


    def __init__(self, image):
        #Reading height and width
        self.image = cv2.imread(image)
        self.h, self.w = self.image.shape[:2]
        self.blue, self.green, self.red = cv2.split(self.image)
        self.integral_image_blue = np.empty((self.h, self.w))
        self.integral_image_green = np.empty((self.h, self.w))
        self.integral_image_red = np.empty((self.h, self.w))

        self.make_integral_image(self.h, self.w, self.blue, self.integral_image_blue)
        self.make_integral_image(self.h, self.w, self.green, self.integral_image_green)
        self.make_integral_image(self.h, self.w, self.red, self.integral_image_red)
        np.savetxt('integral_image_blue.txt', self.integral_image_blue, fmt='%d')
        np.savetxt('integral_image_green.txt', self.integral_image_green, fmt='%d')
        np.savetxt('integral_image_red.txt', self.integral_image_red, fmt='%d')

class Feature:
    def __init__(self, x, y):
        self.featureType = "2-square-left"
        self.integralCoordinate = [[x, y], [x + 3, y + 7], [x + 4, y], [x + 7, y + 7]]

def GenerateFeatures():
    print("Starting features generation")
    feature_list = []
    w = 0
    h = 0
    while h + 7 < 399:
        while w + 7 < 699:
            feature_object = Feature(h, w)
            feature_list.append(feature_object)
            w+=1
        h+=1
    while len(feature_list) > 0:
        sacrifice_object = feature_list.pop(0)
        with open(output_file, "Features") as json_file:
            json.dumps(sacrifice_object.__dict__, json_file)
    print("Feature generation finished")


        

            
            


test_image = TrainingImage('image.png')
# print("Height = {},  Width = {}".format(test_image.h, test_image.w))
# cv2.imwrite('ImageTestgreen.png', test_image.green)
# print(test_image.integral_image)