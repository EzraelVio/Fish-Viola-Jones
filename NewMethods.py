import numpy as np
import cv2
import os

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

test_image = TrainingImage('image.png')
# print("Height = {},  Width = {}".format(test_image.h, test_image.w))
# cv2.imwrite('ImageTestgreen.png', test_image.green)
# print(test_image.integral_image)