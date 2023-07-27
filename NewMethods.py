import numpy as np
import cv2
import os

class TrainingImage:
    def make_integral_image(h, w, ColorChannel, image):
        ChannelName = str(ColorChannel)
        for i in range(h):
            for j in range(w):
                f"integral_image_{ChannelName}"[i][j] = image[i][j]
        
        f"integral_image_{ChannelName}"[0][0] = image[0][0]
        #Calculating 1st row integral image
        for i in range(h-1):
            f"integral_image_{ChannelName}"[i+1][0] = image[i+1][0] + f"integral_image_{ChannelName}"[i][0]
        #Calculating 1st coloum integral image
        for i in range(w-1):
            f"integral_image_{ChannelName}"[0][i+1] = image[0][i+1] + f"integral_image_{ChannelName}"[0][i]
        #Calculating the rest of the integral image
        for i in range(h-1):
            for j in range(w-1):
                f"integral_image_{ChannelName}"[i+1][j+1] = image[i+1][j+1] + f"integral_image_{ChannelName}"[i][j+1] + f"integral_image_{ChannelName}"[i+1][j] - f"integral_image_{ChannelName}"[i][j]


    def __init__(self, image):
        #Reading height and width
        self.image = cv2.imread(image)
        self.h, self.w = self.image.shape[:2]
        blue, green, red = cv2.split(image)

        self.make_integral_image(self.h, self.w, blue, self.image)
        np.savetxt('integral_image_blue.txt', self.integral_image_blue, fmt='%d')

# test_image = TrainingImage('image.png')
# print("Height = {},  Width = {}".format(test_image.h, test_image.w))
# cv2.imwrite('ImageTest.png', test_image.image)
# print(test_image.integral_image)