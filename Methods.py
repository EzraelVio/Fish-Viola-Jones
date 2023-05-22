import numpy as np
import cv2
import os

class TrainingImage:
    def __init__(self, image):
        #Reading height and width
        self.image = cv2.imread(image)
        self.h, self.w = self.image.shape[:2]

        #Rescaling if image width != 72
        if (self.w > 72):
            self.ratio = 72/self.w
            self.dimension = (int(self.w * self.ratio), int(self.h * self.ratio))
            self.image = cv2.resize(self.image, self.dimension, cv2.INTER_LANCZOS4)
            self.h, self.w = self.image.shape[:2]

        #Grayscaling Image
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        #Creating Integral Image
        self.integral_image = np.empty((self.h, self.w))

        for i in range(self.h):
            for j in range(self.w):
                self.integral_image[i][j] = self.image[i][j]
        
        self.integral_image[0][0] = self.image[0][0]
        #Calculating 1st row integral image
        for i in range(self.h-1):
            self.integral_image[i+1][0] = self.image[i+1][0] + self.integral_image[i][0]
        #Calculating 1st coloum integral image
        for i in range(self.w-1):
            self.integral_image[0][i+1] = self.image[0][i+1] + self.integral_image[0][i]
        #Calculating the rest of the integral image
        for i in range(self.h-1):
            for j in range(self.w-1):
                self.integral_image[i+1][j+1] = self.image[i+1][j+1] + self.integral_image[i][j+1] + self.integral_image[i+1][j] - self.integral_image[i][j]

        np.savetxt('integral_image.txt', self.integral_image, fmt='%d')

test_image = TrainingImage('image.png')
print("Height = {},  Width = {}".format(test_image.h, test_image.w))
cv2.imwrite('ImageTest.png', test_image.image)
print(test_image.integral_image)