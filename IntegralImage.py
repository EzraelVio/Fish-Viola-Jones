import numpy as np
import cv2


class createIntegralImage:
    # run LoadImage.load_image before running this
    def make_integral_image(self, h, w, ColorChannel):

        integral_image = (np.empty((self.h, self.w)))
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
        return integral_image


    def __init__(self, image):
        #Reading height and width
        self.h, self.w = image.shape[:2]
        self.blue, self.green, self.red = cv2.split(self.image)
        self.integral_image = (np.empty((self.h, self.w)))
        self.integral_image_BGR = []
        # self.integral_image_blue = np.empty((self.h, self.w))
        # self.integral_image_green = np.empty((self.h, self.w))
        # self.integral_image_red = np.empty((self.h, self.w))

        # catetan, ini ntar pake for each aja
        self.integral_image_BGR.append(self.make_integral_image(self.h, self.w, self.blue))
        self.integral_image_BGR.append(self.make_integral_image(self.h, self.w, self.green))
        self.integral_image_BGR.append(self.make_integral_image(self.h, self.w, self.red))