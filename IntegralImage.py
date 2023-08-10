import numpy as np
import cv2



    # run LoadImage.load_image before running this
def make_integral_image(h, w, ColorChannel):

    integral_image = (np.empty((h, w)))
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


def combine_integral_grb(image):
    #Reading height and width
    h, w = image.shape[:2]
    blue, green, red = cv2.split(image)
    # integral_image = (np.empty((h, w)))
    integral_image_BGR = []
    # integral_image_blue = np.empty((h, w))
    # integral_image_green = np.empty((h, w))
    # integral_image_red = np.empty((h, w))

    # catetan, ini ntar pake for each aja
    integral_image_BGR.append(make_integral_image(h, w, blue))
    integral_image_BGR.append(make_integral_image(h, w, green))
    integral_image_BGR.append(make_integral_image(h, w, red))
    return integral_image_BGR
        