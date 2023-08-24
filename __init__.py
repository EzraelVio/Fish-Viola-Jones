import numpy as np
from LoadImages import *
from HaarFeatures import *
from IntegralImage import *

images, labels = combine_dataset()

# for testing only
features = generate_features(50, 50)

# testing matrix calculation
# features = (235, 576, 50, 50)
feature_value_matrice = compute_feature_with_matrix(images[0], 0, "Two Horizontal", features[94702])

# testing integral image calculation
# b, g, r = combine_integral_grb(images[0])
# feature_value_integral = compute_feature_value(b, "Two Horizontal", features[94702])

# toprint = images[0]
# np.savetxt('Feature_visualisation.txt', toprint[235:235+50, 576:576+50, 0], fmt='%d')
# print(feature_value_matrice)
# print(feature_value_integral)
# cv2.imwrite('ImageTest.png', images[0])

# 2147483647
# 4294897892
# 4294897940
