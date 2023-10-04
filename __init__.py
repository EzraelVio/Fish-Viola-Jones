import numpy as np
from LoadImages import *
from HaarFeatures import *
from IntegralImage import *
from Dataset import *
import time

start_time = time.time()

images, labels = combine_dataset()

# for testing only
features = generate_features(50, 50)
print(len(features))

# testing matrix calculation
# features = (235, 576, 50, 50)
# feature_value_matrice = compute_feature_with_matrix(images[0], 0, "Two Horizontal", features[94702])

# testing integral image calculation
# b, g, r = combine_integral_grb(images[0])
# feature_value_integral = compute_feature_value(b, "Two Horizontal", features[94702])

print("starting...")
new_Data = Dataset(images[0], labels[0], features)
print(new_Data.window_1_features[0])

toprint = images[0]
np.savetxt('Feature_visualisation.txt', toprint[185:185+50, 61:61+50, 0], fmt='%d')
# print(feature_value_matrice)
# print(feature_value_integral)
cv2.imwrite('ImageTest.png', images[0])

end_time = time.time()

print("Execution time:", (end_time - start_time), "s")