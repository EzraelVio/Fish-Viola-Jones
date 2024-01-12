import cv2

target_image_name = 'target.png'
image_unedited = cv2.imread(target_image_name, cv2.IMREAD_UNCHANGED)

position = (10, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (0, 0, 255)

cv2.putText(image_unedited, image_class, position, font, font_scale, font_color, font_thickness)
cv2.imwrite('hasil.jpg', image_unedited)
