import numpy as np
import cv2 as cv

img = cv.imread("images/VBS-2000-03-eplusv-produktbilder-rind.jpg", 0)
template = cv.imread("images/single_carcass.jpg", 0)
img2 = img.copy()
h, w = template.shape

# Choose a method
method = cv.TM_CCOEFF_NORMED

# Perform template matching
result = cv.matchTemplate(img2, template, method)

# Set a threshold
threshold = 0.8

# Find locations where the match score is above the threshold
loc = np.where(np.greater_equal(result, threshold))

# Draw rectangles on the detected objects
for pt in zip(*loc[::-1]):
    cv.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

# Show the result
cv.imshow("result", img2)
cv.waitKey(0)
