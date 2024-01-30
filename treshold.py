import cv2 as cv
import numpy as np

img = cv.imread("images/VBS-2000-02-eplusv-produktbilder-rind.jpg")
blank = np.zeros(img.shape, dtype="uint8")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

adaptiveThresholded = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3
)

cv.imshow("adaptive threshold", adaptiveThresholded)

cv.waitKey(0)
