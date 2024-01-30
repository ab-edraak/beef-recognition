import cv2 as cv
import numpy as np


img = cv.imread("images/VBS-2000-02-eplusv-produktbilder-rind.jpg")

blank = np.zeros(img.shape[:2], dtype="uint8")

b, g, r = cv.split(img)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # cv.imshow("red", r)
# ret, thresh = cv.threshold(r, 235, 255, cv.THRESH_BINARY)
#
# thresh_inv = cv.bitwise_not(thresh)
# mask = cv.cvtColor(thresh_inv, cv.COLOR_GRAY2BGR)
# result_image = cv.addWeighted(img, 1, mask, 1, 0)

# cv.imshow("thresh", thresh)
cv.imshow("original", img)
cv.imshow("bleu", blue)
cv.imshow("green", green)
cv.imshow("red", red)
# cv.imshow("result mask ", result_image)

cv.waitKey(0)
