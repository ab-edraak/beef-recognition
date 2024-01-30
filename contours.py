import cv2 as cv
import numpy as np


img = cv.imread("images/VBS-2000-02-eplusv-produktbilder-rind.jpg")

blank = np.zeros(img.shape, dtype="uint8")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow("blur", blur)
#
# canny = cv.Canny(blur, 125, 175)

# cv.imshow("canny edges", canny)
#

ret, thresh = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)

cv.imshow("thresh", thresh)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

print(len(contours))

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow("contours drawn", blank)

cv.waitKey(0)
