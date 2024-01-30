import cv2 as cv
import numpy as np


img = cv.imread("images/carcass2.jpg")
# blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


ret, thresh = cv.threshold(gray, 25, 170, cv.THRESH_BINARY)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(img, contours, -1, (0, 0, 255), 2)

cv.imshow("contoured img", img)
contour_image = np.zeros_like(img)

# Draw contours on the blank image
# cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display the result
# cv.imshow("Contour Image", contour_image)
cv.waitKey(0)
