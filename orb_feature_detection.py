import cv2
import numpy as np

# img1 = cv2.imread("images/carcass.jpg")
img1 = cv2.imread("images/single_carcass.jpg")
img2 = cv2.imread("images/carcasses.jpg")

img1 = cv2.GaussianBlur(img1, (5, 5), cv2.BORDER_DEFAULT)
img2 = cv2.GaussianBlur(img2, (5, 5), cv2.BORDER_DEFAULT)


orb = cv2.ORB_create(nfeatures=1000000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good.append([m])


img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
imgkp1 = cv2.drawKeypoints(img1, kp1, None)
imgkp2 = cv2.drawKeypoints(img2, kp2, None)


cv2.imshow("img1", imgkp1)
cv2.imshow("img2", imgkp2)
cv2.imshow("img3", img3)

cv2.waitKey(0)
