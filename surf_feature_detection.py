import cv2
import numpy as np

img1 = cv2.imread("images/single_carcass.jpg")
img2 = cv2.imread("images/carcasses.jpg")

surf = cv2.SIFT_create()

kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Adjust the distance threshold as needed
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
imgkp1 = cv2.drawKeypoints(img1, kp1, None)
imgkp2 = cv2.drawKeypoints(img2, kp2, None)

cv2.imshow("img1", imgkp1)
cv2.imshow("img2", imgkp2)
cv2.imshow("img3", img3)

cv2.waitKey(0)
