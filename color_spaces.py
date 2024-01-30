import cv2 as cv


img = cv.imread("images/VBS-2000-02-eplusv-produktbilder-rind.jpg")


spaces = [
    cv.COLOR_BGR2Lab,
    cv.COLOR_BGR2Luv,
    cv.COLOR_BGR2HSV,
    cv.COLOR_BGR2HSV,
    cv.COLOR_BGR2GRAY,
]

for space in spaces:
    changed = cv.cvtColor(img, space)
    cv.imshow("changed", changed)

    cv.waitKey(0)
