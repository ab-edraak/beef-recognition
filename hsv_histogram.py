import cv2

# import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/canal(2).jpg")

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Separate the channels
h, s, v = cv2.split(hsv_image)

# Generate histograms
hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

# Plot histograms
plt.subplot(311)
plt.plot(hist_h, color="red")
plt.title("Hue Histogram")

plt.subplot(312)
plt.plot(hist_s, color="green")
plt.title("Saturation Histogram")

plt.subplot(313)
plt.plot(hist_v, color="blue")
plt.title("Value Histogram")

plt.show()
