import cv2
import numpy as np

# Read the image
img = cv2.imread("images/carcass2.jpg")  # Replace with your image path
img_min = np.zeros(img.shape, dtype="uint8")

height, width, _ = img.shape

for i in range(height):
    for j in range(width):
        # img[i, j] is the RGB pixel at position (i, j)
        # Check each element individually
        changed = False
        for k in range(3):
            if img[i, j, k] <= 0:
                img_min[i, j] = [255, 255, 255]
                changed = True
        if not changed:
            if np.all(img[i, j] == [255, 255, 255]):
                img[i, j] = [0, 0, 0]
            else:
                img_min[i, j] = img[i, j]


# Split the image into its BGR channels
b, g, r = cv2.split(img)

# Split the img_min into its BGR channels
b_min, g_min, r_min = cv2.split(img_min)

# Find the minimum and maximum values for each channel
max_vals = np.max([b.max(), g.max(), r.max()])
min_vals = np.min([b_min.min(), g_min.min(), r_min.min()])


cv2.imshow("img_min", img_min)
cv2.imshow("img", img)
cv2.waitKey(0)
# Display the color range
print("Color Range (BGR):")
print("Min Values:", [b_min.min(), g_min.min(), r_min.min()])
print("Max Values:", [b.max(), g.max(), r.max()])
print("Overall Min Value:", min_vals)
print("Overall Max Value:", max_vals)
