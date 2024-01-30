import cv2
import numpy as np


def detect_beef_carcass(image):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for potential beef carcass colors
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([30, 255, 255])

    # Create a binary mask based on the color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result


# Read the image
img = cv2.imread("images/carcasses.jpg")

# Apply the beef carcass detection function
result_image = detect_beef_carcass(img)

# Display the result
cv2.imshow("Beef Carcass Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
