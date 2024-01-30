import cv2
import numpy as np


def apply_gabor_filter(image, ksize=31, sigma=5, theta=0, lambd=10, gamma=0.5):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a Gabor filter
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)

    # Apply the Gabor filter to the image
    filtered_image = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)

    return filtered_image


# Example Usage:
image = cv2.imread("images/carcasses.jpg")

# Adjust the parameters as needed
filtered_texture = apply_gabor_filter(
    image, ksize=5, sigma=5, theta=0, lambd=10, gamma=0.5
)

# Display the original image and the filtered texture
cv2.imshow("Original Image", image)
cv2.imshow("Filtered Texture", filtered_texture)
cv2.waitKey(0)
cv2.destroyAllWindows()
