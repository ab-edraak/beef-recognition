import cv2
import numpy as np


def detect_beef_carcass_shape(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding or any other preprocessing as needed
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and analyze the shape
    for contour in contours:
        # Calculate the perimeter (arc length) of the contour
        perimeter = cv2.arcLength(contour, True)

        # Approximate the contour to a polygon
        epsilon = 0.02 * perimeter  # Adjust the epsilon value as needed
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get the number of vertices in the polygon
        num_vertices = len(approx)

        # Assuming beef carcass has a specific number of vertices
        # You may need to adjust this based on the characteristics of your beef carcasses
        if 10 <= num_vertices <= 80:
            print("Beef carcass detected!")

            # Draw the contour on the original image
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)  # Contour in green

    # Display the result
    cv2.imshow("Beef Carcass Shape Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example Usage:
detect_beef_carcass_shape("images/carcasses.jpg")
