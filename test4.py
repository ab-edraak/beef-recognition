import cv2
import numpy as np


# def detect_objects(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     (thresh_contours, _) = cv2.findContours(
#         edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#
#     # Filter contours based on area and aspect ratio
#     # filtered_contours = [
#     #     cnt for cnt in contours if cv2.contourArea(cnt) > 10000 and is_rectangular(cnt)
#     # ]
#     filtered_contours = thresh_contours
#     return filtered_contours
#
def detect_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Connected Component Analysis to identify and label connected components
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )

    # Filter components based on area to keep only larger objects
    min_area_threshold = 10000
    filtered_components = [
        i for i, stat in enumerate(stats) if stat[4] > min_area_threshold
    ]

    # Extract contours corresponding to the filtered components
    filtered_contours = [
        get_contour_from_component(labels, i) for i in filtered_components
    ]

    return filtered_contours


def get_contour_from_component(labels, component_index):
    component_mask = (labels == component_index).astype(np.uint8)
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours[0] if contours else None


def is_rectangular(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    return aspect_ratio > 0


img = cv2.imread("images/carcasses.jpg")
contours = detect_objects(img)
cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
cv2.imshow("img", img)

cv2.waitKey(0)
