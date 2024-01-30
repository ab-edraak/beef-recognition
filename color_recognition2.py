import cv2
import numpy as np


def contour(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 25, 170, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image in red
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    return


def draw_bbs(image, boxes):
    # Create a copy of the input image
    image_with_boxes = image.copy()

    # Draw bounding boxes on the image in green
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(
            image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
        )  # Draw a green bounding box

    return image_with_boxes


def non_max_suppression(boxes, overlap_threshold=0.3):
    # Non-maximum suppression to filter out overlapping boxes
    if len(boxes) == 0:
        return []

    # Convert boxes to numpy array
    boxes = np.array(boxes)

    # Initialize a list to keep the selected boxes
    selected_boxes = []

    # Sort boxes based on the size (area)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    indices = np.argsort(area)

    while len(indices) > 0:
        # Get the index of the last (largest) box in the sorted list
        last = len(indices) - 1
        i = indices[last]
        selected_boxes.append(boxes[i])

        # Calculate overlap with the remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[indices[:last], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[indices[:last], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[indices[:last], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[indices[:last], 3])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indices[:last]]

        # Delete indices of the boxes with overlap greater than the threshold
        indices = np.delete(
            indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0]))
        )

    return selected_boxes


def getBoxes(image, contours, min_box_size=1000):
    # Extract bounding boxes from contours
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_size = w * h

        if box_size >= min_box_size:
            bounding_boxes.append((x, y, x + w, y + h))

    return bounding_boxes


def morph(mask):
    # Apply morphological operations to the binary mask
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def cca(mask):
    # Connected Component Analysis to filter unwanted components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Filter unwanted components based on area (you may adjust the area threshold)
    min_area_threshold = 1000
    filtered_mask = np.zeros_like(mask)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] > min_area_threshold:
            filtered_mask[labels == label] = 255

    # Fill in the connected areas
    contours, _ = cv2.findContours(
        filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.fillPoly(filtered_mask, contours, color=(255, 255, 255))

    return filtered_mask


def detect_objects_color(img):
    # Color-based object detection using HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the HSV values of the beef carcass
    lower_bound = np.array([2, 0, 185])
    upper_bound = np.array([25, 255, 255])
    lower_bound_red = np.array([170, 0, 0])
    upper_bound_red = np.array([180, 255, 255])

    # Create binary masks using the inRange function
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    mask_red = cv2.inRange(hsv_img, lower_bound_red, upper_bound_red)
    mask_combined = cv2.bitwise_or(mask, mask_red)
    mask = cca(mask_combined)

    # Apply the mask to the original image
    result_masked = cv2.bitwise_and(img, img, mask=mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = getBoxes(img, contours, 10000)

    return boxes


# Read the input image
image = cv2.imread("images/carcasses.jpg")

# Detect objects based on color
object_boxes = detect_objects_color(image)

# Draw bounding boxes on the image
result = draw_bbs(image, object_boxes)

# Display the result
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
