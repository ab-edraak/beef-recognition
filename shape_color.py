import cv2
import numpy as np


def detect_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > 10000 and is_rectangular(cnt)
    ]

    return filtered_contours


def is_rectangular(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    return aspect_ratio > 9 / 16


def canny_edge_detection(image, threshold1=50, threshold2=150):
    # Step 2: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Apply Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1, threshold2)
    return edges


def contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 25, 170, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    return


def draw_bbs(image, boxes):
    image_with_boxes = image.copy()
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(
            image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
        )  # Draw a green bounding box

    return image_with_boxes


def non_max_suppression(boxes, overlap_threshold=0.3):
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
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_size = w * h

        if box_size >= min_box_size:
            bounding_boxes.append((x, y, x + w, y + h))

    return bounding_boxes


def morph(mask):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def cca(mask):
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
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bboundounds for the HSV values of the beef carcass
    lower_bound = np.array([2, 0, 185])
    upper_bound = np.array([25, 255, 255])
    lower_bound_red = np.array([170, 0, 0])
    upper_bound_red = np.array([180, 255, 255])
    # Create a binary mask using the inRange function
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    mask_red = cv2.inRange(hsv_img, lower_bound_red, upper_bound_red)
    mask_combined = cv2.bitwise_or(mask, mask_red)
    mask = cca(mask_combined)
    # mask = mask_combined
    # Apply the mask to the original img
    result_masked = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_boxes(img, contours):
    boxes = getBoxes(img, contours, 5000)
    # boxes = non_max_suppression(boxes, 0.0)
    boxes = [box for box in enumerate(boxes) if box[0] / box[1] > 9 / 13]
    return boxes


def calculate_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def find_intersecting_objects(contours_method1, contours_method2, iou_threshold):
    intersecting_objects = []

    for cnt1 in contours_method1:
        for cnt2 in contours_method2:
            iou = calculate_iou(cnt1, cnt2)
            if iou > iou_threshold:
                intersecting_objects.append(cnt1)

    return intersecting_objects


# Load the image
img = cv2.imread("images/carcasses.jpg")

# Step 1: Detect objects based on size and shape
shape_contours = detect_objects(img)

# Step 2: Filter objects based on color
color_contours = detect_objects_color(img)

# Step 3: Find intersecting objects
intersecting_objects = find_intersecting_objects(
    color_contours, shape_contours, iou_threshold=0.5
)

# Visualize the results
result_img = img.copy()

# Draw contours from the first method in green
# cv2.drawContours(result_img, shape_contours, -1, (0, 255, 0), 2)

# Draw contours from the second method in red
# cv2.drawContours(result_img, color_contours, -1, (0, 0, 255), 2)

# Draw intersecting contours in blue
boxes = get_boxes(img, intersecting_objects)
result = draw_bbs(img, boxes)
# Display the results
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
