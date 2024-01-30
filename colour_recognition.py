import cv2
import numpy as np


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

        if box_size >= min_box_size and x / y > 9 / 16:
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


img = cv2.imread("images/carcasses.jpg")
# img = cv2.imread("images/VBS-2000-03-eplusv-produktbilder-rind.jpg")
# img = cv2.imread("images/canal(2).jpg")

# Convert the image to HSV color space
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

boxes = getBoxes(img, contours, 10000)
# boxes = non_max_suppression(boxes, 0.0)
result = draw_bbs(img, boxes)

# Display the results
# cv2.imshow("HSV Color Thresholding Mask", mask_combined)
cv2.imshow("Result", result_masked)
# contour(result_masked)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
cv2.imshow("Original img", img)
cv2.imshow("result", result)
# cv2.imshow("Result", result_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
