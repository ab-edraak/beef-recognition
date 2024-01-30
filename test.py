import cv2
import numpy as np


def extract_color_range(image, lower_bound, upper_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    result = cv2.bitwise_and(image, image, mask=mask)

    return result


img = cv2.imread("images/VBS-2000-02-eplusv-produktbilder-rind.jpg")

fat_lower_bound = np.array([100, 50, 50])
fat_upper_bound = np.array([130, 255, 255])

meat_lower_bound = np.array([0, 50, 50])
meat_upper_bound = np.array([20, 255, 255])


additional_color_lower_bound = np.array([0, 30, 30])
additional_color_upper_bound = np.array([30, 255, 255])

fat_regions = extract_color_range(img, fat_lower_bound, fat_upper_bound)

meat_regions = extract_color_range(img, meat_lower_bound, meat_upper_bound)

additional_color_regions = extract_color_range(
    img, additional_color_lower_bound, additional_color_upper_bound
)

combined_regions = cv2.addWeighted(fat_regions, 1, additional_color_regions, 1, 0)
# combined_regions = cv2.addWeighted(combined_regions, 1, additional_color_regions, 1, 0)


cv2.imshow("Combined Regions", combined_regions)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
