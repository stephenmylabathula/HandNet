import cv2
import numpy as np

# Image Click Callback Function
def click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print "(" + str(x) + ", " + str(y) + ")"


# Load Hand Image
hand_image = cv2.imread("hand.jpg")
hand_image = cv2.resize(hand_image, (0, 0), fx=0.2, fy=0.2)
hand_image_gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)

# Ground Truth Hand Points
hand_points_ground_truth = [(484, 327), (531, 374), (576, 422)]

# Finger ROI Coordinates
finger_roi_coordinates = [(438, 268), (576, 422)]

# Ground Truth Hand Points Cropped
hand_points_ground_truth_cropped = [(hand_points_ground_truth[0][0] - finger_roi_coordinates[0][0], hand_points_ground_truth[0][1] - finger_roi_coordinates[0][1]),
                                    (hand_points_ground_truth[1][0] - finger_roi_coordinates[0][0], hand_points_ground_truth[1][1] - finger_roi_coordinates[0][1]),
                                    (hand_points_ground_truth[2][0] - finger_roi_coordinates[0][0], hand_points_ground_truth[2][1] - finger_roi_coordinates[0][1])]

# Crop Out Finger
cropped_finger = hand_image[finger_roi_coordinates[0][1]:finger_roi_coordinates[1][1],
                 finger_roi_coordinates[0][0]:finger_roi_coordinates[1][0]]
cropped_finger_gray = cv2.cvtColor(cropped_finger, cv2.COLOR_BGR2GRAY)

# Get Cropped Image Contours
ret, thresh = cv2.threshold(cropped_finger_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Perform Point Polygon Test to Determine Contour Surrounding Hand (if multiple, take last one)
for c in range(len(contours)):
    if cv2.pointPolygonTest(contours[c], hand_points_ground_truth_cropped[0], False) >= 0 or \
                    cv2.pointPolygonTest(contours[c], hand_points_ground_truth_cropped[1], False) >= 0 or \
                    cv2.pointPolygonTest(contours[c], hand_points_ground_truth_cropped[2], False) >= 0:
        print "Contour: " + str(c)
        contour_bounding_hand = contours[c]

# Show Image
cv2.namedWindow('image')
cv2.setMouseCallback('image', click_callback)
binary_segmented_image = np.zeros(cropped_finger_gray.shape)
cv2.fillPoly(binary_segmented_image, pts=[contour_bounding_hand], color=(255, 255, 255))

#cv2.drawContours(cropped_finger, [contour_bounding_hand], -1, (0, 255, 0), 3, cv2.)

cv2.imshow('image', cv2.drawContours(cropped_finger, [contour_bounding_hand], -1, (0, 255, 0), 3))
cv2.waitKey(0)

cv2.imshow('image', binary_segmented_image)
cv2.waitKey(0)