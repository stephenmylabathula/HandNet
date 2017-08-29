import os
import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt


dataset_path = '/usr/local/google/home/smylabathula/datasets/hand_net_color/5/'


hand_points = np.loadtxt(dataset_path + 'marker_pixels.txt', delimiter=',')


images = os.listdir(dataset_path + 'images')
images.sort()


bbox_height = 200
bbox_width = 200
image_height = 480
image_width = 640


finger_roi_points = np.array([[-1, 0, 0, 0, 0, 0, 0]], dtype=np.int16)


for i in range(len(images)):
    try:
        # Load Image
        img = cv2.imread(dataset_path + 'images/' + images[i])
        # Load Labeled Data
        finger_points = np.array([hand_points[i][1:3], hand_points[i][3:5], hand_points[i][5:7]])

        # Skip if Non Finger
        if 0 in finger_points:
            continue

        # Extact Finger Joint Positions
        finger_tip = np.argmin(finger_points[:, 0])
        finger_root = np.argmax(finger_points[:, 1])
        if finger_tip == finger_root:
            continue
        finger_mid = 3 - finger_root - finger_tip   # the index left out is the middle
        finger_tip = finger_points[finger_tip]
        finger_root = finger_points[finger_root]
        finger_mid = finger_points[finger_mid]

        # Create Bounding Box around Finger
        top_point = (int(np.min(finger_points[:, 0])), int(np.min(finger_points[:, 1])))
        bottom_point = (int(np.max(finger_points[:, 0])), int(np.max(finger_points[:, 1])))
        #   - Stretch Bounding Box to Fit Desired Shape
        width = bottom_point[0] - top_point[0]
        height = bottom_point[1] - top_point[1]
        #   - Throw out Sample if Finger to Big
        if width > bbox_width and height > bbox_height:
            continue
        width_lack_sides = int(math.ceil(float(bbox_width - width) / 2))
        height_lack_sides = int(math.ceil(float(bbox_height - height) / 2))
        #   - Adjust Bounding Box
        top_point = (top_point[0] - width_lack_sides, top_point[1] - height_lack_sides)
        bottom_point = (bottom_point[0] + width_lack_sides, bottom_point[1] + height_lack_sides)

        #   Throw out if Bounding Box Exceeds Image Dimensions
        if top_point[0] < 0 or top_point[1] < 0 or bottom_point[0] >= image_width or bottom_point[1] >= image_height:
            continue

        # Recalculate Finger Positions in Bounding Box
        finger_tip_bbox = (finger_tip[0] - top_point[0], finger_tip[1] - top_point[1])
        finger_mid_bbox = (finger_mid[0] - top_point[0], finger_mid[1] - top_point[1])
        finger_root_bbox = (finger_root[0] - top_point[0], finger_root[1] - top_point[1])

        # Create ROI from Bounding Box
        roi = img[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
        #   - Crop Bounding Box to Fit Desired Shape
        if roi.shape[0] > bbox_height:
            roi = roi[:(bbox_height - roi.shape[0]), :]
        if roi.shape[1] > bbox_width:
            roi = roi[:, :(bbox_width - roi.shape[1])]

        # Plot ROI with Joint Points
        cv2.circle(roi, (int(finger_tip_bbox[0]), int(finger_tip_bbox[1])), 5, (0, 0, 255), -1)  # Red Circle for Tip
        cv2.circle(roi, (int(finger_mid_bbox[0]), int(finger_mid_bbox[1])), 5, (0, 255, 0), -1)  # Green Circle for Mid
        cv2.circle(roi, (int(finger_root_bbox[0]), int(finger_root_bbox[1])), 5, (255, 0, 0), -1)  # Blue Circle for Root
        cv2.imshow("ROI", roi)
        cv2.waitKey(10)

        # Save ROI Points
        roi_points = np.array([finger_tip_bbox, finger_mid_bbox, finger_root_bbox], dtype=np.int16).reshape(6)
        roi_points = np.insert(roi_points, 0, i)
        finger_roi_points = np.append(finger_roi_points, [roi_points], axis=0)
        cv2.imwrite(dataset_path + 'roi/' + "%05d" % (i,) + '.png', roi)
    except IndexError:
        print "Out"

np.savetxt(dataset_path + 'roi_joint_points.txt', finger_roi_points[1:], delimiter=',')