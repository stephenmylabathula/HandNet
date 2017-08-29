import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import contour_segmentation
import region_growing_segmentation

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


# Load Images
def generate_rgb_and_binary_images(num_images_to_load):
    rgb_image_list = os.listdir('/usr/local/google/home/smylabathula/datasets/nyu_hand_data/RGB')
    binary_image_list = os.listdir('/usr/local/google/home/smylabathula/datasets/nyu_hand_data/BINARY')
    rgb_images = np.zeros((num_images_to_load, 240, 320, 3), dtype=np.float32)
    binary_images = np.zeros((num_images_to_load, 240, 320), dtype=np.float32)
    for i in range(num_images_to_load):
        rgb_images[i] = plt.imread('/usr/local/google/home/smylabathula/datasets/nyu_hand_data/RGB/' +
                                  rgb_image_list[i]).astype(np.float32) / 255
        binary_images[i] = plt.imread('/usr/local/google/home/smylabathula/datasets/nyu_hand_data/BINARY/' +
                                     binary_image_list[i]).astype(np.float32) / 255
        if i % 100 == 0:
            print "Loaded " + str(i) + " images..."
    return rgb_images, binary_images


dataset_path = '/usr/local/google/home/smylabathula/datasets/hand_net_color/5/'
bounding_box_list = np.genfromtxt(dataset_path + 'finger_bounding_box.txt', delimiter=',', dtype=np.int16)
markers_list = np.genfromtxt(dataset_path + 'marker_pixels.txt', delimiter=',', dtype=np.int16)
image_list = os.listdir(dataset_path + 'roi_binary')
image_list.sort()
for i in range(len(image_list)):
    image = cv2.imread(dataset_path + 'roi_binary/' + image_list[i])
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.6)
    cv2.imwrite('/usr/local/google/home/smylabathula/datasets/hand_net_color/5/bounding_box/' + image_list[i][:image_list[i].index('.')] + '.png', image)
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    bounding_box_label_index = np.argwhere(bounding_box_list[:, 0] == i)
    markers_label_index = np.argwhere(markers_list[:, 0] == i)
    if len(bounding_box_label_index) < 1:
        cv2.imwrite('/usr/local/google/home/smylabathula/datasets/hand_net_color/5/roi_binary/' + str(i) + '.png',
                    cv2.inRange(image, 0, -1))
        continue
    image_roi = bounding_box_list[bounding_box_label_index.reshape(1)[0]][1:]
    if image_roi[2] + image_roi[3] == 0:
        cv2.imwrite('/usr/local/google/home/smylabathula/datasets/hand_net_color/5/roi_binary/' + str(i) + '.png',
                    cv2.inRange(image, 0, -1))
        continue
    #marker_points = markers_list[markers_label_index.reshape(1)[0]][1:]
    #marker_points = [(marker_points[0], marker_points[1]), (marker_points[2], marker_points[3]), (marker_points[4], marker_points[5])]

    black = cv2.inRange(image, 0, -1)
    cropped_image = cv2.inRange(image[image_roi[1]:image_roi[1] + image_roi[2], image_roi[0]:image_roi[0] + image_roi[3]], (0,0,0), (255,255,255))
    #marker_points_for_roi = [(marker_points[0][0] - image_roi[0], marker_points[0][1] - image_roi[1]),
     #                        (marker_points[1][0] - image_roi[0], marker_points[1][1] - image_roi[1]),
      #                       (marker_points[2][0] - image_roi[0], marker_points[2][1] - image_roi[1])]
    #mask = cv2.inRange(cropped_image, (0, 133, 77), (255, 173, 127))
    black[image_roi[1]:image_roi[1] + image_roi[2], image_roi[0]:image_roi[0] + image_roi[3]] = cropped_image
    #cv2.imshow("Image", black)
    #cv2.waitKey(10)
    cv2.imwrite('/usr/local/google/home/smylabathula/datasets/hand_net_color/5/roi_binary/' + str(i) + '.png', black)
    '''

    # To Draw Rectangle On Image
    '''
    cv2.rectangle(image, (image_roi[0], image_roi[1]), (image_roi[0] + image_roi[3], image_roi[1] + image_roi[2]), (0,255,0), 3)
    cv2.imshow('image', image)
    cv2.waitKey(5)
    '''
    '''
    # To Draw ROI
    roi = image[image_roi[1]:image_roi[1] + image_roi[2], image_roi[0]:image_roi[0] + image_roi[3]]
    cv2.imshow("Im", roi)
    cv2.waitKey(1)
    '''
    '''
    # Generate ROI Crop and Corresponding Marker Points
    cropped_image = image[image_roi[1]:image_roi[1] + image_roi[2], image_roi[0]:image_roi[0] + image_roi[3]]
    marker_points_for_roi = [(marker_points[0][0] - image_roi[0], marker_points[0][1] - image_roi[1]),
                             (marker_points[1][0] - image_roi[0], marker_points[1][1] - image_roi[1]),
                             (marker_points[2][0] - image_roi[0], marker_points[2][1] - image_roi[1])]
    black[image_roi[1]:image_roi[1] + image_roi[2], image_roi[0]:image_roi[0] + image_roi[3]] = cropped_image
    cv2.imshow('image', black)
    cv2.waitKey(10)
    '''
    '''
    if contour_segmentation.contour_segment_image_point_in_polygon_grayscale(cropped_image, marker_points_for_roi):
        black[image_roi[1]:image_roi[1] + image_roi[2], image_roi[0]:image_roi[0] + image_roi[3]] = cropped_image
        cv2.imshow('image', black)
        cv2.waitKey(10)
    '''

    '''
    if marker_points_for_roi[0][0] > 0 or marker_points_for_roi[0][1] > 0:
        region_growing_segmentation.simple_region_growing(cropped_image, marker_points_for_roi[0], 20)
    elif marker_points_for_roi[1][0] > 0 or marker_points_for_roi[1][1] > 0:
        region_growing_segmentation.simple_region_growing(cropped_image, marker_points_for_roi[1], 20)
    elif marker_points_for_roi[2][0] > 0 or marker_points_for_roi[2][1] > 0:
        region_growing_segmentation.simple_region_growing(cropped_image, marker_points_for_roi[2], 20)
    else:
        continue'''
