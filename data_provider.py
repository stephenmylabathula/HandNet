import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import contour_segmentation
import region_growing_segmentation


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


dataset_path = '/usr/local/google/home/smylabathula/datasets/hand_net_grayscale/1/'
bounding_box_list = np.genfromtxt(dataset_path + 'finger_bounding_box.txt', delimiter=',', dtype=np.int16)
markers_list = np.genfromtxt(dataset_path + 'marker_pixels.txt', delimiter=',', dtype=np.int16)
image_list = os.listdir(dataset_path + 'images')
image_list.sort()
for i in range(len(image_list)):
    image = cv2.imread(dataset_path + 'images/' + image_list[i], 0)
    bounding_box_label_index = np.argwhere(bounding_box_list[:, 0] == i)
    markers_label_index = np.argwhere(markers_list[:, 0] == i)
    if len(bounding_box_label_index) < 1:
        continue
    image_roi = bounding_box_list[bounding_box_label_index.reshape(1)[0]][1:]
    if image_roi[2] + image_roi[3] == 0:
        continue
    marker_points = markers_list[markers_label_index.reshape(1)[0]][1:]
    marker_points = [(marker_points[0], marker_points[1]), (marker_points[2], marker_points[3]), (marker_points[4], marker_points[5])]
    # To Draw Rectangle On Image
    '''
    cv2.rectangle(image, (image_roi[0], image_roi[1]), (image_roi[0] + image_roi[3], image_roi[1] + image_roi[2]), (0,255,0), 3)
    cv2.imshow('image', image)
    cv2.waitKey(10)
    '''
    # To Draw ROI
    '''
    cv2.imshow('image', image[image_roi[1]:image_roi[1] + image_roi[2], image_roi[0]:image_roi[0] + image_roi[3]])
    cv2.waitKey(10)
    '''
    # Generate ROI Crop and Corresponding Marker Points
    cropped_image = image[image_roi[1]:image_roi[1] + image_roi[2], image_roi[0]:image_roi[0] + image_roi[3]]
    marker_points_for_roi = [(marker_points[0][0] - image_roi[0], marker_points[0][1] - image_roi[1]),
                             (marker_points[1][0] - image_roi[0], marker_points[1][1] - image_roi[1]),
                             (marker_points[2][0] - image_roi[0], marker_points[2][1] - image_roi[1])]

    '''
    if contour_segmentation.contour_segment_image_point_in_polygon_grayscale(cropped_image, marker_points_for_roi):
        cv2.imshow('image', cropped_image)
        cv2.waitKey(10)
    '''

    if marker_points_for_roi[0][0] > 0 or marker_points_for_roi[0][1] > 0:
        region_growing_segmentation.simple_region_growing(cropped_image, marker_points_for_roi[0], 20)
    elif marker_points_for_roi[1][0] > 0 or marker_points_for_roi[1][1] > 0:
        region_growing_segmentation.simple_region_growing(cropped_image, marker_points_for_roi[1], 20)
    elif marker_points_for_roi[2][0] > 0 or marker_points_for_roi[2][1] > 0:
        region_growing_segmentation.simple_region_growing(cropped_image, marker_points_for_roi[2], 20)
    else:
        continue
