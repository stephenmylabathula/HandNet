import sys
import cv2 as cv
import numpy as np

def simple_region_growing(img, seed, threshold=1):
    dims = img.shape

    reg = np.zeros(dims)

    #parameters
    mean_reg = float(img[seed[1], seed[0]])
    size = 1
    pix_area = dims[0]*dims[1]

    contour = [] # will be [ [[x1, y1], val1],..., [[xn, yn], valn] ]
    contour_val = []
    dist = 0
    # TODO: may be enhanced later with 8th connectivity
    orient = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity
    cur_pix = [seed[0], seed[1]]

    # Spreading
    while dist < threshold and size < pix_area:
        # adding pixels
        for j in range(4):
            # select new candidate
            temp_pix = [cur_pix[0] + orient[j][0], cur_pix[1] + orient[j][1]]

            # check if it belongs to the image
            is_in_img = dims[1] > temp_pix[0] > 0 and dims[0] > temp_pix[1] > 0  # returns boolean
            # candidate is taken if not already selected before
            if is_in_img and (reg[temp_pix[1], temp_pix[0]] == 0):
                contour.append(temp_pix)
                contour_val.append(img[temp_pix[1], temp_pix[0]])
                reg[temp_pix[1], temp_pix[0]] = 150
        # add the nearest pixel of the contour in it
        dist = abs(int(np.mean(contour_val)) - mean_reg)

        dist_list = [abs(i - mean_reg) for i in contour_val ]
        dist = min(dist_list)    # get min distance
        index = dist_list.index(min(dist_list)) # mean distance index
        size += 1 # updating region size
        reg[cur_pix[1], cur_pix[0]] = 255

        # updating mean MUST BE FLOAT
        mean_reg = (mean_reg*size + float(contour_val[index]))/(size+1)
        # updating seed
        cur_pix = contour[index]

        # removing pixel from neigborhood
        del contour[index]
        del contour_val[index]

        cv.imshow("Image", reg)
        cv.waitKey(1)

    return reg



# Load Hand Image
hand_image = cv.imread("hand.jpg")
hand_image = cv.resize(hand_image, (0, 0), fx=0.2, fy=0.2)
hand_image_gray = cv.cvtColor(hand_image, cv.COLOR_BGR2GRAY)

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
cropped_finger_gray = cv.cvtColor(cropped_finger, cv.COLOR_BGR2GRAY)

threshold = 30

out_img = simple_region_growing(cropped_finger_gray, hand_points_ground_truth_cropped[1], threshold)

cv.imshow("Image", out_img)
cv.waitKey(0)