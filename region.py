import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys




def on_mouse(event, x, y, flags, params):
    if event == cv2.CV_EVENT_LBUTTONDOWN:
        print 'Start Mouse Position: ' + str(x) + ', ' + str(y)
        s_box = x, y
        boxes.append(s_box)





def region_growing(img, seed):
    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_threshold = 0.2
    region_size = 1
    intensity_difference = 0
    neighbor_points_list = []
    neighbor_intensity_list = []

    #Mean of the segmented region
    region_mean = img[seed]

    #Input image parameters
    height, width = img.shape
    image_size = height * width

    #Initialize segmented output image
    segmented_img = np.zeros((height, width, 1), np.uint8)

    #Region growing until intensity difference becomes greater than certain threshold
    while (intensity_difference < region_threshold) & (region_size < image_size):
        #Loop through neighbor pixels
        for i in range(4):
            #Compute the neighbor pixel position
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            #Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

            #Add neighbor if inside and not already in segmented_img
            if check_inside:
                if segmented_img[x_new, y_new] == 0:
                    neighbor_points_list.append([x_new, y_new])
                    neighbor_intensity_list.append(img[x_new, y_new])
                    segmented_img[x_new, y_new] = 255

        #Add pixel with intensity nearest to the mean to the region
        distance = abs(neighbor_intensity_list-region_mean)
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        segmented_img[seed[0], seed[1]] = 255
        region_size += 1

        #New region mean
        region_mean = (region_mean*region_size + neighbor_intensity_list[index])/(region_size+1)

        #Update the seed value
        seed = neighbor_points_list[index]
        #Remove the value from the neighborhood lists
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]

    return segmented_img


if __name__ == '__main__':

    boxes = []
    filename = 'hand.jpg'
    img = cv2.imread(filename, 0)
    resized = cv2.resize(img,(256,256))
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', on_mouse, 0,)
    cv2.imshow('input', resized)
    cv2.waitKey()
    print "Starting region growing based on last click"
    seed = boxes[-1]
    cv2.imshow('input', region_growing(resized, seed))
    print "Done. Showing output now"

    cv2.waitKey()
    cv2.destroyAllWindows()







visited_points = None
image = None
four_connectivity = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def RegionGrowing(input_image, start_point, threshold):
    global image, visited_points
    image = input_image
    visited_points = np.zeros(shape=image.shape[:2], dtype=np.int16)
    RegionGrowingHelper(start_point, image[start_point], threshold)


def PointInImage(image, point):
    in_x = 0 <= point[0] < image.shape[1]
    in_y = 0 <= point[1] < image.shape[0]
    return in_x and in_y


def PointInColorRange(image, point, color, threshold):
    point_color = image[point[1], point[0]]
    color_distance = np.linalg.norm(point_color - color)
    return color_distance < threshold


def PointVisited(point):
    global visited_points
    return visited_points[point] > 0


def RegionGrowingHelper(point, mean_color, threshold):
    global image, four_connectivity
    if not PointInImage(image, point):
        return
    if PointVisited():
        return
    if not PointInColorRange(image, point, mean_color, threshold):
        return
    visited_points[point] += 1
    RegionGrowingHelper(np.add(point, four_connectivity[0]), mean_color, threshold)
    RegionGrowingHelper(np.add(point, four_connectivity[1]), mean_color, threshold)
    RegionGrowingHelper(np.add(point, four_connectivity[2]), mean_color, threshold)
    RegionGrowingHelper(np.add(point, four_connectivity[3]), mean_color, threshold)
















