import os
import numpy as np
import matplotlib.pyplot as plt


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

