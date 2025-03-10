import os
import numpy as np


def label_transform(dataset, label):
    label_new = label.copy()
    if dataset == 'potsdam' or dataset == 'vaihingen':
        label_new[label == 5] = 255
        return label_new
    elif dataset == 'GID':
        return label_new
    else:
        return label_new

    # return label_new


def value_to_rgb(anno, flag='potsdam'):
    if flag == 'potsdam' or flag == 'vaihingen':
        label2color_dict = {
            0: [255, 255, 255], # Impervious surfaces (RGB: 255, 255, 255)
            1: [0, 0, 255], # Building (RGB: 0, 0, 255)
            2: [0, 255, 255], # Low vegetation (RGB: 0, 255, 255)
            3: [0, 255, 0], # Tree (RGB: 0, 255, 0)
            4: [255, 255, 0], # Car (RGB: 255, 255, 0)
            5: [255, 0, 0], # Clutter/background (RGB: 255, 0, 0)
            255: [0, 0, 0]
        }
    elif flag == 'loveda':
        label2color_dict = {
            0: [255, 255, 255],  # Background (RGB: 255, 255, 255)
            1: [255, 0, 0],  # Building (RGB: 0, 0, 255)
            2: [255, 255, 0],  # Road (RGB: 255, 255, 0)
            3: [0, 0, 255],  # Water (RGB: 0, 0, 255)
            4: [159, 129, 183],  # Barren (RGB: 159, 129, 183])
            5: [0, 255, 0],  # Forest (RGB: 0, 255, 0)
            6: [255, 195, 128],  # Agricultural (RGB: 255, 195, 128)
            255: [0, 0, 0],  # boundary (RGB: 0, 0, 0)
        }
    else:
        label2color_dict = {}

    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            # cv2: bgr
            color = label2color_dict[anno[i, j, 0]]

            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno