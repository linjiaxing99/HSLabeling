import os
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import check_dir, read, imsave
import concurrent.futures


def read_cls_color(cls):
    # label2color_dict = {
    #     0: [255, 255, 255],  # Impervious surfaces (RGB: 255, 255, 255)
    #     1: [0, 0, 255],  # Building (RGB: 0, 0, 255)
    #     2: [0, 255, 255],  # Low vegetation (RGB: 0, 255, 255)
    #     3: [0, 255, 0],  # Tree (RGB: 0, 255, 0)
    #     4: [255, 255, 0],  # Car (RGB: 255, 255, 0)
    #     5: [255, 0, 0],  # Clutter/background (RGB: 255, 0, 0)
    # }
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
    return label2color_dict[cls]

def caculate_distance(point, contour):
    return cv2.pointPolygonTest(contour, point, True)


def get_scattered_points(matrix, num_points=1):
    coordinates = np.argwhere(matrix == 1)
    coordinates = np.array([(x, y) for (y, x) in coordinates])
    if len(coordinates) <= num_points:
        return coordinates

    scattered_points = []
    while len(scattered_points) < num_points:
        index = np.random.randint(0, len(coordinates))
        point = coordinates[index]
        scattered_points.append(point)
        coordinates = np.delete(coordinates, index, axis=0)

    return scattered_points

def draw_point(label, kernal_size=100, point_size=3, num_points=1):
    h, w, c = label.shape
    label_set = np.unique(label)

    new_mask = np.ones([h, w, c], np.uint8) * 255
    new_mask_vis = np.zeros([h, w, c], np.uint8)

    for cls in label_set:
        if cls != 255:
            color = read_cls_color(cls)

            temp_mask = np.zeros([h, w])
            temp_mask[label[:, :, 0] == cls] = 255
            temp_mask = np.asarray(temp_mask, dtype=np.uint8)
            num_objects, regions, stats, centroids = cv2.connectedComponentsWithStats(temp_mask, connectivity=8)
            # cv2.drawContours(new_mask, contours, -1, color, 1)

            for i in range(num_objects):
                # if np.all(temp_mask[regions == i] == 0) or stats[i][4] < kernal_size:
                if np.all(temp_mask[regions == i] == 0):
                    continue
                else:
                    tmp = np.where(regions == i, 1, 0)
                    scattered_points = get_scattered_points(tmp, num_points=num_points)
                    for j in range(len(scattered_points)):
                        cx, cy = scattered_points[j][0], scattered_points[j][1]
                        new_mask[cy:cy + point_size, cx:cx + point_size, :] = (cls, cls, cls)
                        new_mask_vis[cy:cy + point_size, cx:cx + point_size, :] = color

    # new_mask = cv2.rotate(new_mask, cv2.ROTATE_90_CLOCKWISE)
    # new_mask_vis = cv2.rotate(new_mask_vis, cv2.ROTATE_90_CLOCKWISE)
    return new_mask, new_mask_vis


def make(root_path):
    train_path = root_path + '/train'
    val_path = root_path + '/val'

    paths = [train_path, val_path]
    for path in paths:
        label_path = path + '/largest_surface_label'

        point_label_path = path + '/point_label'
        point_label_vis_path = path + '/point_label_vis'
        check_dir(point_label_path), check_dir(point_label_vis_path)

        list = os.listdir(label_path)
        for i in tqdm(list):
            label = os.path.join(label_path, i)
            label = read(label)
            new_mask, new_mask_vis = draw_point(label, num_points=3)
            imsave(point_label_path + '/' + i, new_mask)
            imsave(point_label_vis_path + '/' + i, new_mask_vis)


if __name__ == '__main__':
    np.random.seed(2333)
    root_path = '/home/isalab206/Downloads/dataset/potsdam'
    make(root_path)
