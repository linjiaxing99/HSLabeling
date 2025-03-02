import os
import random
from utils import check_dir, read, imsave
import numpy as np
import cv2
import shutil
from scipy import ndimage

def read_cls_color(cls):
    label2color_dict = {
        0: [255, 255, 255],  # Impervious surfaces (RGB: 255, 255, 255)
        1: [0, 0, 255],  # Building (RGB: 0, 0, 255)
        2: [0, 255, 255],  # Low vegetation (RGB: 0, 255, 255)
        3: [0, 255, 0],  # Tree (RGB: 0, 255, 0)
        4: [255, 255, 0],  # Car (RGB: 255, 255, 0)
        5: [255, 0, 0],  # Clutter/background (RGB: 255, 0, 0)
    }
    return label2color_dict[cls]

def write_train(budget=8000, value=1):
    train_path = '/home/isalab206/Downloads/dataset/potsdam/train/point_label'
    train_txt = open('/home/isalab206/LJX/HSLabeling-master/datafiles/potsdam/seg_train_point_complexity_' + str(budget) + '.txt', 'w')
    list = os.listdir(train_path)
    budget_statistics = 0
    cls4 = 0

    # budget = budget - 8000
    # f = open('/home/ggm/LJX/HSLabeling-master/datafiles/potsdam/point32000.txt', 'r')
    # alllines = f.readlines()
    # f.close()
    # surface = []
    # for eachline in alllines:
    #     eachline = eachline.strip('\n')
    #     surface.append(eachline)

    for idx, i in enumerate(list):
        # if i in surface:
        #     continue
        label = os.path.join(train_path, i)
        label = read(label)
        h, w, c = label.shape
        label_set = np.unique(label)
        for cls in label_set:
            if cls != 255 and cls != 5:
                temp_mask = np.zeros([h, w])
                temp_mask[label[:, :, 0] == cls] = 255
                temp_mask = np.asarray(temp_mask, dtype=np.uint8)
                num_objects, regions, stats, centroids = cv2.connectedComponentsWithStats(temp_mask, connectivity=8)
                for j in range(num_objects):
                    if np.all(temp_mask[regions == j] == 0):
                        continue
                    else:
                        budget_statistics = budget_statistics + value
                        if cls == 4:
                            cls4 = cls4 + 1
        train_txt.write(i + '\n')
        if budget_statistics > budget:
            break

    train_txt.close()
    print(cls4)
    print(budget_statistics)

def write_train2():
    train_path = '/home/isalab206/Downloads/dataset/potsdam/train/point_label'
    train_txt = open('/home/isalab206/LJX/HSLabeling-master/datafiles/potsdam/seg_train.txt', 'w')
    list = os.listdir(train_path)
    for idx, i in enumerate(list):
        train_txt.write(i + '\n')
    train_txt.close()

def write_train4():
    train_path = '/home/isalab206/Downloads/dataset/potsdam/train/al_vis_20000_HSLabeling'
    f = open('/home/isalab206/LJX/HSLabeling-master/datafiles/potsdam/seg_train_20000_HSLabeling.txt', 'r')
    alllines = f.readlines()
    f.close()
    for eachline in alllines:
        eachline = eachline.strip('\n')
        shutil.copy(os.path.join(train_path, eachline), '/home/isalab206/Downloads/dataset/potsdam/train/al_vis_20000_HSLabeling/' + eachline)

def write_train3(budget, value=5):
    train_path = '/home/isalab206/Downloads/dataset/potsdam/train/largest_surface_label_random'
    train_txt = open('/home/isalab206/LJX/HSLabeling-master/datafiles/potsdam/surface_' + str(budget) + '.txt', 'w')
    list = os.listdir(train_path)
    budget_statistics = 0

    for idx, i in enumerate(list):
        label = os.path.join(train_path, i)
        label = read(label)
        h, w, c = label.shape
        label_set = np.unique(label)
        for cls in label_set:
            if cls != 255 and cls != 5:
                temp_mask = np.zeros([h, w])
                temp_mask[label[:, :, 0] == cls] = 255
                temp_mask = np.asarray(temp_mask, dtype=np.uint8)
                num_objects, regions, stats, centroids = cv2.connectedComponentsWithStats(temp_mask, connectivity=8)
                for j in range(num_objects):
                    if np.all(temp_mask[regions == j] == 0):
                        continue
                    else:
                        budget_statistics = budget_statistics + value
        train_txt.write(i + '\n')
        if budget_statistics > budget:
            break

    train_txt.close()

def generate_point():
    train_path = '/home/isalab206/Downloads/dataset/potsdam/train/largest_surface_label_random'
    point_save = '/home/isalab206/Downloads/dataset/potsdam/train/point_label'
    point_vis_save = '/home/isalab206/Downloads/dataset/potsdam/train/point_vis_label'
    list = os.listdir(train_path)

    for idx, i in enumerate(list):
        label = os.path.join(train_path, i)
        label = read(label)
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
                for j in range(num_objects):
                    if np.all(temp_mask[regions == j] == 0):
                        continue
                    else:
                        cx, cy = int(centroids[j][0]), int(centroids[j][1])
                        new_mask[cy:cy + 3, cx:cx + 3, :] = (cls, cls, cls)
                        new_mask_vis[cy:cy + 3, cx:cx + 3, :] = color
        imsave(point_save + '/' + i, new_mask)
        imsave(point_vis_save + '/' + i, new_mask_vis)


def write_weight():
    train_path = '/home/isalab206/Downloads/dataset/potsdam/train/point_label/'
    f = open('/home/isalab206/LJX/HSLabeling-master/datafiles/potsdam/ppoint20000.txt', 'r')
    alllines = f.readlines()
    f.close()
    for eachline in alllines:
        eachline = eachline.strip('\n')
        label = os.path.join(train_path, eachline)
        label = read(label)
        h, w, c = label.shape
        point_weight = np.zeros([h, w], np.float32)
        line_weight = np.zeros([h, w], np.float32)
        surface_weight = np.zeros([h, w], np.float32)
        label_set = np.unique(label)
        for cls in label_set:
            if cls != 255 and cls != 5:
                temp_mask = np.zeros([h, w])
                temp_mask[label[:, :, 0] == cls] = 255
                temp_mask = np.asarray(temp_mask, dtype=np.uint8)
                num_objects, regions, stats, centroids = cv2.connectedComponentsWithStats(temp_mask, connectivity=8)
                for j in range(num_objects):
                    if np.all(temp_mask[regions == j] == 0):
                        continue
                    else:
                        if stats[j][4] == 9:
                            point_weight[regions == j] = 1
                            continue
                        temp_mask = np.zeros([h, w])
                        temp_mask[regions == j] = 1
                        temp_mask = ndimage.distance_transform_edt(temp_mask)
                        if temp_mask[centroids] < 3:
                            line_weight[regions == j] = 1
                            continue
                        surface_weight[regions == j] = 1

        np.save('/home/ggm/Downloads/dataset/potsdam/train/' + 'point_weight/' + eachline[:-4] + '.npy', point_weight)
        np.save('/home/ggm/Downloads/dataset/potsdam/train/' + 'line_weight/' + eachline[:-4] + '.npy', line_weight)
        np.save('/home/ggm/Downloads/dataset/potsdam/train/' + 'surface_weight/' + eachline[:-4] + '.npy', surface_weight)
        # shutil.copyfile('/home/ggm/Downloads/dataset/potsdam/train/point_label/' + eachline, '/home/ggm/Downloads/dataset/potsdam/train/ppoint32000/' + eachline)

if __name__ == '__main__':
    budget = 6000
    # write_train(budget=budget)
    write_train2()
    # write_train4()
    # generate_point()