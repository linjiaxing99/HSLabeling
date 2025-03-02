from __future__ import print_function
from __future__ import division
import os
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import check_dir, read, imsave
import largestinteriorrectangle as lir
from builtins import range
from past.utils import old_div
from scipy import ndimage, optimize
import pdb
import matplotlib.patches as patches
import multiprocessing
import datetime
from functools import reduce
import largestinteriorrectangle as lir
import copy
import math
import random
from make_longest_line_label import Point, find_longest_segment
from make_largest_surface_label import findRotMaxRect


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


def calculate_area(points):
    # 解构四个点的坐标
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    area = abs(0.5 * ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (x2 * y1 + x3 * y2 + x4 * y3 + x1 * y4)))
    return area


def distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def calculate_new_coordinates_surface(points, scale=0.3, offset_scale=1):
    # 计算原始四边形的面积
    original_area = calculate_area(points)

    # 计算新四边形的面积（原始面积的30%）
    new_area = scale * original_area

    # 计算面积比例因子
    scale_factor = math.sqrt(new_area / original_area)

    # 计算外接矩形
    points = np.float32(points)
    rect = cv2.minAreaRect(points)
    center, angle = rect[0], rect[2]

    # 将角度置为0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 对矩形的各点坐标进行逆时针旋转变换
    rotated_points = cv2.transform(points.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)

    center_x = sum([x for x, _ in rotated_points]) / 4
    center_y = sum([y for _, y in rotated_points]) / 4

    # 计算大矩形的宽度和高度
    width = max([x for x, _ in rotated_points]) - min([x for x, _ in rotated_points])
    height = max([y for _, y in rotated_points]) - min([y for _, y in rotated_points])

    # 计算小矩形的宽度和高度（使其面积等于大矩形的30%）
    small_width = math.sqrt(scale * width * height * (width / height))
    small_height = (scale * width * height) / small_width

    # 计算小矩形的移动范围
    max_x_offset = (width - small_width) / 2 * offset_scale
    max_y_offset = (height - small_height) / 2 * offset_scale

    # 生成随机的偏移量
    offset_x = random.uniform(-max_x_offset, max_x_offset)
    offset_y = random.uniform(-max_y_offset, max_y_offset)

    # 计算新中心点坐标
    new_center_x = center_x + offset_x
    new_center_y = center_y + offset_y

    rotated_point = np.array([[new_center_x, new_center_y]], dtype=np.float32)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
    new_center_point = cv2.transform(rotated_point.reshape(-1, 1, 2), rotation_matrix).reshape(2)

    # 计算顶点相对于中心点的偏移量并计算新四边形的顶点坐标
    new_points = []
    for x, y in points:
        offset_x = (x - new_center_point[0]) * scale_factor
        offset_y = (y - new_center_point[1]) * scale_factor
        new_x = new_center_point[0] + offset_x
        new_y = new_center_point[1] + offset_y
        new_points.append([new_x, new_y])

    return new_points


def calculate_new_coordinates_line(point1, point2, scale=0.5):
    # 计算原线段的中点坐标
    # center = (point1 + point2) / 2

    # 计算原线段的长度
    original_length = distance(point1, point2)

    # 计算新线段的长度（原线段长度的50%）
    new_length = scale * original_length

    # 计算原线段的方向向量
    direction = (point2 - point1) / original_length

    start_point = point1 + direction.__dot__(new_length / 2)
    end_point = point2 - direction.__dot__(new_length / 2)

    # t = random.uniform(0.25, 0.75)
    t = random.uniform(0, 1)
    # 根据参数方程计算点的坐标
    center_x = start_point.x + t * (end_point.x - start_point.x)
    center_y = start_point.y + t * (end_point.y - start_point.y)
    new_center = Point(center_x, center_y)
    # 计算新线段的起点坐标和终点坐标
    new_point1 = new_center - direction.__dot__(new_length / 2)
    new_point2 = new_center + direction.__dot__(new_length / 2)

    return (new_point1, new_point2)


def generate_line(points, line_length, offscale=1):
    random_index = random.randint(0, 1)
    # 计算外接矩形
    points = np.float32(points)
    rect = cv2.minAreaRect(points)
    center, angle = rect[0], rect[2]

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 对矩形的各点坐标进行逆时针旋转变换
    rotated_points = cv2.transform(points.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
    x2, y2 = rotated_points[random_index]
    x4, y4 = rotated_points[random_index + 2]
    # 计算矩形的对角线长度
    # diagonal_length = math.sqrt((x2 - x4)**2 + (y2 - y4)**2)

    # x1, y1 = points[0]
    # x3, y3 = points[2]
    # diagonal_length = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    if abs(x2 - x4) < 1e-8:
        return 0
    # 计算对角线斜率
    diagonal_slope = (y2 - y4) / (x2 - x4)

    # 计算线段在x轴上的距离
    x_delta = abs(line_length / math.sqrt(1 + diagonal_slope**2))
    y_delta = abs(diagonal_slope * x_delta)

    min_x = min(x2, x4)
    max_x = max(x2, x4)
    min_y = min(y2, y4)
    max_y = max(y2, y4)

    offset_x = random.uniform(-(max_x - min_x - x_delta) / 2, (max_x - min_x - x_delta) / 2) * offscale
    offset_y = random.uniform(-(max_y - min_y - y_delta) / 2, (max_y - min_y - y_delta) / 2) * offscale

    center_x = min_x + x_delta / 2 + (max_x - min_x - x_delta) / 2 + offset_x
    center_y = min_y + y_delta / 2 + (max_y - min_y - y_delta) / 2 + offset_y

    if random_index == 1:
        start_x = center_x - x_delta / 2
        start_y = center_y - y_delta / 2
        end_x = center_x + x_delta / 2
        end_y = center_y + y_delta / 2
    else:
        start_x = center_x - x_delta / 2
        start_y = center_y + y_delta / 2
        end_x = center_x + x_delta / 2
        end_y = center_y - y_delta / 2

    rotated_point_start = np.array([[start_x, start_y]], dtype=np.float32)
    rotated_point_end = np.array([[end_x, end_y]], dtype=np.float32)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
    rotated_point_start = cv2.transform(rotated_point_start.reshape(-1, 1, 2), rotation_matrix).reshape(2)
    rotated_point_end = cv2.transform(rotated_point_end.reshape(-1, 1, 2), rotation_matrix).reshape(2)

    return (Point(rotated_point_start[0], rotated_point_start[1]), Point(rotated_point_end[0], rotated_point_end[1]))


FFT_MEASURE_MAX = np.sqrt(np.power(0.5, 2) + np.power(0.5, 2))
def fft_measure(img):
    np_img = img
    fft = np.fft.fft2(np_img)

    fft_abs = np.abs(fft)

    n = fft.shape[0]
    pos_f_idx = n // 2
    df = np.fft.fftfreq(n=n)  # type: ignore

    amplitude_sum = fft_abs[:pos_f_idx, :pos_f_idx].sum()
    mean_x_freq = (fft_abs * df)[:pos_f_idx, :pos_f_idx].sum() / amplitude_sum
    mean_y_freq = (fft_abs.T * df).T[:pos_f_idx, :pos_f_idx].sum() / amplitude_sum

    mean_freq = np.sqrt(np.power(mean_x_freq, 2) + np.power(mean_y_freq, 2))

    # mean frequency in range 0 to np.sqrt(0.5^2 + 0.5^2)
    return mean_freq / FFT_MEASURE_MAX


def draw_sl(label, kernal_size=100, scale_surface=1, scale_line=1):

    h, w, c = label.shape
    label_set = np.unique(label)

    new_mask_surface = np.ones([h, w, c], np.uint8) * 255
    new_mask_vis_surface = np.zeros([h, w, c], np.uint8)
    new_mask_vis_surface = np.ones([h, w, c], np.uint8) * 227
    new_mask_line = np.ones([h, w, c], np.uint8) * 255
    new_mask_vis_line = np.zeros([h, w, c], np.uint8)
    new_mask_vis_line = np.ones([h, w, c], np.uint8) * 227

    for cls in label_set:
        if cls != 255:
            color = read_cls_color(cls)

            temp_mask = np.zeros([h, w])
            temp_mask[label[:, :, 0] == cls] = 255
            temp_mask = np.asarray(temp_mask, dtype=np.uint8)
            num_objects, regions, stats, centroids = cv2.connectedComponentsWithStats(temp_mask, connectivity=8)
            # cv2.drawContours(new_mask, contours, -1, color, 1)

            for i in range(num_objects):
                if np.all(temp_mask[regions == i] == 0) or stats[i][4] < kernal_size:
                    continue
                else:
                    tmp = np.where(regions == i, 1, 0)
                    complexity = 1 - fft_measure(tmp)
                    # get surface
                    tmp = np.where(regions == i, 0, 1)  # 满足大于0的值保留，不满足的设为1
                    rect_coord_ori, angle, coord_out_rot = findRotMaxRect(tmp,
                                                                          flag_opt=True,
                                                                          nbre_angle=25,
                                                                          flag_parallel=True,
                                                                          flag_out='rotation',
                                                                          flag_enlarge_img=False,
                                                                          limit_image_size=100)
                    if rect_coord_ori == None or calculate_area(rect_coord_ori) <= 4:
                        continue
                    for j in range(4):
                        rect_coord_ori[j][1] = tmp.shape[0] - rect_coord_ori[j][1]
                    # new_rect_coord_ori = copy.deepcopy(rect_coord_ori)
                    for j in range(4):
                        k = rect_coord_ori[j][1]
                        rect_coord_ori[j][1] = rect_coord_ori[j][0]
                        rect_coord_ori[j][0] = tmp.shape[0] - k
                    new_rect_coord_ori = calculate_new_coordinates_surface(rect_coord_ori, scale=complexity)
                    pts = np.round(new_rect_coord_ori).astype(int)
                    pts = pts.reshape((-1, 2))
                    cv2.fillPoly(new_mask_surface, [pts], (int(cls), int(cls), int(cls)))
                    cv2.fillPoly(new_mask_vis_surface, [pts], tuple([int(c) for c in color]))

                    # get line
                    tmp = np.where(regions == i, 1, 0)  # 满足大于0的值保留，不满足的设为0
                    ans, longest_segment = find_longest_segment(tmp)
                    diagonal = min(distance(Point(rect_coord_ori[0][0], rect_coord_ori[0][1]), Point(rect_coord_ori[2][0], rect_coord_ori[2][1])),
                                   distance(Point(rect_coord_ori[1][0], rect_coord_ori[1][1]), Point(rect_coord_ori[3][0], rect_coord_ori[3][1])))
                    if diagonal <= ans * complexity:
                        new_longest_segment = calculate_new_coordinates_line(longest_segment[0], longest_segment[1],
                                                                         scale=complexity)
                    else:
                        new_longest_segment = generate_line(rect_coord_ori, line_length=ans * complexity)
                        if new_longest_segment == 0:
                            new_longest_segment = calculate_new_coordinates_line(longest_segment[0], longest_segment[1],
                                                                             scale=complexity)
                    cv2.line(new_mask_line, (int(new_longest_segment[0].x), int(new_longest_segment[0].y)),
                             (int(new_longest_segment[1].x), int(new_longest_segment[1].y)),
                             color=(int(cls), int(cls), int(cls)), thickness=3)
                    cv2.line(new_mask_vis_line, (int(new_longest_segment[0].x), int(new_longest_segment[0].y)),
                             (int(new_longest_segment[1].x), int(new_longest_segment[1].y)),
                             color=color, thickness=3)

    # new_mask_surface = cv2.rotate(new_mask_surface, cv2.ROTATE_90_CLOCKWISE)
    # new_mask_vis_surface = cv2.rotate(new_mask_vis_surface, cv2.ROTATE_90_CLOCKWISE)
    return new_mask_surface, new_mask_vis_surface, new_mask_line, new_mask_vis_line


def make(root_path):
    train_path = root_path + '/train'
    val_path = root_path + '/val'

    paths = [train_path, val_path]
    for path in paths:
        label_path = path + '/label'

        surface_label_path = path + '/largest_surface_label_random_complexity'
        surface_label_vis_path = path + '/largest_surface_label_vis_random_complexity'
        line_label_path = path + '/longest_line_label_random_complexity'
        line_label_vis_path = path + '/longest_line_label_vis_random_complexity'
        check_dir(surface_label_path), check_dir(surface_label_vis_path)
        check_dir(line_label_path), check_dir(line_label_vis_path)

        list = os.listdir(label_path)
        for i in tqdm(list):
        # for i in list:
        #     print(i)
            label = os.path.join(label_path, i)
            label = read(label)
            new_mask_surface, new_mask_vis_surface, new_mask_line, new_mask_vis_line = draw_sl(label)
            imsave(surface_label_path + '/' + i, new_mask_surface)
            imsave(surface_label_vis_path + '/' + i, new_mask_vis_surface)
            imsave(line_label_path + '/' + i, new_mask_line)
            imsave(line_label_vis_path + '/' + i, new_mask_vis_line)

if __name__ == '__main__':
    # root_path = '/home/ggm/Downloads/dataset/potsdam'
    # # random.seed(2333)
    # make(root_path)

    label_path = '/home/isalab206/Downloads/dataset/potsdam/train/vis_label_vis'

    surface_label_path = '/home/isalab206/Downloads/dataset/potsdam/train/vis_surface'
    surface_label_vis_path = '/home/isalab206/Downloads/dataset/potsdam/train/vis_surface_vis'
    line_label_path = '/home/isalab206/Downloads/dataset/potsdam/train/vis_line'
    line_label_vis_path = '/home/isalab206/Downloads/dataset/potsdam/train/vis_line_vis'
    check_dir(surface_label_path), check_dir(surface_label_vis_path)
    check_dir(line_label_path), check_dir(line_label_vis_path)

    list = os.listdir(label_path)
    for i in tqdm(list):
        # for i in list:
        #     print(i)
        label = os.path.join(label_path, i)
        label = read(label)
        new_mask_surface, new_mask_vis_surface, new_mask_line, new_mask_vis_line = draw_sl(label)
        imsave(surface_label_path + '/' + i, new_mask_surface)
        imsave(surface_label_vis_path + '/' + i, new_mask_vis_surface)
        imsave(line_label_path + '/' + i, new_mask_line)
        imsave(line_label_vis_path + '/' + i, new_mask_vis_line)
