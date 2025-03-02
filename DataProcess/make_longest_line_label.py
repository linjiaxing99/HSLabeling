import os
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import check_dir, read, imsave
from PIL import Image
import math
from itertools import combinations
from threading import Thread, Lock
from multiprocessing import Pool
import multiprocessing


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

# class Point:
#     def __init__(self, x=0.0, y=0.0):
#         self.x = x
#         self.y = y
#
#     def __eq__(self, other):
#         return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)
#
#     def __lt__(self, other):
#         if math.isclose(self.x, other.x):
#             return self.y < other.y
#         return self.x < other.x
#
#     def __sub__(self, other):
#         return Point(self.x - other.x, self.y - other.y)
#
#     def __xor__(self, other):
#         return self.x * other.y - self.y * other.x
#
#     def __mul__(self, other):
#         return self.x * other.x + self.y * other.y


class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __lt__(self, other):
        if math.isclose(self.x, other.x):
            return self.y < other.y
        return self.x < other.x

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __xor__(self, other):
        return self.x * other.y - self.y * other.x

    def __mul__(self, other):
        return self.x * other.x + self.y * other.y

    def __dot__(self, other):
        return Point(self.x * other, self.y * other)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)


class Line:
    def __init__(self, s=None, t=None):
        self.s = s if s is not None else Point()
        self.t = t if t is not None else Point()

    def crosspoint(self, v):
        a1 = (v.t - v.s) ^ (self.s - v.s)
        a2 = (v.t - v.s) ^ (self.t - v.s)
        x = (self.s.x * a2 - self.t.x * a1) / (a2 - a1)
        y = (self.s.y * a2 - self.t.y * a1) / (a2 - a1)
        return Point(x, y)

    def pointonseg(self, q):
        return math.isclose((q - self.s) ^ (self.t - self.s), 0.0) and (q - self.s) * (q - self.t) <= 0.0


def sgn(x, epsilon=1e-10):
    if x > epsilon:
        return 1
    elif x < -epsilon:
        return -1
    else:
        return 0


def distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


# def point_on_poly(point, polygon):
#     # 将多边形的坐标转换为 np.array 类型
#     polygon_arr = np.array([(p.x, p.y) for p in polygon], dtype=np.int32)
#
#     # 将点的坐标转换为 np.array 类型
#     point_arr = tuple([int(round(point.x)), int(round(point.y))])
#
#     # 使用 pointPolygonTest 函数来判断点是否在多边形上
#     result = cv2.pointPolygonTest(polygon_arr, point_arr, False)
#     return result >= 0

def point_on_poly(q, polygon):
    n = len(polygon)
    res = 0

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        line = Line(p1, p2)

        if line.pointonseg(q):
            return True

        d1 = sgn(p1.y - q.y)
        d2 = sgn(p2.y - q.y)
        k = sgn((p2 - p1) ^ (q - p1))

        if k > 0 and d1 <= 0 and d2 > 0:
            res += 1
        if k < 0 and d2 <= 0 and d1 > 0:
            res -= 1

    return True if res else False


def calculate_segment(poly, p1, p2):
    v = []
    l = Line(poly[p1], poly[p2])
    n = len(poly) - 1
    for i in range(n):
        if sgn((l.t - l.s) ^ (poly[i] - l.s)) * sgn((l.t - l.s) ^ (poly[i + 1] - l.s)) <= 0:
            v1 = l.t - l.s
            v2 = poly[i + 1] - poly[i]

            if sgn(v1 ^ v2) == 0:
                v.append(poly[i])
                v.append(poly[i + 1])
            else:
                v.append(l.crosspoint(Line(poly[i], poly[i + 1])))

    unique_points = []
    # 遍历原始列表，去除相等的对象
    for point in v:
        if point not in unique_points:
            unique_points.append(point)
    v = unique_points
    v.sort()
    cnt = len(v)
    res = 0
    segment = []
    ans = 0
    longest_segment = []

    for i in range(1, cnt):
        if point_on_poly(Point((v[i - 1].x + v[i].x) / 2.0, (v[i - 1].y + v[i].y) / 2.0), poly):
            res += distance(v[i - 1], v[i])
            if segment == []:
                segment.append(v[i - 1])
                segment.append(v[i])
            else:
                segment.append(v[i])
        else:
            if res > ans:
                ans = res
                longest_segment = segment
            res = 0
            segment = []
            if distance(v[i], v[-1]) <= ans:
                continue
        if res > ans:
            ans = res
            longest_segment = segment
    return ans, longest_segment


# def find_longest_segment(mask):
#     mask = np.asarray(mask, dtype=np.uint8)
#     contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     approxs = []
#     for contour in contours:
#         epsilon = 0.01 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         approxs.extend(approx)
#     poly = [Point(point[0][0], point[0][1]) for point in approxs]
#
#     n = len(poly)
#     poly.append(poly[0])
#     if n < 3:
#         return distance(poly[0], poly[1]), (poly[0], poly[1])
#     combinations_list = list(combinations(range(n), 2))
#     # 创建进程池
#     cpus = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=cpus)
#     # 并行计算结果
#     results = pool.starmap(calculate_segment, [(poly, p1, p2) for p1, p2 in combinations_list])
#     # 关闭进程池
#     pool.close()
#     pool.join()
#     # 解析计算结果
#     ans = 0
#     longest_segment = []
#     for res, segment in results:
#         if res > ans:
#             ans = res
#             longest_segment = segment.copy()
#
#     return ans, (longest_segment[0], longest_segment[-1])

def find_longest_segment(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    approxs = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approxs.extend(approx)
    poly = [Point(point[0][0], point[0][1]) for point in approxs]

    n = len(poly)
    poly.append(poly[0])
    ans = 0.0
    longest_segment = []
    if n < 3:
        return distance(poly[0], poly[1]), (poly[0], poly[1])
    for p1, p2 in combinations(range(n), 2):
        v = []
        l = Line(poly[p1], poly[p2])

        for i in range(n):
            if sgn((l.t - l.s) ^ (poly[i] - l.s)) * sgn((l.t - l.s) ^ (poly[i + 1] - l.s)) <= 0:
                v1 = l.t - l.s
                v2 = poly[i + 1] - poly[i]

                if sgn(v1 ^ v2) == 0:
                    v.append(poly[i])
                    v.append(poly[i + 1])
                else:
                    v.append(l.crosspoint(Line(poly[i], poly[i + 1])))

        unique_points = []
        # 遍历原始列表，去除相等的对象
        for point in v:
            if point not in unique_points:
                unique_points.append(point)
        v = unique_points
        v.sort()
        cnt = len(v)
        res = 0
        segment = []

        for i in range(1, cnt):
            if point_on_poly(Point((v[i - 1].x + v[i].x) / 2.0, (v[i - 1].y + v[i].y) / 2.0), poly):
                res += distance(v[i - 1], v[i])
                if segment == []:
                    segment.append(v[i - 1])
                    segment.append(v[i])
                else:
                    segment.append(v[i])
            else:
                if res > ans:
                    ans = res
                    longest_segment = segment
                res = 0
                segment = []
                if distance(v[i], v[-1]) <= ans:
                    continue
            if res > ans:
                ans = res
                longest_segment = segment

    return ans, (longest_segment[0], longest_segment[-1])


def draw_longline(label, kernal_size=50):
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
                if np.all(temp_mask[regions == i] == 0) or stats[i][4] < kernal_size:
                    continue
                else:
                    tmp = np.where(regions == i, 1, 0)  # 满足大于0的值保留，不满足的设为0
                    ans, longest_segment = find_longest_segment(tmp)
                    cv2.line(new_mask, (int(longest_segment[0].x), int(longest_segment[0].y)), (int(longest_segment[1].x), int(longest_segment[1].y)),
                             color=(int(cls), int(cls), int(cls)), thickness=3)
                    cv2.line(new_mask_vis, (int(longest_segment[0].x), int(longest_segment[0].y)), (int(longest_segment[1].x), int(longest_segment[1].y)),
                             color=color, thickness=3)

    return new_mask, new_mask_vis


def make(root_path):
    train_path = root_path + '/train'
    val_path = root_path + '/val'

    paths = [train_path, val_path]
    for path in paths:
        label_path = path + '/label'

        point_label_path = path + '/long_line_label_test'
        point_label_vis_path = path + '/long_line_label_vis_test'
        check_dir(point_label_path), check_dir(point_label_vis_path)

        list = os.listdir(label_path)
        for i in tqdm(list):
        # for i in list:
        #     print(i)
            label = os.path.join(label_path, i)
            label = read(label)
            new_mask, new_mask_vis = draw_longline(label)
            imsave(point_label_path + '/' + i, new_mask)
            imsave(point_label_vis_path + '/' + i, new_mask_vis)


if __name__ == '__main__':
    root_path = '/home/isalab206/Downloads/dataset/potsdam'
    make(root_path)
