from __future__ import print_function
from __future__ import division
import os
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import check_dir, read, imsave
from builtins import range
from past.utils import old_div
from scipy import ndimage, optimize
import pdb
import matplotlib.patches as patches
import multiprocessing
import datetime
from functools import reduce
import largestinteriorrectangle as lir
import math
import time


def expand_image_boundary(image, expand_size=1.5):
    height, width = image.shape

    # 计算新的图像尺寸
    new_height = int(height * expand_size)
    new_width = int(width * expand_size)

    # 创建新的扩展边界图像
    expanded_image = np.ones((new_height, new_width), dtype=image.dtype)
    top = int((new_height - height) / 2)
    bottom = height + top
    left = int((new_width - width) / 2)
    right = width + left

    # 将原始图像复制到新图像中间位置
    expanded_image[top: bottom, left: right] = image

    return expanded_image, top, left

########################################################################
def residual(angle, data):
    data, _, _ = expand_image_boundary(data)
    nx, ny = data.shape
    M = cv2.getRotationMatrix2D((old_div((nx - 1), 2), old_div((ny - 1), 2)), angle[0], 1)
    RotData = cv2.warpAffine(data, M, (nx, ny), flags=cv2.INTER_NEAREST, borderValue=1)
    RotData = np.logical_not(RotData.astype(np.bool_))
    # rectangle = findMaxRect(RotData)
    cv_grid = RotData.astype("uint8") * 255
    contours, _ = cv2.findContours(cv_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contour = contours[1][:, 0, :]
    if len(contours) == 0:
        return 1. / 1e-6
    else:
        contour = max(contours, key=len)[:, 0, :]
        rectangle = lir.lir(RotData, contour)

    return 1. / (rectangle[2] * rectangle[3])


########################################################################
def residual_star(args):
    return residual(*args)


########################################################################
def get_rectangle_coord(angle, data, flag_out=None):
    data0 = data
    nx, ny = data.shape
    M0 = cv2.getRotationMatrix2D((old_div((nx - 1), 2), old_div((ny - 1), 2)), angle, 1)
    data, top, left = expand_image_boundary(data)
    nx, ny = data.shape
    M = cv2.getRotationMatrix2D((old_div((nx - 1), 2), old_div((ny - 1), 2)), angle, 1)
    RotData = cv2.warpAffine(data, M, (nx, ny), flags=cv2.INTER_NEAREST, borderValue=1)
    # rectangle = findMaxRect(RotData)
    RotData = np.logical_not(RotData.astype(np.bool_))
    # rectangle = findMaxRect(RotData)
    cv_grid = RotData.astype("uint8") * 255
    contours, _ = cv2.findContours(cv_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contour = contours[1][:, 0, :]
    contour = max(contours, key=len)[:, 0, :]
    rectangle = lir.lir(RotData, contour)

    if flag_out:
        return (rectangle[1] - top, rectangle[0] - left, rectangle[1] + rectangle[3] - 1 - top, rectangle[0] + rectangle[2] - 1 - left), M0, RotData
    else:
        return (rectangle[1] - top, rectangle[0] - left, rectangle[1] + rectangle[3] - 1 - top, rectangle[0] + rectangle[2] - 1 - left), M0


########################################################################
def findRotMaxRect(data_in, flag_opt=False, flag_parallel=False, nbre_angle=10, flag_out=None, flag_enlarge_img=False,
                   limit_image_size=300):
    '''
    flag_opt     : True only nbre_angle are tested between 90 and 180
                        and a opt descent algo is run on the best fit
                   False 100 angle are tested from 90 to 180.
    flag_parallel: only valid when flag_opt=False. the 100 angle are run on multithreading
    flag_out     : angle and rectangle of the rotated image are output together with the rectangle of the original image
    flag_enlarge_img : the image used in the function is double of the size of the original to ensure all feature stay in when rotated
    limit_image_size : control the size numbre of pixel of the image use in the function.
                       this speeds up the code but can give approximated results if the shape is not simple
    '''

    # time_s = datetime.datetime.now()

    # make the image square
    # ----------------
    nx_in, ny_in = data_in.shape
    if nx_in != ny_in:
        n = max([nx_in, ny_in])
        data_square = np.ones([n, n])
        xshift = old_div((n - nx_in), 2)
        yshift = old_div((n - ny_in), 2)
        if yshift == 0:
            data_square[xshift:(xshift + nx_in), :] = data_in[:, :]
        else:
            data_square[:, yshift:(yshift + ny_in)] = data_in[:, :]
    else:
        xshift = 0
        yshift = 0
        data_square = data_in

    # apply scale factor if image bigger than limit_image_size
    # ----------------
    if data_square.shape[0] > limit_image_size:
        data_small = cv2.resize(data_square, (limit_image_size, limit_image_size), interpolation=0)
        scale_factor = old_div(1. * data_square.shape[0], data_small.shape[0])
    else:
        data_small = data_square
        scale_factor = 1

    # set the input data with an odd number of point in each dimension to make rotation easier
    # ----------------
    nx, ny = data_small.shape
    nx_extra = -nx;
    ny_extra = -ny
    if nx % 2 == 0:
        nx += 1
        nx_extra = 1
    if ny % 2 == 0:
        ny += 1
        ny_extra = 1
    data_odd = np.ones([data_small.shape[0] + max([0, nx_extra]), data_small.shape[1] + max([0, ny_extra])])
    data_odd[:-nx_extra, :-ny_extra] = data_small
    nx, ny = data_odd.shape

    nx_odd, ny_odd = data_odd.shape

    if flag_enlarge_img:
        data = np.zeros([2 * data_odd.shape[0] + 1, 2 * data_odd.shape[1] + 1]) + 1
        nx, ny = data.shape
        data[old_div(nx, 2) - old_div(nx_odd, 2):old_div(nx, 2) + old_div(nx_odd, 2),
        old_div(ny, 2) - old_div(ny_odd, 2):old_div(ny, 2) + old_div(ny_odd, 2)] = data_odd
    else:
        data = np.copy(data_odd)
        nx, ny = data.shape

    # print (datetime.datetime.now()-time_s).total_seconds()
    if data.min() == 1.:
        if flag_out is None:
            return None
        elif flag_out == 'rotation':
            return None, None, None
    if flag_opt:
        myranges_brute = ([(90., 180.), ])
        coeff0 = np.array([0., ])
        coeff1 = optimize.brute(residual, myranges_brute, args=(data,), Ns=nbre_angle, finish=None)
        popt = optimize.fmin(residual, coeff1, args=(data,), xtol=5, ftol=1.e-5, disp=False)
        angle_selected = popt[0]

        # rotation_angle = np.linspace(0,360,100+1)[:-1]
        # mm = [residual(aa,data) for aa in rotation_angle]
        # plt.plot(rotation_angle,mm)
        # plt.show()
        # pdb.set_trace()

    else:
        rotation_angle = np.linspace(90, 180, 100 + 1)[:-1]
        args_here = []
        for angle in rotation_angle:
            args_here.append([angle, data])

        if flag_parallel:

            # set up a pool to run the parallel processing
            cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation

            results = pool.map(residual_star, args_here)

            pool.close()
            pool.join()


        else:
            results = []
            for arg in args_here:
                results.append(residual_star(arg))

        argmin = np.array(results).argmin()
        angle_selected = args_here[argmin][0]
    rectangle, M_rect_max, RotData = get_rectangle_coord(angle_selected, data, flag_out=True)
    # rectangle, M_rect_max  = get_rectangle_coord(angle_selected,data)

    # print (datetime.datetime.now()-time_s).total_seconds()

    # invert rectangle
    M_invert = cv2.invertAffineTransform(M_rect_max)
    rect_coord = [rectangle[:2], [rectangle[0], rectangle[3]],
                  rectangle[2:], [rectangle[2], rectangle[1]]]

    # ax = plt.subplot(111)
    # ax.imshow(RotData.T,origin='lower',interpolation='nearest')
    # patch = patches.Polygon(rect_coord, edgecolor='k', facecolor='None', linewidth=2)
    # ax.add_patch(patch)
    # plt.show()

    rect_coord_ori = []
    for coord in rect_coord:
        rect_coord_ori.append(np.dot(M_invert, [coord[0], (ny - 1) - coord[1], 1]))

    # transform to numpy coord of input image
    coord_out = []
    for coord in rect_coord_ori:
        coord_out.append([scale_factor * round(coord[0] - (old_div(nx, 2) - old_div(nx_odd, 2)), 0) - xshift, \
                          scale_factor * round((ny - 1) - coord[1] - (old_div(ny, 2) - old_div(ny_odd, 2)),
                                               0) - yshift])

    coord_out_rot = []
    coord_out_rot_h = []
    for coord in rect_coord:
        coord_out_rot.append([scale_factor * round(coord[0] - (old_div(nx, 2) - old_div(nx_odd, 2)), 0) - xshift, \
                              scale_factor * round(coord[1] - (old_div(ny, 2) - old_div(ny_odd, 2)), 0) - yshift])
        coord_out_rot_h.append([scale_factor * round(coord[0] - (old_div(nx, 2) - old_div(nx_odd, 2)), 0), \
                                scale_factor * round(coord[1] - (old_div(ny, 2) - old_div(ny_odd, 2)), 0)])

    # M = cv2.getRotationMatrix2D( ( (data_square.shape[0]-1)/2, (data_square.shape[1]-1)/2 ), angle_selected,1)
    # RotData = cv2.warpAffine(data_square,M,data_square.shape,flags=cv2.INTER_NEAREST,borderValue=1)
    # ax = plt.subplot(121)
    # ax.imshow(data_square.T,origin='lower',interpolation='nearest')
    # ax = plt.subplot(122)
    # ax.imshow(RotData.T,origin='lower',interpolation='nearest')
    # patch = patches.Polygon(coord_out_rot_h, edgecolor='k', facecolor='None', linewidth=2)
    # ax.add_patch(patch)
    # plt.show()

    # coord for data_in
    # ----------------
    # print scale_factor, xshift, yshift
    # coord_out2 = []
    # for coord in coord_out:
    #    coord_out2.append([int(np.round(scale_factor*coord[0]-xshift,0)),int(np.round(scale_factor*coord[1]-yshift,0))])

    # print (datetime.datetime.now()-time_s).total_seconds()

    if flag_out is None:
        return coord_out
    elif flag_out == 'rotation':
        return coord_out, angle_selected, coord_out_rot
    else:
        print('bad def in findRotMaxRect input. stop')
        pdb.set_trace()


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

def calculate_area_center(points):
    # 解构四个点的坐标
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    area = abs(0.5 * ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (x2 * y1 + x3 * y2 + x4 * y3 + x1 * y4)))
    # side1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # side2 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    # ar = max(side1, side2) / min(side1, side2)
    # 计算横坐标平均值和纵坐标平均值
    x_avg = (x1 + x2 + x3 + x4) / 4
    y_avg = (y1 + y2 + y3 + y4) / 4

    return area, (round(x_avg), round(y_avg))

def calculate_new_coordinates(points, scale=0.3):
    # 计算原始四边形的面积
    original_area = calculate_area(points)

    # 计算新四边形的面积（原始面积的30%）
    new_area = scale * original_area
    if new_area < 4:
        return points

    # 计算面积比例因子
    scale_factor = math.sqrt(new_area / original_area)

    # 计算中心点坐标
    center_x = sum([x for x, _ in points]) / 4
    center_y = sum([y for _, y in points]) / 4

    # 计算顶点相对于中心点的偏移量并计算新四边形的顶点坐标
    new_points = []
    for x, y in points:
        offset_x = (x - center_x) * scale_factor
        offset_y = (y - center_y) * scale_factor
        new_x = center_x + offset_x
        new_y = center_y + offset_y
        new_points.append([new_x, new_y])

    return new_points


def draw_surface(label, kernal_size=100, scale=0.3):
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
                    tmp = np.where(regions == i, 0, 1)  # 满足大于0的值保留，不满足的设为1
                    rect_coord_ori, angle, coord_out_rot = findRotMaxRect(tmp,
                                                                          flag_opt=True,
                                                                          nbre_angle=25,
                                                                          flag_parallel=True,
                                                                          flag_out='rotation',
                                                                          flag_enlarge_img=False,
                                                                          limit_image_size=100)
                    if rect_coord_ori == None:
                        continue
                    for i in range(4):
                        rect_coord_ori[i][1] = tmp.shape[0] - rect_coord_ori[i][1]
                    rect_coord_ori = calculate_new_coordinates(rect_coord_ori, scale=scale)
                    pts = np.round(rect_coord_ori).astype(int)
                    pts = pts.reshape((-1, 2))
                    cv2.fillPoly(new_mask, [pts], (int(cls), int(cls), int(cls)))
                    cv2.fillPoly(new_mask_vis, [pts], tuple([int(c) for c in color]))

    new_mask = cv2.rotate(new_mask, cv2.ROTATE_90_CLOCKWISE)
    new_mask_vis = cv2.rotate(new_mask_vis, cv2.ROTATE_90_CLOCKWISE)
    return new_mask, new_mask_vis


def make(root_path):
    train_path = root_path + '/train'
    val_path = root_path + '/val'

    paths = [train_path, val_path]
    for path in paths:
        label_path = path + '/label'

        surface_label_path = path + '/largest_surface_label_test'
        surface_label_vis_path = path + '/largest_surface_label_vis_test'
        check_dir(surface_label_path), check_dir(surface_label_vis_path)

        list = os.listdir(label_path)
        for i in tqdm(list):
        # for i in list:
        #     print(i)
            label = os.path.join(label_path, i)
            label = read(label)
            new_mask, new_mask_vis = draw_surface(label)
            imsave(surface_label_path + '/' + i, new_mask)
            imsave(surface_label_vis_path + '/' + i, new_mask_vis)


if __name__ == '__main__':
    root_path = '/home/isalab206/Downloads/dataset/potsdam'
    make(root_path)
