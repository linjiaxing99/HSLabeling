import sys
sys.path.append('/home/isalab206/LJX/HSLabeling-master')
from utils import *
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from make_longest_line_label import find_longest_segment
from make_largest_surface_label import findRotMaxRect, calculate_area_center
from make_point_label_random import get_scattered_points
from scipy.ndimage import center_of_mass
from scipy.optimize import leastsq, curve_fit
import matplotlib.pyplot as plt
from utils import util
from scipy.optimize import fsolve
from sympy import symbols, exp
from scipy import ndimage
print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda:0")
print('Device:', device)

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

# def find_grids_with_mask(mask, grid_size):
#     indices = np.empty((2, 0), dtype=int)
#     # 遍历每个格子
#     for i in range(grid_size):
#         for j in range(grid_size):
#             # 计算当前格子的坐标范围
#             grid_left = i * 16
#             grid_right = (i + 1) * 16
#             grid_top = j * 16
#             grid_bottom = (j + 1) * 16
#
#             # 检查格子内是否有包含新掩码像素的像素
#             if np.any(mask[grid_left:grid_right, grid_top:grid_bottom]):
#                 indices = np.append(indices, [[i], [j]], axis=1)
#
#     return indices

def instance_score(out_seg, instance):
    # out_seg -> 1, k, h, w   instance -> np.uint8
    instance_seg = out_seg.cpu().numpy().squeeze(0) * instance[np.newaxis, ...]
    score = np.sum(instance_seg, axis=(-2, -1)) / np.sum(instance, axis=(-2, -1))
    max_score = np.max(score)
    cls = np.argmax(score)
    return max_score, cls


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


def sort_contours_by_hierarchy(contours, hierarchy, parent_index=-1):
    sorted_contours = []
    child_indices = [i for i, h in enumerate(hierarchy[0]) if h[3] == parent_index]

    for child_index in child_indices:
        child_contour = contours[child_index]
        sorted_contours.append(child_contour)

        # Recursive call to sort child's children contours
        sorted_contours.extend(sort_contours_by_hierarchy(contours, hierarchy, child_index))

    return sorted_contours


def Fun(x, a, b):  # 定义拟合函数形式
    return a * (1 - np.exp(-b * x))/(1 + np.exp(-b * x))


def curve_get(x, y):
    # param_bounds=(0, np.inf)#设定a和b的下界和上界。
    para, pcov = curve_fit(Fun, x, y, bounds=([0, 0], [np.inf, np.inf]), maxfev=100000)  # 进行拟合

    y_fitted = Fun(x, para[0], para[1])  # 画出拟合后的曲线
    plt.figure
    plt.plot(x, y, 'r', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()
    print(para)
    return para


def line_information(length, min_dist, max_dist, p):
    if length <= max_dist:
        slice_dist = length
        print('*')
    else:
        slice_dist = max_dist
        print('%')
    div, mod = divmod(length, slice_dist)
    information = div * Fun(slice_dist, p[0], p[1])
    slice_dist = slice_dist / 2
    while slice_dist > (min_dist):
        current_div, current_mod = divmod(length, slice_dist)
        # print(min_dist)
        information = information + (current_div - div) * Fun(slice_dist, p[0], p[1])
        div = current_div
        slice_dist = slice_dist / 2
    return information


def func(i, params):
    x = i[0]
    a = params
    # list_e1 = [math.exp(-x * ai) for ai in a[:-1]]
    list_e1 = [2 / (exp(-x * ai) + 1) - 1 for ai in a[:-1]]
    return [sum(list_e1) - a[-1]]


def line_weight(feat_shape, segmentation_mask, line_mask, overlope_indices, complexity):
    indices = np.nonzero(segmentation_mask)[0]
    indices = np.setdiff1d(indices, overlope_indices)
    line_indices = np.nonzero(line_mask)[0]
    line_indices = np.setdiff1d(line_indices, overlope_indices)
    if line_indices.size == 0 or indices.size == 0:
        return 0
    distance_mask = np.zeros((feat_shape, feat_shape), np.uint8)
    distance_mask.flat[indices] = 1
    tmp_center = center_of_mass(distance_mask)  # 计算几何中心
    center_x = round(tmp_center[0])
    center_y = round(tmp_center[1])
    x_coords = np.arange(feat_shape)
    y_coords = np.arange(feat_shape).reshape(-1, 1)
    distance = np.sqrt((center_x - x_coords) ** 2 + (center_y - y_coords) ** 2)
    distance[distance_mask == 0] = 0
    distance_values = distance.flatten()[indices]
    distance_values = np.floor(distance_values)
    unique_distance = np.unique(distance_values)
    distance_values = np.searchsorted(unique_distance, distance_values)
    distance = np.zeros_like(distance_values, np.float32)
    C = max(distance_values)
    L = round(line_indices.size * complexity)
    if 2 * C + 2 <= L:
        L = 2 * C
    for Li in range(int(C) + 1):
        if Li == 0:
            distance[distance_values == Li] = min(L / (2 * C - L + 2), 1.0) / float(np.sum(distance_values == Li))
        elif Li <= max(L - 1, 2 * C - L + 1) - C:
            distance[distance_values == Li] = min((2 * L) / (2 * C - L + 2), 2.0) / float(np.sum(distance_values == Li))
        else:
            distance[distance_values == Li] = min((2 * (C - Li + 1)) / (2 * C - L + 2), 2.0) / float(np.sum(distance_values == Li))
    return distance


def surface_weight(feat_shape, segmentation_mask, surface_mask, overlope_indices, complexity):
    indices = np.nonzero(segmentation_mask)[0]
    indices = np.setdiff1d(indices, overlope_indices)
    surface_indices = np.nonzero(surface_mask)[0]
    surface_indices = np.setdiff1d(surface_indices, overlope_indices)
    if surface_indices.size == 0:
        return 0
    distance_mask = np.zeros((feat_shape, feat_shape), np.uint8)
    distance_mask.flat[indices] = 1
    distance_mask = ndimage.distance_transform_edt(distance_mask)
    distance_values = distance_mask.flatten()[indices]
    distance_values = np.append(distance_values, surface_indices.size * complexity)
    r = fsolve(func, x0=[1], args=distance_values)
    z = symbols('z')
    # func2 = exp(-r[0] * z)
    func2 = 2 / (exp(-r[0] * z) + 1) - 1
    # distance.flat[indices] = np.array([func2.subs(z, val) for val in distance.flat[indices]])
    # distance = distance.flatten()
    distance = np.array([float(func2.subs(z, val)) for val in distance_mask.flatten()[indices]])
    return distance


def main():
    opt = Point_Options().parse()
    log_path, checkpoint_path, predict_path, _, _ = create_save_path(opt)
    train_txt_path, _, _ = create_data_path(opt)

    # load train and val dataset
    train_dataset = Dataset_point(opt, train_txt_path, flag='train', transform=None)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    max_instance = 35
    feat_shape = 32
    sam_path = '/mnt/vdf/isalab205/LJX_data/dataset/loveda/train/sam_pkl/'
    shape_path = '/mnt/vdf/isalab205/LJX_data/dataset/loveda/train/sam_shape/'
    path = '/home/isalab206/Downloads/dataset/loveda/train/'
    for batch_idx, batch in enumerate(tqdm(loader)):
        img, _, _, filename = batch
        filename = filename[0]
        b, c, h, w = img.shape
        f = open(os.path.join(sam_path, (filename[:-4] + '.pkl')), 'rb')
        sam_pkl = pickle.loads(f.read())
        f.close()

        instance_mask = np.ones((h, w)) * 255
        instance_num = 0
        for childs in sam_pkl.children:
            M = childs.value['segmentation']
            instance_mask[M] = instance_num
            instance_num = instance_num + 1

        # split instance
        split_instance = []
        instance_mask_current = np.ones((h, w), np.uint8) * 255
        for i in range(instance_num):
            tmp_mask = np.where(instance_mask == i, 1, 0)
            area = np.sum(tmp_mask)
            if area > 200:
                split_instance.append({'segmentation': tmp_mask, 'area': area})
                instance_mask_current[tmp_mask.astype(np.bool_)] = int(len(split_instance) - 1)
        split_instance = split_instance[:max_instance]

        # time_start = time.time()
        downsampled = int(h / feat_shape)
        # overlope class
        max_budget = 5 * len(split_instance)
        truelabel = util.read(os.path.join(path + 'label', filename))
        truelabel = truelabel[:, :, 0]
        overlope_mask = np.zeros((feat_shape, feat_shape), np.uint8)
        for i in range(feat_shape):
            for j in range(feat_shape):
                if truelabel[int(i * downsampled + downsampled / 2), int(j * downsampled + downsampled / 2)] >= opt.num_classes:
                    overlope_mask[i, j] = 1
        overlope_mask = overlope_mask.ravel()
        overlope_indices = np.where(overlope_mask == 1)[0]
        if len(overlope_mask) - len(overlope_indices) < 2 * max_budget:
            f = open(shape_path + filename[:-4] + '.pkl', 'wb')
            content = pickle.dumps(split_instance)
            f.write(content)
            f.close()

        for i in range(len(split_instance)):
            tmp_mask = split_instance[i]['segmentation']
            # fft measure
            fft = 1 - fft_measure(tmp_mask)
            # largest surface and longest line
            segmentation_mask = cv2.resize(tmp_mask.astype(np.float_), (feat_shape, feat_shape), interpolation=cv2.INTER_AREA)
            segmentation_surface_mask = np.where(segmentation_mask >= 0.5, 1, 0)
            segmentation_line_mask = np.where(segmentation_mask >= 0.5, 1, 0)
            line_length, line_point = find_longest_segment(tmp_mask)
            tmp_mask_flipped = ~tmp_mask.astype(np.bool_)
            rect_coord_ori, angle, coord_out_rot = findRotMaxRect(tmp_mask_flipped.astype(np.uint8),
                                                                  flag_opt=True,
                                                                  nbre_angle=25,
                                                                  flag_parallel=True,
                                                                  flag_out='rotation',
                                                                  flag_enlarge_img=False,
                                                                  limit_image_size=100)

            # surface_mask = np.zeros((feat.shape[2], feat.shape[3]), np.uint8)
            surface_mask = np.zeros((h, w), np.uint8)
            line_mask = np.zeros((feat_shape, feat_shape), np.uint8)
            if rect_coord_ori != None:
                surface_area, center = calculate_area_center(rect_coord_ori)
                split_instance[i]['center'] = center
                for j in range(4):
                    rect_coord_ori[j][1] = h - rect_coord_ori[j][1]
                for j in range(4):
                    k = rect_coord_ori[j][1]
                    # rect_coord_ori[j][1] = rect_coord_ori[j][0] / downsampled
                    # rect_coord_ori[j][0] = (h - k) / downsampled
                    rect_coord_ori[j][1] = rect_coord_ori[j][0]
                    rect_coord_ori[j][0] = (h - k)
                pts = np.round(rect_coord_ori).astype(int)
                pts = pts.reshape((-1, 2))
                cv2.fillPoly(surface_mask, [pts], 1)
                surface_mask = cv2.resize(surface_mask, (feat_shape, feat_shape), interpolation=cv2.INTER_AREA)
                cv2.line(line_mask, (int(line_point[0].x / downsampled), int(line_point[0].y / downsampled)),
                         (int(line_point[1].x / downsampled), int(line_point[1].y / downsampled)), color=1,
                         thickness=1)
                segmentation_line_mask = np.logical_or(segmentation_line_mask, line_mask)
                # cv2.line(line_mask, (5, 16), (16, 16), color=1, thickness=1)
                split_instance[i]['surface_mask'] = surface_mask.ravel()
                split_instance[i]['line_mask'] = line_mask.ravel()
                split_instance[i]['length'] = line_length
                split_instance[i]['segmentation_surface_mask'] = segmentation_surface_mask.ravel()
                split_instance[i]['segmentation_line_mask'] = segmentation_line_mask.ravel()
                split_instance[i]['complexity'] = fft
                split_instance[i]['surface_pts'] = pts
                split_instance[i]['line_point'] = line_point
            else:
                center = get_scattered_points(tmp_mask)[0]
                split_instance[i]['center'] = center
                cv2.line(line_mask, (int(line_point[0].x / downsampled), int(line_point[0].y / downsampled)),
                         (int(line_point[1].x / downsampled), int(line_point[1].y / downsampled)), color=1,
                         thickness=1)
                split_instance[i]['line_mask'] = line_mask.ravel()
                split_instance[i]['length'] = line_length
                split_instance[i]['surface_mask'] = None
                split_instance[i]['segmentation_surface_mask'] = segmentation_surface_mask.ravel()
                split_instance[i]['segmentation_line_mask'] = segmentation_line_mask.ravel()
                split_instance[i]['complexity'] = fft
                split_instance[i]['line_point'] = line_point
            # probabilistic weight
            split_instance[i]['line_distance'] = line_weight(feat_shape, segmentation_line_mask.ravel(), line_mask.ravel(), overlope_indices, fft)
            split_instance[i]['surface_distance'] = surface_weight(feat_shape, segmentation_surface_mask.ravel(), surface_mask.ravel(), overlope_indices, fft)


        f = open(shape_path + filename[:-4] + '.pkl', 'wb')
        content = pickle.dumps(split_instance)
        f.write(content)
        f.close()



if __name__ == '__main__':
    main()

