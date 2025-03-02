import sys
import torch
sys.path.append('/home/isalab206/LJX/HSLabeling')
from utils import *
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from scipy import ndimage
from DataProcess.make_longest_line_label import find_longest_segment
from DataProcess.make_largest_surface_label import findRotMaxRect, calculate_area_center, calculate_area
from DataProcess.make_point_label_random import get_scattered_points, read_cls_color
from scipy.ndimage import center_of_mass
from scipy.optimize import leastsq, curve_fit
import matplotlib.pyplot as plt
from kneed import KneeLocator
from utils import util
from scipy.optimize import fsolve
from sympy import symbols, exp
import math
from einops import rearrange, repeat
from scipy import ndimage
# print("PyTorch Version: ", torch.__version__)
# print('cuda', torch.version.cuda)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device("cuda:0")
# print('Device:', device)

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def find_grids_with_mask(mask, grid_size):
    indices = np.empty((2, 0), dtype=int)
    # 遍历每个格子
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算当前格子的坐标范围
            grid_left = i * 16
            grid_right = (i + 1) * 16
            grid_top = j * 16
            grid_bottom = (j + 1) * 16

            # 检查格子内是否有包含新掩码像素的像素
            if np.any(mask[grid_left:grid_right, grid_top:grid_bottom]):
                indices = np.append(indices, [[i], [j]], axis=1)

    return indices

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


def generate(opt, current_budget_part, budget_part=2000):
    device = torch.device("cuda:0")

    log_path, checkpoint_path, predict_path, _, _ = create_save_path(opt)
    # train_txt_path, _, _ = create_data_path(opt)
    data_inform_path = os.path.join(opt.data_inform_path, opt.dataset)
    train_txt_path = os.path.join(data_inform_path, 'all_images.txt')

    # load train and val dataset
    train_dataset = Dataset_point(opt, train_txt_path, flag='train', transform=None)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers,
                        pin_memory=opt.pin)

    model = Seg_Net(opt)
    if os.path.exists(checkpoint_path) and os.listdir(checkpoint_path):
        checkpoint = torch.load(checkpoint_path + '/model_best_' + str(int(current_budget_part)) + '.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('resume success')
    model = model.to(device)
    model.eval()

    in_channels = 512
    num_prototype = 10
    feat_shape = 32
    surface_cost = 5
    line_cost = 3
    budget = 0
    surface_budget = 0
    line_budget = 0
    point_budget = 0
    shape_path = '/mnt/vdf/isalab205/LJX_data/dataset/loveda/train/sam_shape/'
    # shape_path = '/mnt/vdf/isalab205/LJX_data/dataset/potsdam/train/sam_shape2/'
    # path = '/home/isalab206/Downloads/dataset/potsdam/train/'
    path = os.path.join(opt.data_root, opt.dataset, 'train')
    # prototypes = nn.Parameter(torch.zeros(opt.num_classes, num_prototype, in_channels), requires_grad=False).to(device)
    # prototypes = trunc_normal_(prototypes, std=0.02)
    acc = []
    # feat_norm = nn.LayerNorm(in_channels)
    # mask_norm = nn.LayerNorm(opt.num_classes)
    # labeled_path = path + '/' + 'al_corr5_8000/'
    label_path = path + '/' + 'al_' + opt.data_label + '/'
    label_vis_path = path + '/' + 'al_vis_' + opt.data_label + '/'
    surface_weight_path = path + '/' + 'surface_weight_' + opt.data_label + '/'
    line_weight_path = path + '/' + 'line_weight_' + opt.data_label + '/'
    point_weight_path = path + '/' + 'point_weight_' + opt.data_label + '/'
    check_dir(label_path), check_dir(label_vis_path), check_dir(surface_weight_path)
    check_dir(line_weight_path), check_dir(point_weight_path)
    labeled_list = os.listdir(label_path)
    for batch_idx, batch in enumerate(tqdm(loader)):
        img, _, _, filename = batch
        filename = filename[0]
        if filename in labeled_list:
            continue
        print(budget)
        print(filename)
        keyb = 0
        b, c, h, w = img.shape
        f = open(os.path.join(shape_path, (filename[:-4] + '.pkl')), 'rb')
        split_instance = pickle.loads(f.read())
        f.close()
        with torch.no_grad():
            input = img.cuda(non_blocking=True)
            output, feat = model(input)
            torch.cuda.synchronize()
        _c = rearrange(feat, 'b c h w -> (b h w) c')
        # _c = feat_norm(feat_)
        _c = F.normalize(_c, p=2, dim=-1)
        # prototypes = prototypes.data.copy_(F.normalize(prototypes, p=2, dim=-1))
        downsampled = int(h / feat_shape)

        if os.path.exists(checkpoint_path) and os.listdir(checkpoint_path):
            output = output.view(b, h, w, opt.num_classes)
            probs = torch.softmax(output, dim=-1)
            # log_probs = torch.log2(probs.clamp(min=1e-10))
            # entropy = -(probs * log_probs).sum(dim=-1)
            # entropy = entropy / np.log2(opt.num_classes)  # b, h, w 将熵的范围压缩到0-1之间
            # entropy = F.avg_pool2d(entropy, kernel_size=feat_shape, stride=feat_shape)
            # entropy = entropy.view(feat_shape * feat_shape)

            top2 = torch.topk(probs, k=2, dim=-1).values # b x h x w x k
            entropy = torch.squeeze(1 - (top2[:, :, :, 0] - top2[:, :, :, 1]).abs(), -1)  # b x h x w x 1
            entropy = F.avg_pool2d(entropy, kernel_size=downsampled, stride=downsampled)
            entropy = entropy.view(feat_shape * feat_shape)

            # entropy, _ = torch.max(probs, dim=-1, keepdim=True)
            # entropy = torch.squeeze((1.0 - entropy) * (opt.num_classes / (opt.num_classes - 1)), -1)
            # # print(entropy.shape)
            # # entropy = entropy.permute(2, 0, 1)
            # # entropy = (1.0 - probs.max(dim=-1)[0]) * (opt.num_classes / (opt.num_classes - 1))  # b x h x w
            # entropy = F.avg_pool2d(entropy, kernel_size=feat_shape, stride=feat_shape)
            # entropy = entropy.view(feat_shape * feat_shape)
        else:
            entropy = torch.ones(feat_shape * feat_shape).to(device)

        # split instance
        instance_mask_current = np.ones((h, w), np.uint8) * 255
        for i in range(len(split_instance)):
            tmp_mask = split_instance[i]['segmentation']
            instance_mask_current[tmp_mask.astype(np.bool_)] = i

        # time_start = time.time()
        max_budget = 5 * len(split_instance)

        # overlope class
        truelabel = util.read(os.path.join(path + '/label', filename))
        truelabel = truelabel[:, :, 0]
        overlope_mask = np.zeros((feat.shape[2], feat.shape[3]), np.uint8)
        for i in range(feat.shape[2]):
            for j in range(feat.shape[3]):
                if truelabel[int(i * downsampled + downsampled / 2), int(j * downsampled + downsampled / 2)] >= opt.num_classes:
                    overlope_mask[i, j] = 1
        overlope_mask = overlope_mask.ravel()
        overlope_indices = np.where(overlope_mask == 1)[0]
        if len(overlope_mask) - len(overlope_indices) < 2 * max_budget:
            continue

        # complexity estimation
        corr = _c.matmul(_c.permute(1, 0))  # (b h w) c, c (b h w)
        corr = 1 - corr  # 对角线值为0
        corr[overlope_indices, :] = 0
        corr[:, overlope_indices] = 0
        corr = corr * entropy

        mean_values = torch.mean(corr, dim=1)
        # mean_values = mean_values * entropy
        point_max, point_index = torch.max(mean_values, dim=0)
        # point_max, point_index = torch.max(entropy, dim=0)
        div, mod = divmod(point_index.item(), feat_shape)
        x = int(div * downsampled + downsampled / 2)
        y = int(mod * downsampled + downsampled / 2)
        indicator = (x, y)
        label_first = [{'type': 'point', 'vector': [point_index.item()], 'instance': instance_mask_current[x, y],
                  'information': entropy[point_index].item(), 'indicator': indicator}]


        tmp_budget = 1
        budget_label = {}
        budget_label[0] = None
        budget_label[tmp_budget] = label_first
        while tmp_budget < max_budget:
            label = budget_label[tmp_budget]
            label_vector = [item for sublist in [x['vector'] for x in label] for item in sublist]
            label_vector = list(set(label_vector))
            tmp_corr = corr[label_vector, :]
            min_values, _ = torch.min(tmp_corr, dim=0)
            point_max, point_index = torch.max(min_values, dim=0)
            information = point_max.item() + budget_label[tmp_budget][-1]['information']
            div, mod = divmod(point_index.item(), feat_shape)
            x = int(div * downsampled + downsampled / 2)
            y = int(mod * downsampled + downsampled / 2)
            indicator = (x, y)
            label = budget_label[tmp_budget] + [{'type': 'point', 'vector': [point_index.item()],
                                                 'instance': instance_mask_current[x, y], 'information': information,
                                                 'indicator': indicator}]
            tmp_budget = tmp_budget + 1
            budget_label[tmp_budget] = label
        x = []
        y = []
        for key in budget_label:
            if budget_label[key] != None:
                x.append(key)
                y.append(budget_label[key][-1]['information'])
        x = np.array(x)
        y = np.array(y)
        kl = KneeLocator(x, y, curve="concave", direction="increasing", S=1)
        knee = kl.elbow
        # kl.plot_knee()
        if knee == None:
            continue
        label_list = budget_label[knee]
        min_information = label_list[-1]['information'] - label_list[-2]['information']


        tmp_budget = 1
        budget_label = {}
        budget_label[0] = None
        budget_label[tmp_budget] = label_first
        while tmp_budget < max_budget:
            label = budget_label[tmp_budget]
            label_vector = [item for sublist in [x['vector'] for x in label] for item in sublist]
            label_vector = list(set(label_vector))
            tmp_corr = corr[label_vector, :]
            min_values, _ = torch.min(tmp_corr, dim=0)
            point_max, point_index = torch.max(min_values, dim=0)
            information = point_max.item() + budget_label[tmp_budget][-1]['information']
            div, mod = divmod(point_index.item(), feat_shape)
            x = int(div * downsampled + downsampled / 2)
            y = int(mod * downsampled + downsampled / 2)
            indicator = (x, y)
            label = budget_label[tmp_budget] + [{'type': 'point', 'vector': [point_index.item()],
            'instance': instance_mask_current[x, y], 'information': information, 'indicator': indicator}]
            tmp_budget = tmp_budget + 1
            budget_label[tmp_budget] = label

            if tmp_budget >= line_cost:
                if budget_label[tmp_budget - line_cost] == None:
                    for i in range(len(split_instance)):
                        line_mask = split_instance[i]['line_mask']
                        length = split_instance[i]['length']
                        segmentation_mask = split_instance[i]['segmentation_line_mask']
                        complexity = split_instance[i]['complexity']
                        distance = split_instance[i]['line_distance']
                        indices = np.nonzero(segmentation_mask)[0]
                        indices = np.setdiff1d(indices, overlope_indices)
                        line_indices = np.nonzero(line_mask)[0]
                        line_indices = np.setdiff1d(line_indices, overlope_indices)
                        if line_indices.size == 0 or indices.size == 0:
                            continue
                        if isinstance(distance, int):
                            distance = np.array([0])
                        if distance.size < indices.size:
                            distance = np.pad(distance, (0, indices.size - distance.size), 'constant', constant_values=0)
                        elif distance.size > indices.size:
                            distance = distance[:indices.size]
                        distance = torch.from_numpy(distance)
                        distance = distance.to('cuda')
                        tmp_corr = corr[indices][:, indices]
                        tmp_corr = tmp_corr * distance
                        tmp_entropy = entropy[indices]
                        mean_values = torch.mean(tmp_corr, dim=1)
                        mean_values = mean_values * tmp_entropy
                        point_max, point_index = torch.max(mean_values, dim=0)
                        information = tmp_entropy[point_index].item()
                        # tmp_entropy = entropy[indices]
                        # point_max, point_index = torch.max(tmp_entropy, dim=0)
                        # information = point_max.item()
                        vector = [point_index.item()]
                        div, mod = divmod(indices[point_index.item()], feat_shape)
                        x = int(div * downsampled + downsampled / 2)
                        y = int(mod * downsampled + downsampled / 2)
                        indicator = (x, y)
                        if truelabel[x, y] >= opt.num_classes:
                            continue
                        for k in range(len(indices) - 1):
                            tmp = tmp_corr[vector, :]
                            min_values, _ = torch.min(tmp, dim=0)
                            point_max, point_index = torch.max(min_values, dim=0)
                            vector.append(point_index.item())
                            information = information + point_max.item()
                        # information = information * complexity
                        if information >= budget_label[tmp_budget][-1]['information']:
                            budget_label[tmp_budget] = [{'type': 'line', 'vector': indices, 'instance': i,
                                                         'information': information, 'indicator': indicator}]
                else:
                    label = budget_label[tmp_budget - line_cost]
                    instance_list = [x['instance'] for x in label]
                    for i in range(len(split_instance)):
                        if i in instance_list:
                            continue
                        label_vector = [item for sublist in [x['vector'] for x in label] for item in sublist]
                        label_vector = list(set(label_vector))
                        information = label[-1]['information']
                        line_mask = split_instance[i]['line_mask']
                        length = split_instance[i]['length']
                        segmentation_mask = split_instance[i]['segmentation_line_mask']
                        complexity = split_instance[i]['complexity']
                        distance = split_instance[i]['line_distance']
                        indices = np.nonzero(segmentation_mask)[0]
                        indices = np.setdiff1d(indices, overlope_indices)
                        line_indices = np.nonzero(line_mask)[0]
                        line_indices = np.setdiff1d(line_indices, overlope_indices)
                        if line_indices.size == 0 or indices.size == 0:
                            continue
                        if isinstance(distance, int):
                            distance = np.array([0])
                        if distance.size < indices.size:
                            distance = np.pad(distance, (0, indices.size - distance.size), 'constant', constant_values=0)
                        elif distance.size > indices.size:
                            distance = distance[:indices.size]
                        distance = torch.from_numpy(distance)
                        distance = distance.to('cuda')

                        for k in range(len(indices)):
                            tmp_corr = corr[label_vector][:, indices]
                            tmp_corr = tmp_corr * distance
                            min_values, _ = torch.min(tmp_corr, dim=0)
                            point_max, point_index = torch.max(min_values, dim=0)
                            label_vector.append(indices[point_index.item()])
                            information = information + point_max.item()
                            if k == 0:
                                div, mod = divmod(indices[point_index.item()], feat_shape)
                                x = int(div * downsampled + downsampled / 2)
                                y = int(mod * downsampled + downsampled / 2)
                                indicator = (x, y)
                        # information = information * complexity
                        if truelabel[x, y] >= opt.num_classes:
                            continue
                        if information >= budget_label[tmp_budget][-1]['information']:
                            budget_label[tmp_budget] = budget_label[tmp_budget - line_cost] + [{'type': 'line', 'vector': indices, 'instance': i, 'information': information, 'indicator': indicator}]

            if tmp_budget >= surface_cost:
                if budget_label[tmp_budget - surface_cost] == None:
                    for i in range(len(split_instance)):
                        surface_mask = split_instance[i]['surface_mask']
                        segmentation_mask = split_instance[i]['segmentation_surface_mask']
                        complexity = split_instance[i]['complexity']
                        distance = split_instance[i]['surface_distance']
                        if not np.any(surface_mask):
                            continue
                        indices = np.nonzero(segmentation_mask)[0]
                        indices = np.setdiff1d(indices, overlope_indices)
                        surface_indices = np.nonzero(surface_mask)[0]
                        surface_indices = np.setdiff1d(surface_indices, overlope_indices)
                        if surface_indices.size == 0:
                            continue
                        if isinstance(distance, int):
                            distance = np.array([0])
                        if distance.size < indices.size:
                            distance = np.pad(distance, (0, indices.size - distance.size), 'constant', constant_values=0)
                        elif distance.size > indices.size:
                            distance = distance[:indices.size]
                        distance = torch.from_numpy(distance)
                        distance = distance.to('cuda')
                        tmp_corr = corr[indices][:, indices]
                        tmp_corr = tmp_corr * distance
                        tmp_entropy = entropy[indices]
                        mean_values = torch.mean(tmp_corr, dim=1)
                        mean_values = mean_values * tmp_entropy
                        point_max, point_index = torch.max(mean_values, dim=0)
                        information = tmp_entropy[point_index].item()
                        # tmp_entropy = entropy[indices]
                        # point_max, point_index = torch.max(tmp_entropy, dim=0)
                        # information = point_max.item()
                        vector = [point_index.item()]
                        div, mod = divmod(indices[point_index.item()], feat_shape)
                        x = int(div * downsampled + downsampled / 2)
                        y = int(mod * downsampled + downsampled / 2)
                        indicator = (x, y)
                        if truelabel[x, y] >= opt.num_classes:
                            continue
                        for k in range(len(indices) - 1):
                            tmp = tmp_corr[vector, :]
                            min_values, _ = torch.min(tmp, dim=0)
                            point_max, point_index = torch.max(min_values, dim=0)
                            vector.append(point_index.item())
                            information = information + point_max.item()
                        # information = information * (surface_indices.size / indices.size) * complexity
                        # information = information * complexity
                        # print(information)
                        if information >= budget_label[tmp_budget][-1]['information']:
                            budget_label[tmp_budget] = [{'type': 'surface', 'vector': indices, 'instance': i,
                                                         'information': information, 'indicator': indicator}]
                else:
                    label = budget_label[tmp_budget - surface_cost]
                    instance_list = [x['instance'] for x in label]
                    for i in range(len(split_instance)):
                        if i in instance_list:
                            continue
                        label_vector = [item for sublist in [x['vector'] for x in label] for item in sublist]
                        label_vector = list(set(label_vector))
                        information = label[-1]['information']
                        surface_mask = split_instance[i]['surface_mask']
                        segmentation_mask = split_instance[i]['segmentation_surface_mask']
                        complexity = split_instance[i]['complexity']
                        distance = split_instance[i]['surface_distance']
                        if not np.any(surface_mask):
                            continue
                        indices = np.nonzero(segmentation_mask)[0]
                        indices = np.setdiff1d(indices, overlope_indices)
                        surface_indices = np.nonzero(surface_mask)[0]
                        surface_indices = np.setdiff1d(surface_indices, overlope_indices)
                        if surface_indices.size == 0:
                            continue
                        if isinstance(distance, int):
                            distance = np.array([0])
                        if distance.size < indices.size:
                            distance = np.pad(distance, (0, indices.size - distance.size), 'constant', constant_values=0)
                        elif distance.size > indices.size:
                            distance = distance[:indices.size]
                        distance = torch.from_numpy(distance)
                        distance = distance.to('cuda')

                        for k in range(len(indices)):
                            tmp_corr = corr[label_vector][:, indices]
                            tmp_corr = tmp_corr * distance
                            min_values, _ = torch.min(tmp_corr, dim=0)
                            point_max, point_index = torch.max(min_values, dim=0)
                            label_vector.append(indices[point_index.item()])
                            information = information + point_max.item()
                            if k == 0:
                                div, mod = divmod(indices[point_index.item()], feat_shape)
                                x = int(div * downsampled + downsampled / 2)
                                y = int(mod * downsampled + downsampled / 2)
                                indicator = (x, y)
                        if truelabel[x, y] >= opt.num_classes:
                            continue
                        # information = label[-1]['information'] + (information - label[-1]['information']) * (surface_indices.size / indices.size) * complexity
                        if information >= budget_label[tmp_budget][-1]['information']:
                            budget_label[tmp_budget] = budget_label[tmp_budget - surface_cost] + [{'type': 'surface',
                              'vector': indices, 'instance': i, 'information': information, 'indicator': indicator}]

        # end_time = time.time()
        # time_cost = end_time - time_start
        # print("complexity time", time_cost)
        x = []
        y = []
        for key in budget_label:
            if budget_label[key] != None:
                x.append(key)
                y.append(budget_label[key][-1]['information'])
        x = np.array(x)
        y = np.array(y)
        kl = KneeLocator(x, y, curve="concave", direction="increasing", S=0.5)
        knee = kl.elbow
        kl.plot_knee()
        if knee == None:
            continue
        # knee = min(knee, 10)
        # max_budget = 10
        label_list = budget_label[knee]

        # label surface、line
        truelabel = util.read(os.path.join(path + '/label', filename))
        labeled_mask = np.ones([h, w, c], np.uint8) * 255
        new_mask = np.ones([h, w, c], np.uint8) * 255
        new_mask_vis = np.zeros([h, w, c], np.uint8)
        point_weight = np.zeros([h, w], np.float32)
        line_weight = np.zeros([h, w], np.float32)
        surface_weight = np.zeros([h, w], np.float32)
        label_instance = [d for d in label_list if d.get('type') in ['surface', 'line']]
        if label_instance != []:
            label_instance = sorted(label_instance, key=lambda x: x['type'] == 'surface', reverse=True)
            tmp_budget = 0
            instance_vector = []
            for i in range(len(label_instance)):
                type = label_instance[i]['type']
                indicator = label_instance[i]['indicator']
                ix, iy = indicator
                cls = truelabel[ix, iy, 0]
                # color = read_cls_color(cls)
                if labeled_mask[ix, iy][0] != 255:
                    type = 'point'
                if type == 'surface' or type == 'line':
                    cls_mask = np.where(truelabel[:, :, 0] == cls, 1, 0)
                    cls_mask = np.asarray(cls_mask, dtype=np.uint8)
                    num_objects, regions, stats, centroids = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
                    region = regions[ix, iy]
                    label_indices = np.where(regions == region)
                    if type == 'surface':
                        type_label = util.read(os.path.join(path + '/largest_surface_label_random_complexity', filename))
                        type_label_vis = util.read(os.path.join(path + '/largest_surface_label_vis_random_complexity', filename))
                        surface_mask = np.ones([h, w], np.uint8) * 255
                        surface_mask[label_indices] = type_label[:, :, 0][label_indices]
                        surface_mask = np.where(surface_mask == cls, 1, 0)
                        indices = np.where(surface_mask == 1)
                        surface_mask = cv2.resize(surface_mask.astype('uint8'), (feat.shape[2], feat.shape[3]), interpolation=cv2.INTER_AREA)
                        surface_mask = surface_mask.ravel()
                        surface_indices = np.nonzero(surface_mask)[0]
                        if surface_indices.size == 0:
                            continue
                        surface_indices = np.setdiff1d(surface_indices, overlope_indices)
                        instance_vector = instance_vector + list(surface_indices)
                        budget = budget + surface_cost
                        surface_weight[indices] = 1
                        surface_budget = surface_budget + surface_cost
                        tmp_budget = tmp_budget + surface_cost
                    else:
                        type_label = util.read(os.path.join(path + '/longest_line_label_random_complexity', filename))
                        type_label_vis = util.read(os.path.join(path + '/longest_line_label_vis_random_complexity', filename))
                        line_mask = np.ones([h, w], np.uint8) * 255
                        line_mask[label_indices] = type_label[:, :, 0][label_indices]
                        line_mask = np.where(line_mask == cls, 1, 0)
                        indices = np.where(line_mask == 1)
                        if len(indices[0]) == 0:
                            continue
                        leftmost_index = np.min(indices[1])
                        leftmost_coordinate = (leftmost_index, indices[0][np.where(indices[1] == leftmost_index)[0][0]])
                        rightmost_index = np.max(indices[1])
                        rightmost_coordinate = (rightmost_index, indices[0][np.where(indices[1] == rightmost_index)[0][0]])
                        upmost_index = np.min(indices[0])
                        upmost_coordinate = (indices[1][np.where(indices[0] == upmost_index)[0][0]], upmost_index)
                        downmost_index = np.max(indices[0])
                        downmost_coordinate = (indices[1][np.where(indices[0] == downmost_index)[0][0]], downmost_index)
                        long1 = math.sqrt((leftmost_coordinate[0] - rightmost_coordinate[0]) ** 2 + (leftmost_coordinate[1] - rightmost_coordinate[1]) ** 2)
                        long2 = math.sqrt((upmost_coordinate[0] - downmost_coordinate[0]) ** 2 + (upmost_coordinate[1] - downmost_coordinate[1]) ** 2)
                        if long1 >= long2:
                            line_point1 = leftmost_coordinate
                            line_point2 = rightmost_coordinate
                        else:
                            line_point1 = upmost_coordinate
                            line_point2 = downmost_coordinate
                        line_mask = np.zeros((feat.shape[2], feat.shape[3]), np.uint8)
                        cv2.line(line_mask,
                                 (int(line_point1[0] / downsampled), int(line_point1[1] / downsampled)),
                                 (int(line_point2[0] / downsampled), int(line_point2[1] / downsampled)), color=1,
                                 thickness=1)
                        line_mask = line_mask.ravel()
                        line_indices = np.nonzero(line_mask)[0]
                        line_indices = np.setdiff1d(line_indices, overlope_indices)
                        instance_vector = instance_vector + list(line_indices)
                        budget = budget + line_cost
                        line_weight[indices] = 1
                        line_budget = line_budget + line_cost
                        tmp_budget = tmp_budget + line_cost
                    new_mask[indices] = type_label[indices]
                    new_mask_vis[indices] = type_label_vis[indices]
                    labeled_mask[label_indices] = truelabel[label_indices]
                # else:
                #     new_mask[ix:ix + 3, iy:iy + 3, :] = (cls, cls, cls)
                #     new_mask_vis[ix:ix + 3, iy:iy + 3, :] = color
                #     point_weight[ix:ix + 3, iy:iy + 3] = 1
                #     point_budget = point_budget + 1
                #     budget = budget + 1
                #     tmp_budget = tmp_budget + 1
                #     instance_vector.append(int(ix / downsampled) * feat.shape[2] + int(iy / downsampled))

            instance_vector = list(set(instance_vector))
            if len(instance_vector) == 0:
                keyb = 1
            else:
                tmp_corr = corr[instance_vector][:, instance_vector]
                # sum_values = torch.sum(tmp_corr, dim=1)
                # point_max, point_index = torch.max(sum_values, dim=0)
                # information = 0
                tmp_entropy = entropy[instance_vector]
                mean_values = torch.mean(tmp_corr, dim=1)
                mean_values = mean_values * tmp_entropy
                point_max, point_index = torch.max(mean_values, dim=0)
                information = tmp_entropy[point_index].item()
                information_first = tmp_entropy[point_index].item()
                # tmp_entropy = entropy[instance_vector]
                # point_max, point_index = torch.max(tmp_entropy, dim=0)
                # information = point_max.item()
                # information_first = point_max.item()
                vector = [point_index.item()]
                for k in range(len(instance_vector) - 1):
                    tmp = tmp_corr[vector, :]
                    min_values, _ = torch.min(tmp, dim=0)
                    point_max, point_index = torch.max(min_values, dim=0)
                    vector.append(point_index.item())
                    information = information + point_max.item()
                budget_label = {}
                budget_label[1] = [{'type': 'surface', 'vector': instance_vector, 'instance': 0, 'information': information_first, 'indicator': 0}]
                budget_label[tmp_budget] = [{'type': 'surface', 'vector': instance_vector, 'instance': 0, 'information': information, 'indicator': 0}]

                while tmp_budget < max_budget:
                    label = budget_label[tmp_budget]
                    label_vector = [item for sublist in [x['vector'] for x in label] for item in sublist]
                    label_vector = list(set(label_vector))
                    tmp_corr = corr[label_vector, :]
                    min_values, _ = torch.min(tmp_corr, dim=0)
                    point_max, point_index = torch.max(min_values, dim=0)
                    information = point_max.item() + budget_label[tmp_budget][-1]['information']
                    div, mod = divmod(point_index.item(), feat_shape)
                    x = int(div * downsampled + downsampled / 2)
                    y = int(mod * downsampled + downsampled / 2)
                    indicator = (x, y)
                    cls = truelabel[x, y, 0]
                    if cls >= opt.num_classes:
                        corr[point_index.item(), :] = 0
                        corr[:, point_index.item()] = 0
                        continue
                    label = budget_label[tmp_budget] + [{'type': 'point', 'vector': [point_index.item()],
                                                         'instance': instance_mask_current[x, y],
                                                         'information': information, 'indicator': indicator}]
                    tmp_budget = tmp_budget + 1
                    budget_label[tmp_budget] = label

                # x = []
                # y = []
                # for j, (key, value) in enumerate(budget_label.items()):
                #     if budget_label[key] != None:
                #         x.append(key)
                #         y.append(budget_label[key][-1]['information'])
                # x = np.array(x)
                # y = np.array(y)
                # kl = KneeLocator(x, y, curve="concave", direction="increasing", S=0.5)
                # knee = kl.elbow
                # kl.plot_knee()
                # if knee == None:
                #     continue
                # label_list = budget_label[knee]

                budget_label_list = list(budget_label.items())
                for i in range(len(budget_label_list) - 1):
                    diff = budget_label_list[i + 1][-1][-1]['information'] - budget_label_list[i][-1][-1]['information']
                    if diff < min_information:
                        i = budget_label_list[i][0]
                        label_list = budget_label[i]
                        break

        if keyb == 1:
            continue

        # label point
        # label_list = sorted(label_list, key=lambda item: ('surface', 'line', 'point').index(item['type']))
        label_list = [d for d in label_list if d['type'] == 'point']
        for i in range(len(label_list)):
            indicator = label_list[i]['indicator']
            ix, iy = indicator
            cls = truelabel[ix, iy, 0]
            if cls >= opt.num_classes:
                continue
            color = read_cls_color(cls)
            new_mask[ix:ix+3, iy:iy+3, :] = (cls, cls, cls)
            new_mask_vis[ix:ix+3, iy:iy+3, :] = color
            point_weight[ix:ix+3, iy:iy+3] = 1
            point_budget = point_budget + 1
            budget = budget + 1


        imsave(label_path + filename, new_mask)
        imsave(label_vis_path + filename, new_mask_vis)
        np.save(surface_weight_path + filename[:-4] + '.npy', surface_weight)
        np.save(line_weight_path + filename[:-4] + '.npy', line_weight)
        np.save(point_weight_path + filename[:-4] + '.npy', point_weight)
        if budget > budget_part:
            print('surface_budget', surface_budget)
            print('line_budget', line_budget)
            print('point_budget', point_budget)
            return surface_budget, line_budget, point_budget


        # # prototype evaluation
        # true = 0
        # all = 0
        # for i in range(len(split_instance)):
        #     tmp_mask = split_instance[i]['segmentation']
        #     indices = np.where(tmp_mask == 1)
        #     label_mask = np.ones([h, w], np.uint8) * 255
        #     label_mask[indices] = new_mask[indices[0], indices[1], 0]
        #     unique_counts = np.bincount(label_mask[label_mask != 255])
        #     if len(unique_counts) == 0:  # 若只有255，则返回空
        #         continue
        #     else:
        #         result = np.argmax(unique_counts)
        #     score, cls = instance_score(out_seg, tmp_mask)
        #     if result == cls and score > 0.9:
        #         true = true + 1
        #     if score > 0.9:
        #         all = all + 1
        # # acc = true / all
        # if all != 0:
        #     acc.append(true / all)
        # if len(acc) > 10:
        #     acc = acc[-10:]
        # # if np.sum(acc) / len(acc) > 0.8:
        # #     use_pro = True
        # print(np.sum(acc) / len(acc))


        # # prototype update
        # out_seg = F.interpolate(out_seg.float(), size=feat.size()[2:], mode='bilinear')
        # gt_seg = torch.max(out_seg, 1)[1].view(-1)
        # prototypes, contrast_logits, contrast_target = prototype_learning(prototypes, _c, out_seg, gt_seg, masks,
        #                                                                   opt.num_classes, num_prototype, gamma=gamma)


if __name__ == '__main__':
    generate()

