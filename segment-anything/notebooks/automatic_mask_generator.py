import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import pickle
import copy
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def calculate_iofs(gt_mask, pred_masks):
    # 将gt_mask扩展为与pred_masks相同的形状
    expanded_gt_mask = np.expand_dims(gt_mask, axis=0)
    expanded_gt_mask = np.repeat(expanded_gt_mask, len(pred_masks), axis=0)
    # 计算交集
    intersection = np.logical_and(expanded_gt_mask, pred_masks)
    # 计算交集面积
    intersection_area = np.sum(intersection, axis=(1, 2))
    # 计算预测掩码的面积
    pred_area = np.sum(pred_masks, axis=(1, 2))
    # 计算IoF
    iofs = intersection_area / pred_area
    return iofs

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def masks_nms(anns, iof_thresh=0.5, area_thresh=200, miss_thresh=1000):
    if len(anns) == 0:
        return
    root = Node(1)

    # miss
    mask_all = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1]), dtype=bool)
    for ann in anns:
        mask_all = np.logical_or(mask_all, ann['segmentation'])
    mask_all = np.logical_not(mask_all)
    temp_mask = np.asarray(mask_all, dtype=np.uint8)
    num_objects, regions, stats, centroids = cv2.connectedComponentsWithStats(temp_mask, connectivity=8)
    for i in range(num_objects):
        if np.all(temp_mask[regions == i] == 0) or stats[i][4] < miss_thresh:
            continue
        else:
            tmp = np.where(regions == i, True, False)
            root.add_child(Node({'segmentation': tmp, 'area': np.sum(tmp)}))

    anns = [value for i, value in enumerate(anns) if anns[i]['area'] > area_thresh]
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    index = list(range(len(sorted_anns)))
    while len(index) > 0:
        i = index[0]
        root.add_child(Node(sorted_anns[i]))
        if len(index) <= 1:
            break
        iofs = calculate_iofs(sorted_anns[i]['segmentation'], [sorted_anns[j]['segmentation'] for j in index[1:]])
        idx_subs = np.where(iofs >= iof_thresh)[0] + 1
        node_index = len(root.children) - 1
        mask_sub = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=bool)

        for idx_sub in idx_subs:
            if root.children[node_index].children == []:
                root.children[node_index].add_child(sorted_anns[index[idx_sub]])
                mask_sub[sorted_anns[index[idx_sub]]['segmentation']] = True
            else:
                intersection = np.logical_and(mask_sub, sorted_anns[index[idx_sub]]['segmentation'])
                sorted_anns[index[idx_sub]]['segmentation'] = np.logical_xor(intersection, sorted_anns[index[idx_sub]]['segmentation'])
                area = np.sum(sorted_anns[index[idx_sub]]['segmentation'])
                if area > (miss_thresh / 2):
                    sorted_anns[index[idx_sub]]['area'] = area
                    root.children[node_index].add_child(sorted_anns[index[idx_sub]])
                    mask_sub[sorted_anns[index[idx_sub]]['segmentation']] = True

        # mask_part = copy.deepcopy(root.children[node_index].value)
        # intersection = np.logical_and(mask_sub, root.children[node_index].value['segmentation'])
        # mask_part['segmentation'] = np.logical_xor(intersection, mask_part['segmentation'])
        # area = np.sum(mask_part['segmentation'])
        # if area > area_thresh:
        #     mask_part['area'] = area
        #     root.children[node_index].add_child(mask_part)

        sorted_children = sorted(root.children[node_index].children, key=(lambda x: x['area']), reverse=True)
        root.children[node_index].children = sorted_children

        idx = np.where(iofs < iof_thresh)[0] + 1
        index = [index[j] for j in idx]  # 处理剩余的边框

    # ax = plt.gca()
    # ax.set_autoscale_on(False)
    #
    # img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    # img[:, :, 3] = 0
    # for child in root.children:
    #     m = child.value['segmentation']
    #     color_mask = np.concatenate([np.random.random(3), [0.35]])
    #     img[m] = color_mask
    # for childs in root.children:
    #     for child in childs.children:
    #         m = child['segmentation']
    #         color_mask = np.concatenate([np.random.random(3), [0.35]])
    #         img[m] = color_mask

    # ax.imshow(img)

    return root

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    sam_checkpoint = "../weight/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=200,  # Requires open-cv to run post-processing
    )
    train_path = '/home/isalab206/Downloads/dataset/loveda/train/img/'
    save_path = '/mnt/vdf/isalab205/LJX_data/dataset/loveda/train/sam_pkl/'
    for i in tqdm(os.listdir(train_path)):
        image = cv2.imread(train_path + i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        masks = mask_generator.generate(image)
        root = masks_nms(masks)
        f = open(save_path + i[:-4] + '.pkl', 'wb')
        content = pickle.dumps(root)
        f.write(content)
        f.close()

