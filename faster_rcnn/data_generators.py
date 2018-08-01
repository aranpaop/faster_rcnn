""" 生成数据 """
import numpy as np
import cv2
import random
from . import data_augment
import itertools


# 得到经过共享层计算后的特征图尺寸
def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 16
    return get_output_length(width), get_output_length(height)


# 计算两个bboxes的并集
def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


# 计算两个bboxes的交集
def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


# 图像通道归一化处理
def normalize_img(x_img, C):
    x_img = x_img[:,:, (2, 1, 0)]  # 将BGR转换成RGB通道顺序
    x_img = x_img.astype(np.float32)
    x_img[:, :, 0] -= C.img_channel_mean[0]
    x_img[:, :, 1] -= C.img_channel_mean[1]
    x_img[:, :, 2] -= C.img_channel_mean[2]
    x_img /= C.img_scaling_factor
    x_img = np.expand_dims(x_img, axis = 0)
    return x_img


# 计算IoU
def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    return area_i / (area_u + 1e-6)


# 计算按照统一最短边resize后的图像尺寸
def get_new_img_size(width, height, img_min_side = 300):
    if width <= height:
        f = img_min_side / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = img_min_side / height
        resized_width = int(f * width)
        resized_height = img_min_side
    return resized_width, resized_height, f


# 定义样本选择类
class SampleSelector:
    def __init__(self, class_count):
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    # 忽略无样本的标签，在标签空间内循环
    def skip_sample_for_balanced_class(self, img_data):
        class_in_img = False
        for bbox in img_data['bboxes']:
            cls_name = bbox['class']
            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break
        if class_in_img:
            return False
        else:
            return True


# 计算rpn的分类情况及回归梯度
def calc_rpn(C, img_data, width, height, resized_width, resized_height):
    downscale = C.rpn_stride
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # 得到经过共享层计算后的特征图尺寸
    (output_width, output_height) = get_img_output_length(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # 初始化输出
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # 记录图像中每个bounding box在图像进行resize后的尺寸
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / width)
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / width)
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / height)
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / height)

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            # 计算每个anchor的x、y跨度
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # 计算当前anchor的x坐标
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # 忽略超出图像边界的anchor
                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):
                    # 计算当前anchor的y坐标
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # 忽略超出图像边界的anchor
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # 初始化bounding box类型与每个bounding box的最大IoU
                    bbox_type = 'neg'
                    best_iou_for_loc = 0

                    for bbox_num in range(num_bboxes):
                        # 计算当前anchor与每个bounding box的IoU
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # 依据判断anchor内是否包含对象的条件，对包含对象的anchor计算梯度值用于坐标回归
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2
                            cxa = (x1_anc + x2_anc) / 2
                            cya = (y1_anc + y2_anc) / 2

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        # 忽略标签为背景的bounding box
                        if img_data['bboxes'][bbox_num]['class'] != 'bg':
                            # 更新每个bounding box对应的最佳anchor的参数
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # 将IoU大于0.7的anchor设为pos
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # 更新最大IoU，记录坐标回归参数
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # 将IoU处于0.3与0.7之间的anchor设为neutral
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    if bbox_type == 'neg': # 将标签为neg的anchor视为可用但不包含对象
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral': # 将标签为neutral的anchor视为不可用且不包含对象
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos': # 将标签为pos的anchor视为可用且包含对象
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # 保证每一个bounding box都有至少一个标签为pos的anchor与之对应
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[best_anchor_for_bbox[idx, 0],
                           best_anchor_for_bbox[idx, 1],
                           best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[best_anchor_for_bbox[idx, 0],
                          best_anchor_for_bbox[idx, 1],
                          best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[best_anchor_for_bbox[idx, 0],
                       best_anchor_for_bbox[idx, 1],
                       start:start + 4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis = 0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis = 0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis = 0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # 考虑到pos样本要远小于neg样本，这里规定了anchor数上限，并使得pos样本与neg样本数量相等
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis = 1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis = 1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


# 得到用于分类的anchor
def get_anchor_gt(all_img_data, class_count, img_path, C, mode = 'train'):
    sample_selector = SampleSelector(class_count)
    while True:
        if mode == 'train':
            np.random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:

                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # 自动进行数据增广
                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, img_path, C, augment = True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, img_path, C, augment = False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                # 按最小边统一大小，得到图像resize后的尺寸
                (resized_width, resized_height, _) = get_new_img_size(width, height, C.im_size)

                # 进行图像resize
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation = cv2.INTER_CUBIC)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height)
                except:
                    continue
                x_img = normalize_img(x_img, C)

                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling

                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(e)
                continue
