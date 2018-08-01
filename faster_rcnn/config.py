""" 参数 """
from math import *


class Config:
    def __init__(self):
        # 数据增广参数,训练时设置为True，测试时设置为False
        self.use_horizontal_flips = False # 水平翻转
        self.use_vertical_flips = False # 垂直翻转
        self.rot_90 = False # 90°旋转

        # anchor参数
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1 / sqrt(2), 2 / sqrt(2)], [2 / sqrt(2), 1 / sqrt(2)]]

        # 将图片的最小边resize后的size
        self.im_size = 300

        # 图像通道归一化参数
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1

        # roi数量参数
        self.num_rois = 8

        # rpn步长参数
        self.rpn_stride = 16

        # 分类样本均衡化参数
        self.balanced_classes = False

        # 标准差比例参数
        self.std_scaling = 4
        self.classifier_regr_std = [8, 8, 4, 4]

        # rpn的roi最大最小重叠率参数
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # 分类器的roi最大最小重叠率参数
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # 类别图参数
        self.class_mapping = None

        # 基本网络的的参数文件路径
        self.base_net_weights = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

        # 模型参数保存路径
        self.model_path = 'faster_rcnn_weights.h5'

        # 参数文件保存路径
        self.config_filename = 'config.pickle'
