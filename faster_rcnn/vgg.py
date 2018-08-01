""" 定义神经网络 """
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.layers import TimeDistributed
from . import RoiPoolingConv


# 定义基本网络作为共享层，这里为VGG-16的前13层
def nn_base(img_input, trainable=False):
    # 模块1
    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1', trainable = trainable)(img_input)
    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2', trainable = trainable)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool')(x)

    # 模块2
    x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1', trainable = trainable)(x)
    x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2', trainable = trainable)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool')(x)

    # 模块3
    x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1', trainable = trainable)(x)
    x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2', trainable = trainable)(x)
    x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3', trainable = trainable)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool')(x)

    # 模块4
    x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv1', trainable = trainable)(x)
    x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2', trainable = trainable)(x)
    x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3', trainable = trainable)(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block4_pool')(x)

    # 模块5
    x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv1', trainable = trainable)(x)
    x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2', trainable = trainable)(x)
    x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv3', trainable = trainable)(x)

    return x


# 定义rpn，包含共享层，一个卷积层，以及分类层与回归层
def rpn(base_layers, num_anchors):

    x = Conv2D(512, (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'normal', name = 'rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation = 'sigmoid', kernel_initializer = 'uniform', name = 'rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation = 'linear', kernel_initializer = 'zero', name = 'rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


# 定义分类网络
def classifier(base_layers, roi_input, num_rois, nb_classes, trainable = False):
    pooling_regions = 7
    # 经过RoiPoolingConv层得到num_rois个区域进行训练
    out_roi_pool = RoiPoolingConv.RoiPoolingConv(pooling_regions, num_rois)([base_layers, roi_input])

    out = TimeDistributed(Flatten(name = 'flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(512, activation = 'relu', name = 'fc1', trainable = trainable))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(512, activation = 'relu', name = 'fc2', trainable = trainable))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    # 进行多分类
    out_class = TimeDistributed(Dense(nb_classes, activation = 'softmax', kernel_initializer = 'zero'), name = 'dense_class_{}'.format(nb_classes))(out)
    # 进行bounding boxes回归
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation = 'linear', kernel_initializer = 'zero'), name = 'dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
