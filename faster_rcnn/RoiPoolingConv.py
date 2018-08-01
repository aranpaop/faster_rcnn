""" 自定义的RoiPoolingConv层 """
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class RoiPoolingConv(Layer):
    """
    参数：
        pool_size: 池化区域的大小
        num_rois: 兴趣区域的数量
    输入：
        [X_img, X_roi]：
        X_img:
            (1, rows, cols, channels)
        X_roi:
            (1, num_rois, 4)
    输出：
        `(1, num_rois, pool_size, pool_size， channels)`
    """

    def __init__(self, pool_size, num_rois, **kwargs):

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    # 提取num_rois个样本用于训练
    def call(self, x, mask = None):

        img = x[0]
        rois = x[1]
        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis = 0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output
