import tensorflow as tf


class Distillation_Model2:

    def __init__(self, input_data):
        self.disp_console = True
        self._build_network(input_data)

    def _build_network(self, x):
        self.conv1 = self._conv_layer(1, x, 64, 2, bn=False)
        self.pool2 = self._pool_layer(2, self.conv1)
        self.conv3 = self._conv_layer(3, self.pool2, 128, 1, bn=True)
        self.conv4 = self._conv_layer(4, self.conv3, 128, 1, bn=True)
        self.pool5 = self._pool_layer(5, self.conv4)
        self.conv6 = self._conv_layer(6, self.pool5, 256, 2, bn=True)
        self.conv7 = self._conv_layer(7, self.conv6, 256, 1, bn=True)
        self.pool8 = self._pool_layer(8, self.conv7)
        self.flatten = tf.layers.flatten(self.pool8)
        self.fc9 = tf.layers.dense(self.flatten, 1024, activation=tf.nn.relu, name="fc9")
        self.fc10 = tf.layers.dense(self.fc9, 5)

    def _conv_layer(self, idx, input, filters, strides, bn=False):
        if self.disp_console:
            print("Conv{} shape={} output_channels={}".format(idx, input.shape, filters))
        conv = tf.layers.conv2d(input,
                                filters=filters,
                                kernel_size=3,
                                strides=strides,
                                padding="same",
                                name="conv" + str(idx))
        if bn:
            return tf.layers.batch_normalization(conv, name=str(idx) + "bn_conv")
        return conv

    def _pool_layer(self, idx, x):
        if self.disp_console:
            print("Pool{} shape={} ".format(idx, x.shape))
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool" + str(idx))
