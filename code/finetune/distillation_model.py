import tensorflow as tf


class Distillation_Model:

    def __init__(self, input_data):
        self.disp_console = True
        self.alpha = 0.1
        self._build_network(input_data)

    def _build_network(self, x):
        self.conv_1 = self._conv_layer(1, x, 64, 5, 2)
        self.pool_2 = self._pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self._conv_layer(3, self.pool_2, 128, 3, 1, bn=True)
        self.conv_4 = self._conv_layer(4, self.conv_3, 128, 3, 1, bn=True)
        self.pool_5 = self._pooling_layer(5, self.conv_3, 2, 2)
        self.conv_6 = self._conv_layer(6, self.pool_5, 256, 3, 2, bn=True)
        self.conv_7 = self._conv_layer(7, self.conv_6, 256, 3, 1, bn=True)
        self.pool_8 = self._pooling_layer(8, self.conv_7, 2, 2)
        self.fc_9 = self._fc_layer(9, self.pool_8, 1024, flat=True, linear=False)
        self.fc_10 = self._fc_layer(10, self.fc_9, 5, flat=False, linear=True)

    def _conv_layer(self, idx, inputs, output_channels, size, stride, bn=False):
        inputs_channels = inputs.get_shape()[3]
        weights = tf.Variable(tf.truncated_normal([size, size, int(inputs_channels),
                                                   output_channels], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[output_channels]))
        conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1],
                            padding="SAME", name=str(idx) + "_conv")
        conv_biases = tf.add(conv, biases, name=str(idx) + "_conv_biases")
        if self.disp_console:
            print("Layer {0}: Type=Conv Size={1}x{2} Stride={3} Input_channels={4} \
                Output_channels={5}".format(idx, size, size, stride,
                                            inputs_channels, output_channels))
        if bn:
            return tf.layers.batch_normalization(tf.maximum(self.alpha * conv_biases, conv_biases), name=str(idx) + "bn_conv")
        return tf.maximum(self.alpha * conv_biases, conv_biases, name=str(idx) + "_leaky_relu")

    def _pooling_layer(self, idx, inputs, size, stride):
        if self.disp_console:
            print("Layer {0}: Type=Pool Size={1}x{2} Stride={3}".format(idx, size, size, stride))
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1],
                              padding="SAME", name=str(idx) + "_pool")

    def _fc_layer(self, idx, inputs, flattens, flat=False, linear=False):
        inputs_shape = inputs.get_shape().as_list()
        if flat:
            dim = inputs_shape[1] * inputs_shape[2] * inputs_shape[3]
            input_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            input_processed = tf.reshape(input_transposed, [-1, dim])
        else:
            dim = inputs_shape[1]
            input_processed = inputs

        weights = tf.Variable(tf.truncated_normal([dim, flattens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[flattens]))
        res = tf.add(tf.matmul(input_processed, weights), biases, name=str(idx) + "_fc")
        if self.disp_console:
            print("Layer {0}: Type=Fc Flattens={1} Flat={2} Activation={3}".
                  format(idx, flattens, flat, linear))
        if linear:
            return res
        return tf.maximum(self.alpha * res, res, name=str(idx) + "_fc")
