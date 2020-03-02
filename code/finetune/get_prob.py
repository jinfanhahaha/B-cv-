from unkown import DataSet
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
import numpy as np
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
train_data = DataSet("./data/train_data.txt")
slim = tf.contrib.slim

y_prob = np.ones([len(train_data.X), 5])

with tf.Graph().as_default():
    x = tf.placeholder("float32", [None, 224, 224, 3])
    with tf.device("/gpu:0"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, end_points = vgg.vgg_16(
                x, num_classes=5, is_training=False)
            print(end_points)
        fc8 = end_points["vgg_16/fc8"]

        prob = tf.nn.softmax(fc8 / 20)

        print(prob.shape)
        saver = tf.train.Saver()
        load_fn = slim.assign_from_checkpoint_fn("./ckpt/model.ckpt", slim.get_model_variables())
        with tf.Session() as sess:
            load_fn(sess)
            for i in range(len(train_data.X)):
                res = sess.run(prob, feed_dict={x: train_data.X[0].reshape([1, 224, 224, 3])})
                y_prob[i] = res
                print(res[0].shape)
            np.save("./npy/y_prob.npy", y_prob)


