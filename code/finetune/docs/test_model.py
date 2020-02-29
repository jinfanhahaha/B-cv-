import ast
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
import numpy as np
import os
import time


slim = tf.contrib.slim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

with open("image_net1000.txt", "r") as f:
    data = f.read()
    images_dicts = ast.literal_eval(data)

with tf.Graph().as_default():
    x = tf.placeholder("float32", [None, 224, 224, 3])
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(
            x, num_classes=1000, is_training=False)
        print(end_points)
    pred = tf.argmax(end_points["vgg_16/fc8"], 1)
    print(pred.shape)
    saver = tf.train.Saver()
    load_fn = slim.assign_from_checkpoint_fn("../ckpt/vgg_16.ckpt", slim.get_model_variables())
    with tf.Session() as sess:
        load_fn(sess)
        image = np.load("./npy/1.npy")
        t1 = time.time()
        prob = sess.run(pred, feed_dict={x: image})
        print(images_dicts[prob[0]])
        print(time.time()-t1)
