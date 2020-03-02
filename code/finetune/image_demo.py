import ast
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
import numpy as np
import os
import time
import cv2


slim = tf.contrib.slim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

with open("./data/flower.txt", "r") as f:
    data = f.read()
    images_dicts = ast.literal_eval(data)


def get_image(filename):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = cv2.imread(filename)
    image = np.array(cv2.resize(image, (224, 224))).reshape([1, 224, 224, 3]).astype(np.float)
    image[:, :, 0] -= _R_MEAN
    image[:, :, 1] -= _G_MEAN
    image[:, :, 2] -= _B_MEAN
    return image


with tf.Graph().as_default():
    x = tf.placeholder("float32", [None, 224, 224, 3])
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(
            x, num_classes=5, is_training=False)
        print(end_points)
    pred = tf.argmax(end_points["vgg_16/fc8"], 1)
    print(pred.shape)
    saver = tf.train.Saver()
    load_fn = slim.assign_from_checkpoint_fn("./ckpt/model.ckpt", slim.get_model_variables())
    with tf.Session() as sess:
        load_fn(sess)
        image = get_image("./data/flower_photos/sunflowers/3.jpg")
        img = cv2.imread("./data/flower_photos/sunflowers/3.jpg")
        t1 = time.time()
        prob = sess.run(pred, feed_dict={x: image})
        print(images_dicts[prob[0]])
        cv2.putText(img, images_dicts[prob[0]], (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 2)
        print(time.time()-t1)
        cv2.imshow("hh", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
