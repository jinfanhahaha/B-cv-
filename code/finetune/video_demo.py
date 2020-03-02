import ast
import tensorflow as tf
import numpy as np
import os
import cv2
from core.distillation_model2 import Distillation_Model2


slim = tf.contrib.slim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
with open("./data/flower.txt", "r") as f:
    data = f.read()
    images_dicts = ast.literal_eval(data)


def get_image(image):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = np.array(cv2.resize(image, (224, 224))).reshape([1, 224, 224, 3]).astype(np.float)
    image[:, :, 0] -= _R_MEAN
    image[:, :, 1] -= _G_MEAN
    image[:, :, 2] -= _B_MEAN
    return image


with tf.Graph().as_default():
    x = tf.placeholder("float32", [None, 224, 224, 3])
    model = Distillation_Model2(x)
    pred = tf.argmax(model.fc10, 1)
    print(pred.shape)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./ckpt/distillation_model.ckpt")
        # image = get_image("./data/flower_photos/sunflowers/3.jpg")
        # img = cv2.imread("./data/flower_photos/sunflowers/3.jpg")
        capture = cv2.VideoCapture("./data/test/3.mp4")
        while True:
            ret, frame = capture.read()
            img = frame.copy()
            image = get_image(frame)
            prob = sess.run(pred, feed_dict={x: image})
            print(images_dicts[prob[0]])
            cv2.putText(img, images_dicts[prob[0]], (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("hh", img)
            cv2.waitKey(50)
            cv2.destroyAllWindows()
