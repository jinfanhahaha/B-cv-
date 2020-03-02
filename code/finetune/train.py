import numpy as np
import tensorflow as tf
from core.distillation_model2 import Distillation_Model2
from core.dataset import DataSet
import os


os.environ["CODA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CODA_VISIBLE_DEVICES"] = "3"

learn_rate = 1e-7
ckpt_file = "./ckpt/train_model.ckpt"
trainset = DataSet('./data/train_data.txt')
testset = DataSet('./data/test_data.txt')

input_data = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32, name='input_data')
label = tf.placeholder(shape=[None, 5], dtype=tf.float32, name="label")

with tf.device("/gpu:0"):
    model = Distillation_Model2(input_data)
    y_ = model.fc10
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y_))
    correct_prediction = tf.equal(tf.argmax(label, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    train_epoch_acc = []
    test_epoch_acc = []
    test_epoch_loss = []
    batch_size = 20
    train_steps = 1000
    epochs = 30
    iter = 0
    for epoch in range(epochs):
        for i in range(train_steps):
            batch_data, batch_labels = trainset.next_batch(batch_size)
            _, train_step_loss, train_step_acc = sess.run(
                [train_op, loss, accuracy], feed_dict={
                    input_data: batch_data,
                    label: batch_labels
                })
            print("Epoch %d, train step: %d, train_loss: %.2f" % (epoch, i, train_step_loss))
            train_epoch_acc.append(np.mean(train_step_acc))
            test_data, test_labels = testset.next_batch(batch_size)
            test_step_loss, test_step_acc = sess.run(
                [loss, accuracy], feed_dict={
                    input_data: test_data,
                    label: test_labels
                })
            test_epoch_acc.append(test_step_acc)
            test_epoch_loss.append(test_step_loss)
            if iter % 10 == 0:
                print("Epoch %d, test step: %d, test loss: %.2f" % (epoch, i, np.mean(test_epoch_loss)))
                print("Train Acc %.2f" % np.mean(train_epoch_acc))
                print("Test Acc %.2f" % np.mean(test_epoch_acc))
                train_epoch_acc = []
                test_epoch_acc = []
                test_epoch_loss = []
            iter += 1
    saver.save(sess, ckpt_file)

