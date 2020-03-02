import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
import os
from core.dataset import DataSet


slim = tf.contrib.slim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

with tf.Graph().as_default():
    x = tf.placeholder("float32", [None, 224, 224, 3])
    y = tf.placeholder("float32", [None, 5])
    with tf.device("/cpu:0"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, end_points = vgg.vgg_16(
                x, num_classes=5, is_training=True)
            print(end_points)
        fc8 = end_points["vgg_16/fc8"]
        print(fc8.shape)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc8))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(fc8, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variables_to_restore = []
        trainable_var_list = []
        for var in tf.global_variables():
            if var.name.split("/")[1] == "fc8":
                trainable_var_list.append(var)
                continue
            variables_to_restore.append(var)
        print(variables_to_restore)
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=trainable_var_list)
        load_fn = slim.assign_from_checkpoint_fn("./ckpt/vgg_16.ckpt", variables_to_restore)
        saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        load_fn(sess)
        batch_size = 20
        train_steps = 1000
        epochs = 10
        trainset = DataSet("./data/train_data.txt")
        testset = DataSet("./data/test_data.txt")
        for epoch in range(epochs):
            for i in range(train_steps):
                batch_data, batch_labels = trainset.next_batch(batch_size)
                _, train_step_loss, train_step_acc = sess.run(
                    [train_op, loss, accuracy], feed_dict={
                        x: batch_data,
                        y: batch_labels
                    })
                print("Loss: ", train_step_loss)
                print("Acc: ", train_step_acc)
            batch_test_data, batch_test_labels = testset.next_batch(100)
            test_step_loss, test_step_acc = sess.run(
                [loss, accuracy], feed_dict={
                    x: batch_test_data,
                    y: batch_test_labels
                })
            print("Test Loss: ", test_step_loss)
            print("Test Acc: ", test_step_acc)
        saver.save(sess, "./ckpt/model.ckpt")



