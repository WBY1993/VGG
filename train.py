import os
import numpy as np
import tensorflow as tf
import input_data
import vgg_16

BATCH_SIZE = 64
IMG_H = 32
IMG_W = 32
CLASS_NUM = 240
LEARNING_RATE = 1e-4
MAX_STEP = 100000
SAVE_DIR = "/home/rvlab/program/VGG/log/"

# #######################
#  write tfrecords file
# #######################
# images, labels = input_data.get_files("/home/rvlab/program/WBYFace/")
# index = int(len(images) * 4 / 5)
# tra_images = images[:index]
# tra_labels = labels[:index]
# val_images = images[index:]
# val_labels = labels[index:]
# input_data.write_tfrecord(tra_images, tra_labels, "/home/rvlab/program/WBYFace_tra.tfrecords")
# input_data.write_tfrecord(val_images, val_labels, "/home/rvlab/program/WBYFace_val.tfrecords")


# ######################
#  read tfrecords file
# ######################
with tf.name_scope("input"):
    tra_img_batch, tra_lab_batch = input_data.read_tfrecord("/home/rvlab/program/WBYFace_tra.tfrecords",
                                                            batch_size=BATCH_SIZE,
                                                            image_H=IMG_H,
                                                            image_W=IMG_W)
    val_img_batch, val_lab_batch = input_data.read_tfrecord("/home/rvlab/program/WBYFace_val.tfrecords",
                                                            batch_size=BATCH_SIZE,
                                                            image_H=IMG_H,
                                                            image_W=IMG_W)

x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, 3])
y = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])
logits = vgg_16.inference(x, CLASS_NUM)
loss = vgg_16.losses(logits, y)
acc = vgg_16.evaluation(logits, y)
train_op = vgg_16.training(loss, LEARNING_RATE)
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#######################
#  fine-tuning
#######################
# data_dict = np.load("/home/rvlab/program/VGG/vgg16.npy", encoding="latin1").item()
# for layer_name in data_dict:
#     if layer_name not in ["fc8"]:
#         with tf.variable_scope(layer_name, reuse=True):
#             for key, data in zip(("weights", "biases"), data_dict[layer_name]):
#                 sess.run(tf.get_variable(key).assign(data))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
tra_summary_writer = tf.summary.FileWriter(os.path.join(SAVE_DIR, "tra"), sess.graph)
val_summary_writer = tf.summary.FileWriter(os.path.join(SAVE_DIR, "val"), sess.graph)

for step in range(MAX_STEP):
    data_batch, label_batch = sess.run([tra_img_batch, tra_lab_batch])
    _, summary_str, tra_loss, tra_acc = sess.run([train_op, summary_op, loss, acc],
                                                 feed_dict={x: data_batch,
                                                            y: label_batch})
    if step % 100 == 0 or (step + 1) == MAX_STEP:
        tra_summary_writer.add_summary(summary_str, step)
        print("Train Step %d, loss: %.4f, accuracy: %.4f" % (step, tra_loss, tra_acc))
    if step % 1000 == 0 or (step + 1) == MAX_STEP:
        data_batch, label_batch = sess.run([val_img_batch, val_lab_batch])
        summary_str, val_loss, val_acc = sess.run([summary_op, loss, acc],
                                                  feed_dict={x: data_batch,
                                                             y: label_batch})
        val_summary_writer.add_summary(summary_str, step)
        print("## Val Step %d, loss: %.4f, accuracy: %.4f" % (step, val_loss, val_acc))
    if (step + 1) % 10000 == 0 or (step + 1) == MAX_STEP:
        saver.save(sess, os.path.join(SAVE_DIR, "model.ckpt"), global_step=step)

coord.request_stop()
coord.join(threads)
sess.close()
