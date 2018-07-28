import tensorflow as tf


def conv(layer_name, x, in_channel, out_channel, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True):
    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name="weights",
                                  shape=[ksize[0],
                                         ksize[1],
                                         in_channel,
                                         out_channel],
                                  trainable=is_train,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        biases = tf.get_variable(name="biases",
                                 shape=[out_channel],
                                 trainable=is_train,
                                 initializer=tf.constant_initializer(0.1))

        x = tf.nn.conv2d(x, weights, stride, padding="SAME")
        x = tf.nn.bias_add(x, biases)
    return x


def pool(layer_name, x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1]):
    with tf.variable_scope(layer_name):
        x = tf.nn.max_pool(x, ksize=ksize, strides=stride, padding="SAME")
    return x


def fc(layer_name, x, in_channel, out_channel, is_train=True):
    with tf.variable_scope(layer_name):
        x = tf.reshape(x, shape=[-1, in_channel])
        weights = tf.get_variable(name="weights",
                                  shape=[in_channel, out_channel],
                                  trainable=is_train,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))

        biases = tf.get_variable(name="biases",
                                 shape=[out_channel],
                                 trainable=is_train,
                                 initializer=tf.constant_initializer(0.1))

        x = tf.matmul(x, weights)
        x = tf.nn.bias_add(x, biases)
    return x


def relu(layer_name, x):
    with tf.variable_scope(layer_name):
        x = tf.nn.relu(x)
    return x


def batch_norm(layer_name, x):
    with tf.variable_scope(layer_name):
        batch_mean, batch_var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x=x,
                                      mean=batch_mean,
                                      variance=batch_var,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=1e-3)
    return x


def dropout(layer_name, x):
    with tf.variable_scope(layer_name):
        x = tf.nn.dropout(x, keep_prob=0.5)
    return x


def inference(x, class_num):
    '''
    :param x: [batch_size, height, width, channel]
    :param class_num:
    :return:
    '''
    # input data: [batch_size, 224, 224, 3]
    x = conv("conv1_1", x, 3, 64, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu1_1", x)
    x = conv("conv1_2", x, 64, 64, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu1_2", x)
    x = pool("pool1", x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

    x = conv("conv2_1", x, 64, 128, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu2_1", x)
    x = conv("conv2_2", x, 128, 128, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu2_2", x)
    x = pool("pool2", x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

    x = conv("conv3_1", x, 128, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu3_1", x)
    x = conv("conv3_2", x, 256, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu3_2", x)
    x = conv("conv3_3", x, 256, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu3_3", x)
    x = pool("pool3", x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

    x = conv("conv4_1", x, 256, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu4_1", x)
    x = conv("conv4_2", x, 512, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu4_2", x)
    x = conv("conv4_3", x, 512, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu4_3", x)
    x = pool("pool4", x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

    x = conv("conv5_1", x, 512, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu5_1", x)
    x = conv("conv5_2", x, 512, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu5_2", x)
    x = conv("conv5_3", x, 512, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    x = relu("relu5_3", x)
    x = pool("pool5", x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

    x = fc("fc6", x, 1*1*512, 4096, is_train=True)
    x = relu("relu6", x)
    x = dropout("batch_norm6", x)
    x = fc("fc7", x, 4096, 4096, is_train=True)
    x = relu("relu7", x)
    x = dropout("batch_norm7", x)
    x = fc("fc8", x, 4096, class_num, is_train=True)

    return x


def losses(logits, labels):
    '''
    :param logits: [batch_size, class_num]
    :param labels: [batch_size]
    :return:
    '''
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss", loss)
    return loss


def evaluation(logits, labels):
    '''
    :param logits: [batch_size, class_num]
    :param labels: [batch_size]
    :return:
    '''
    with tf.name_scope("accuracy"):
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar("accuracy", accuracy)
    return accuracy


def training(loss, learning_rate):
    '''
    :param loss:
    :param learning_rate:
    :return:
    '''
    with tf.name_scope("optimizer"):
        global_step = tf.Variable(0, trainable=False)
        decayed_lr = tf.train.exponential_decay(learning_rate=learning_rate,
                                                global_step=global_step,
                                                decay_steps=10000,
                                                decay_rate=0.5,
                                                staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
