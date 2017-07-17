#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import threading
import shutil
import time
import numpy as np

"""
net即网络参数
每一组第一个参数是类别，分为c卷积、p池化、r扁平化、f全连接
卷积层第二个参数是卷积和边长，第三个是输出频道数，可选第四个参数是激活函数（若不写则默认使用tf.nn.sigmoid）
池化层第二个参数是池化核边长，可选第三个参数是池化函数（若不写则默认使用tf.nn.max_pool）
"""
net = [['c', 7, 32],
       ['p', 2],
       ['c', 17, 64],
       ['p', 3],
       ['c', 7, 128],
       ['p', 3],
       ['r'],
       ['f', 1024],
       ]
iteration = 200
batch_size = 100
optimizer = tf.train.AdamOptimizer(0.00001)
global_act_fun = tf.nn.relu

ten = tf.constant(10.0, dtype=tf.float32)


def weight_variable(shape, mean=0.0, stddev=0.01):
    initial = tf.random_normal(shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, value=0.0):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)


def new_conv_layer(input_layer, kernel_size, input_channel, output_channel, act_fun=tf.nn.sigmoid):
    W = weight_variable([kernel_size, kernel_size, input_channel, output_channel])
    b = bias_variable([output_channel])
    # input:shape=[batch, in_height, in_width, in_channels]
    # W:shape=[filter_height, filter_width, in_channels, out_channels]
    return act_fun(tf.nn.conv2d(input_layer, W, strides=[1, 1, 1, 1], padding='VALID') + b)


def new_pool_layer(input_layer, pool_size, pool_fun=tf.nn.avg_pool):
    # input:shape=[batch, height, width, channels]
    # ksize:shape=[batch, height, width, channels]
    return pool_fun(input_layer,
                    ksize=[1, pool_size, pool_size, 1],
                    strides=[1, pool_size, pool_size, 1],
                    padding='VALID')


def new_full_layer(input_layer, input_channel, output_channel, act_fun=tf.nn.sigmoid):
    W = weight_variable([input_channel, output_channel])
    b = bias_variable([output_channel])
    return act_fun(tf.matmul(input_layer, W) + b)


def train_model(train_data, test_data):
    print('training initializing')

    train_n = len(train_data["x"])
    h, w = train_data["x"][0].shape
    test_n = len(test_data["x"])

    # 初始化
    x = tf.placeholder(tf.float32, shape=[None, h, w], name="input")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="label")
    input_layer = tf.reshape(x, [-1, h, w, 1])  # 从普通状态变为运算状态

    # 搭网络结构
    layers = list()
    curr_channel = 1
    curr_shape = [h, w]
    curr_length = curr_shape[0] * curr_shape[1] * curr_channel
    for i, paras in enumerate(net):
        this_input_layer = input_layer if i == 0 else layers[i-1]
        if paras[0] == 'c':
            act_fun = global_act_fun
            if len(paras) == 4:
                act_fun = paras[-1]
            layer = new_conv_layer(this_input_layer, paras[1], curr_channel, paras[2], act_fun=act_fun)
            curr_channel = paras[2]
            curr_shape = [_ + 1 - paras[1] for _ in curr_shape]
        elif paras[0] == 'p':
            pool_fun = tf.nn.max_pool
            if len(paras) == 3:
                pool_fun = paras[-1]
            layer = new_pool_layer(this_input_layer, paras[1], pool_fun)
            if not np.sum([_ % paras[1] for _ in curr_shape]) == 0:  # 若长和宽有一个不能整除pooling长度，则报错
                print("WARNING: unsuitable pooling")
            curr_shape = [_ // paras[1] for _ in curr_shape]
        elif paras[0] == 'r':
            curr_length = curr_shape[0] * curr_shape[1] * curr_channel
            layer = tf.reshape(this_input_layer, [-1, curr_length])
        elif paras[0] == 'f':
            act_fun = global_act_fun
            if len(paras) == 3:
                act_fun = paras[-1]
            layer = new_full_layer(this_input_layer, curr_length, paras[1], act_fun=act_fun)
            curr_length = paras[1]
        else:
            layer = None
        layers.append(layer)

    W = weight_variable([curr_length, 1])
    b = bias_variable([1])
    prediction = tf.matmul(layers[-1], W) + b
    loss = tf.losses.mean_squared_error(y, prediction)
    loss_out = tf.losses.mean_squared_error(y, tf.div(tf.round(tf.scalar_mul(ten, prediction)), ten))
    train_step = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('start training')
    #file = open(r'.\results\new.txt', 'w')
    for i in range(iteration):
        print("iteration %g" % i)

        # 分批，每批数量为batch_size，总量为train_n
        n = train_n // batch_size
        batch_i = i % n
        batch_x = train_data["x"][batch_i * batch_size: (batch_i + 1) * batch_size - 1]
        batch_y = train_data["y"][batch_i * batch_size: (batch_i + 1) * batch_size - 1]

        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

        train_accuracy = loss_out.eval(session=sess, feed_dict={x: batch_x, y: batch_y})
        print(prediction.eval(session=sess, feed_dict={x: batch_x, y: batch_y}))
        print('loss: %g' % train_accuracy)

        #file.write(str(f)+'\t')

        print('\n')

    print('test loss: %g' % loss_out.eval(session=sess, feed_dict={x: test_data["x"], y: test_data["y"]}))

    #file.close()
    sess.close()

#if __name__ == "__main__":
    #data = readData(r".\newData\sparse\random_full_0.01.data", 'average_rate')
    #with open(r'.\newData\sparse\random_full_0.01_ave-sorted.serialized', 'rb') as f: data = pickle.load(f)
    #train_model(data["train"], data["test"])
