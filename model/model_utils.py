# -*- coding: utf-8 -*-

# Author: Yang Zhang
# Mail: zyziszy@foxmail.com
# Apache 2.0.

import math
import numpy as np
import tensorflow as tf
from scipy import stats
import random

def get_skew_and_kurt(data):
    '''calculate skew and kurt'''
    data = np.array(data)
    data = data.transpose()
    print(data.shape)  # test
    skew = []
    kurt = []
    for i in data:
        # print(len(i))
        skew.append(stats.skew(i))
        kurt.append(stats.kurtosis(i))

    skew_mean = sum(skew)/len(skew)  
    kurt_mean = sum(kurt)/len(kurt)

    # print('skew:', skew_mean)  # test
    # print('kurt:', kurt_mean)  # test
    return skew_mean, kurt_mean

def shuffle_data_table(data, table):
    '''random shuffle data and table'''
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    table = table[index]
    return data, table

def shuffle_data(data):
    '''random shuffle data'''
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    return data

def MLP_net(input, layer_name, n_hidden, acitvate="elu"):
    '''tensorflow-layer'''
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)

    w_str = 'w_'+str(layer_name)
    b_str = 'b_'+str(layer_name)

    w = tf.get_variable(
        w_str, [input.get_shape()[1], n_hidden], initializer=w_init)
    b = tf.get_variable(b_str, [n_hidden], initializer=b_init)

    output = tf.matmul(input, w) + b

    if acitvate == 'tanh':
        output = tf.nn.tanh(output)
    elif acitvate == 'sigmoid':
        output = tf.nn.sigmoid(output)
    else:
        output = tf.nn.elu(output)
    return output
