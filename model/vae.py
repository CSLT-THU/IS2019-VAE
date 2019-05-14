# -*- coding: utf-8 -*-

# Author: Yang Zhang
# Mail: zyziszy@foxmail.com
# Apache 2.0.

from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import random
from model.model_utils import *
from scipy import stats


class VAE(object):
    model_name = "VAE"     # name for checkpoint

    def __init__(self,
                 sess,
                 dataset_path,
                 checkpoint_dir,
                 log_dir,
                 epoch,
                 batch_size,
                 z_dim,
                 n_hidden,
                 learning_rate,
                 KL_weigth,
                 cohesive_weight,
                 beta1='0.5',
                 spk_path="./data/voxceleb_combined_200000/spk.npz"
                 ):

        self.KL_weigth = KL_weigth
        self.cohesive_weight = cohesive_weight
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        '''load datasets'''
        # training datasets
        self.dataset_path = dataset_path
        self.input_data = np.load(self.dataset_path)['vector']
        self.input_utt = np.load(self.dataset_path)['utt']

        # input_data | spk_list
        self.spk_list = np.load(spk_path)['spk_list']
        self.spker = np.load(spk_path)['spker']

        '''
        count spker number
        self.spk_count_shape : (spk_num, z_dim)
        '''
        spk_count = [0 for _ in range(len(self.spker))]
        for i in self.spk_list:
            spk_count[int(i)] += 1

        self.spk_count = []
        for i in range(len(spk_count)):
            temp = [spk_count[i] for _ in range(z_dim)]
            self.spk_count.append(temp)

        self.epoch = epoch
        self.batch_size = batch_size

        # get number of batches for a single epoch
        self.num_batches = len(self.input_data) // self.batch_size

        self.z_dim = z_dim
        self.n_hidden = n_hidden

        self.dnn_input_dim = 512
        self.dnn_output_dim = 512

        self.z_dim = z_dim         # dimension of v-vector

        # train
        self.learning_rate = learning_rate
        self.beta1 = beta1

        '''build the model'''
        self.chech_data()
        self.build_model()

    def chech_data(self):
        '''use to check data'''
        pass

    # Gaussian MLP Encoder
    def MPL_encoder(self, x, n_hidden, n_output):
        with tf.variable_scope("gaussian_MLP_encoder"):

            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # layer-1
            net = MLP_net(input=x, layer_name="p1",
                          n_hidden=n_hidden, acitvate='sigmoid')
            # layer-2
            net = MLP_net(input=net, layer_name="p2",
                          n_hidden=n_hidden, acitvate='elu')
            # layer-3
            net = MLP_net(input=net, layer_name="p3",
                          n_hidden=n_hidden, acitvate='elu')
            # layer-4
            net = MLP_net(input=net, layer_name="p4",
                          n_hidden=n_hidden, acitvate='tanh')

            wo = tf.get_variable(
                'wo', [net.get_shape()[1], n_output * 2], initializer=w_init)
            bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
            gaussian_params = tf.matmul(net, wo) + bo

            mean = gaussian_params[:, :n_output]
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

        return mean, stddev

    # Bernoulli decoder
    def MLP_decoder(self, z, n_hidden, n_output):
        with tf.variable_scope("bernoulli_MLP_decoder"):

            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # layer-1
            net = MLP_net(input=z, layer_name="q1",
                          n_hidden=n_hidden, acitvate="tanh")
            # layer-2
            net = MLP_net(input=z, layer_name="q2",
                          n_hidden=n_hidden, acitvate="elu")
            # layer-3
            net = MLP_net(input=net, layer_name="q3",
                          n_hidden=n_hidden, acitvate='sigmoid')

            # output
            wo = tf.get_variable(
                'wo', [net.get_shape()[1], n_output], initializer=w_init)
            bo = tf.get_variable('bo', [n_output], initializer=b_init)

            y = tf.matmul(net, wo) + bo

        return y

    # update utt table
    def update_table(self, mean):
        '''
        return utt_table
        '''
        mean = np.array(mean, dtype=np.float32)
        num_spker = len(self.spker)
        counter = np.array(self.spk_count, dtype=np.int32)

        '''init mean'''
        spk_table = np.zeros(shape=(num_spker, self.z_dim), dtype=np.float32)

        '''mean'''
        for i in range(mean.shape[0]):
            spk_table[self.spk_list[i]] += mean[i]

        '''calculate average of mean'''
        spk_table = spk_table/counter

        '''get utt table'''
        utt_table = np.zeros(shape=mean.shape, dtype=np.float32)
        for i in range(utt_table.shape[0]):
            utt_table[i] += spk_table[self.spk_list[i]][0]

        return utt_table

    def build_model(self):
        # some parameters
        """ Graph Input """
        self.inputs = tf.placeholder(
            tf.float32, [None, self.dnn_input_dim], name='input_vector')
        self.inputs_table = tf.placeholder(
            tf.float32, [None, self.z_dim], name='input_table')
        """ Loss Function """

        # encoding
        self.mu, self.sigma = self.MPL_encoder(
            self.inputs, self.n_hidden, self.z_dim)

        # sampling by re-parameterization
        z = self.mu + self.sigma * \
            tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # decoding
        self.out = self.MLP_decoder(
            z, self.n_hidden, self.dnn_output_dim)

        '''loss'''

        # reconstruct loss
        re_construct_loss = tf.reduce_sum(tf.square(self.inputs-self.out), 1)
        self.re_construct_loss = tf.reduce_mean(re_construct_loss)

        # KL Loss
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(
            self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1, 1)
        self.KL_divergence = self.KL_weigth*tf.reduce_mean(KL_divergence)

        # cohesive_loss
        self.cohesive_loss = self.cohesive_weight * \
            tf.losses.mean_squared_error(self.mu, self.inputs_table)

        # all loss
        self.loss = self.re_construct_loss + self.KL_divergence + self.cohesive_loss

        """ Training """
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.loss, var_list=t_vars)

        """ Summary """
        re_loss = tf.summary.scalar("re_loss", self.re_construct_loss)
        kl_sum = tf.summary.scalar("kl_loss", self.KL_divergence)
        cohesive_loss = tf.summary.scalar("cohesive_loss", self.cohesive_loss)
        loss_sum = tf.summary.scalar("loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

        # initialize all variables
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep=250)

        # summary writer
        self.writer = tf.summary.FileWriter(
            self.log_dir + '/' + self.model_name, self.sess.graph)

    def train(self):

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load_ckp(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        for epoch in range(start_epoch, self.epoch):
            mean = self.sess.run(self.mu, feed_dict={
                self.inputs: self.input_data})
            table = self.update_table(mean)

            input_data, table = shuffle_data_table(self.input_data, table)

            # print(table.shape)
            # print(self.spk_count)
            # print(len(self.spk_count))
            # c = input('break')

            for idx in range(start_batch_id, self.num_batches):
                batch_data = input_data[idx *
                                        self.batch_size:(idx+1)*self.batch_size]
                batch_table = table[idx *
                                    self.batch_size:(idx+1)*self.batch_size]
                # update autoencoder
                _, summary_str, loss, re_loss, kl_loss, cohesive_loss = self.sess.run([self.optim, self.merged_summary_op, self.loss, self.re_construct_loss, self.KL_divergence, self.cohesive_loss],
                                                                                      feed_dict={self.inputs: batch_data, self.inputs_table: batch_table})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] loss: %.8f, re_loss: %.8f, kl: %.8f, cohesive: %.8f, "
                      % (epoch, idx, self.num_batches, loss, re_loss, kl_loss, cohesive_loss))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model (every 10 epoch)
            if epoch % 10 == 0:
                self.save_ckp(self.checkpoint_dir, counter)

        # save model for final step
        self.save_ckp(self.checkpoint_dir, counter)

    def predict(self, input_vector):
        # restore check-point if it exits
        could_load, _ = self.load_ckp(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        predict_mu = self.sess.run(
            self.mu, feed_dict={self.inputs: input_vector})
        return predict_mu

    def visualize_results(self, epoch):
        pass

    def save_ckp(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load_ckp(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
            
    def all_cpk_paths(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        return ckpt.all_model_checkpoint_paths

    def eval(self, input_vector, n):

        all_ckp_paths = self.all_cpk_paths()

        cpk_path = all_ckp_paths[int(n)]
        self.saver.restore(self.sess, os.path.join(
            self.checkpoint_dir, cpk_path))

        predict_mu = self.sess.run(
            self.mu, feed_dict={self.inputs: input_vector})

        return predict_mu
