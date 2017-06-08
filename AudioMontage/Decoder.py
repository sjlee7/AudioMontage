import tensorflow as tf
import numpy as np


from Utils import FLAGS



class Decoder(object):

    def __init__(self, name, info):
        self.n = FLAGS.hidden_n # hidden number of first dense ouput layer
        self.name = name
        self.info = info

    def decode(self, h, reuse=False):
        self.reuse = reuse
        with tf.variable_scope(self.name, reuse=self.reuse):

            self.h = h

            self.outdim_info = self.info['outdim']
            self.kernel_info = self.info['kernel']
            self.stride_info = self.info['stride']
            
            dense_layer = tf.layers.dense(self.h, 8*8*self.n, reuse = self.reuse)


            h_0 = tf.reshape(dense_layer, [FLAGS.bn, 8, 8, self.n])


            conv_layer_1 = tf.layers.conv2d(h_0, self.outdim_info[0], self.kernel_info[0], self.stride_info[0], padding = "same", activation = tf.nn.elu, reuse = self.reuse)  
            conv_layer_2 = tf.layers.conv2d(conv_layer_1, self.outdim_info[1], self.kernel_info[1], self.stride_info[1], padding = "same", activation = tf.nn.elu, reuse = self.reuse)  

            h_1 = tf.image.resize_nearest_neighbor(h_0, [16,16]) # first skip connection
            resized_conv_layer_2 = tf.image.resize_nearest_neighbor(conv_layer_2, [16,16]) 
            upsample_layer_1 = tf.concat([h_1, resized_conv_layer_2], axis=3)


            conv_layer_3 = tf.layers.conv2d(upsample_layer_1, self.outdim_info[2], self.kernel_info[2], self.stride_info[2], padding = "same", activation = tf.nn.elu, reuse = self.reuse)
            conv_layer_4 = tf.layers.conv2d(conv_layer_3, self.outdim_info[3], self.kernel_info[3], self.stride_info[3],padding = "same", activation = tf.nn.elu, reuse = self.reuse)


            h_2 = tf.image.resize_nearest_neighbor(h_0, [32,32]) # second skip connection
            resized_conv_layer_4 = tf.image.resize_nearest_neighbor(conv_layer_4, [32,32]) 
            upsample_layer_2 = tf.concat([h_2, resized_conv_layer_4], axis=3)


            conv_layer_5 = tf.layers.conv2d(upsample_layer_2, self.outdim_info[4], self.kernel_info[4], self.stride_info[4], padding = "same", activation = tf.nn.elu, reuse = self.reuse)
            conv_layer_6 = tf.layers.conv2d(conv_layer_5, self.outdim_info[5], self.kernel_info[5], self.stride_info[5], padding = "same", activation = tf.nn.elu, reuse = self.reuse)


            output_img = tf.layers.conv2d(conv_layer_6, self.outdim_info[6], self.kernel_info[6], self.stride_info[6], padding = "same", activation = tf.nn.elu, reuse = self.reuse)

            return output_img