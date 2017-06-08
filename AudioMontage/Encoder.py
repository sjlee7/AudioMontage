import tensorflow as tf
import numpy as np


from Utils import FLAGS



class Encoder(object):

	def __init__(self, name, info):
		self.n = FLAGS.hidden_n # Hidden number of first dense ouput layer
		self.name = name
		self.info = info

	def encode(self, image, reuse=False):
		self.reuse = reuse        
		with tf.variable_scope(self.name, reuse=self.reuse):
			
			self.image = image # Get image

			print("image_info : ", self.image)

			self.outdim_info = self.info['outdim']
			self.kernel_info = self.info['kernel']
			self.stride_info = self.info['stride']

			conv_layer_0 = tf.layers.conv2d(self.image, \
											self.outdim_info[0], \
											self.kernel_info[0], \
											self.stride_info[0], \
											padding = "same", \
											activation = tf.nn.elu,\
											reuse = self.reuse,\
											)  


			print("conv_layer_0 info : ", conv_layer_0)

			conv_layer_1 = tf.layers.conv2d(conv_layer_0, \
											self.outdim_info[1], \
											self.kernel_info[1], \
											self.stride_info[1], \
											padding = "same", \
											activation = tf.nn.elu,\
											reuse = self.reuse,\
											)  

			print("conv_layer_1 info : ", conv_layer_1)

			conv_layer_2 = tf.layers.conv2d(conv_layer_1, \
											self.outdim_info[2], \
											self.kernel_info[2], \
											self.stride_info[2], \
											padding = "same", \
											activation = tf.nn.elu,\
											reuse = self.reuse,\
											)  

			print("conv_layer_2 info : ", conv_layer_2)

			subsample_layer_1 = tf.layers.conv2d(conv_layer_2, \
											 	 self.outdim_info[3], \
												 self.kernel_info[3], \
												 self.stride_info[3], \
												 padding = "same", \
												 activation = tf.nn.elu,\
												 reuse = self.reuse,\
												 )  


			print("subsample_layer_1 info : ", subsample_layer_1)

			conv_layer_3 = tf.layers.conv2d(subsample_layer_1, \
											self.outdim_info[4], \
											self.kernel_info[4], \
											self.stride_info[4], \
											padding = "same", \
											activation = tf.nn.elu,\
											reuse = self.reuse,\
											)  


			print("conv_layer_3 info : ", conv_layer_3)

			conv_layer_4 = tf.layers.conv2d(conv_layer_3, \
											self.outdim_info[5], \
											self.kernel_info[5], \
											self.stride_info[5], \
											padding = "same", \
											activation = tf.nn.elu,\
											reuse = self.reuse,\
											)  

			print("conv_layer_4 info : ", conv_layer_4)

			subsample_layer_2 = tf.layers.conv2d(conv_layer_4, \
											 	 self.outdim_info[6], \
												 self.kernel_info[6], \
												 self.stride_info[6], \
												 padding = "same", \
												 activation = tf.nn.elu,\
												 reuse = self.reuse,\
												 )  


			print("subsample_layer_2 info : ", subsample_layer_2)

			conv_layer_5 = tf.layers.conv2d(subsample_layer_2, \
											self.outdim_info[7], \
											self.kernel_info[7], \
											self.stride_info[7], \
											padding = "same", \
											activation = tf.nn.elu,\
											reuse = self.reuse,\
											)  

			print("conv_layer_4 info : ", conv_layer_4)

			conv_layer_6 = tf.layers.conv2d(conv_layer_5, \
											self.outdim_info[8], \
											self.kernel_info[8], \
											self.stride_info[8], \
											padding = "same", \
											activation = tf.nn.elu,\
											reuse = self.reuse,\
											)  

			print("conv_layer_6 info : ", conv_layer_6)

			reshaped_vector = tf.reshape(conv_layer_6, [FLAGS.bn, 8*8*3*self.n])

			embedding_vector = tf.layers.dense(reshaped_vector, 8*8, reuse = self.reuse)

			return embedding_vector
