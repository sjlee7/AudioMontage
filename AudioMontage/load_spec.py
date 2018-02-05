"""
Python2 & Python3 
Version Compatible
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Utils import generate_z, FLAGS, get_loss, norm_img, denorm_img
from Loader import Spectrogram_Loader
from PIL import Image

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd
from spec_encoder import spec_encoder
import librosa
from scipy.spatial.distance import cosine
from sklearn.preprocessing import scale


def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
	return tf.Variable(tf.zeros(shape))

n = FLAGS.hidden_n

					
# Encoder_infos = {
# 					"outdim":[n,n,2*n,2*n,2*n,3*n,3*n, 3*n, 3*n, 4*n, 4*n, 4*n, 4*n, 4*n, 4*n, 6*n, 6*n, 6*n, 8*n, 8*n, 8*n],\
# 					"kernel":[ \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 								[3, 3], \
# 							], \
# 					"stride":[ \
# 								[1, 1], \
# 								[1, 1], \
# 								[1, 1], \
# 								[4, 4], \
# 								[1, 1], \
# 								[1, 1], \
# 								[2, 2], \
# 								[1, 1], \
# 								[1, 1], \
# 								[2, 2], \
# 								[1, 1], \
# 								[1, 1], \
# 								[2, 2], \
# 								[1, 1], \
# 								[1, 1], \
# 								[2, 2], \
# 								[1, 1], \
# 								[1, 1], \
# 								[3, 3], \
# 								[1, 1], \
# 								[1, 1], \
# 							], \
# 				} 

# E = spec_encoder("Encoder", Encoder_infos)

weights = {
        'wconv1':init_weights([4, 157, 128]),
        'wconv2':init_weights([4, 128, 256]),
        'wconv3':init_weights([4, 256, 256]),
        'wconv4':init_weights([4, 256, 512]),
        'wconv5':init_weights([4, 512, 512]),
        # 'wconv5':init_weights([4, 128, 512, 512]),
        # 'wconv6':init_weights([4, 128, 192, 256]),
        # 'wconv7':init_weights([4, 128, 256, 512]),
        'bconv1':init_biases([128]),
        'bconv2':init_biases([256]),
        'bconv3':init_biases([256]),
        'bconv4':init_biases([512]),
        'bconv5':init_biases([512]),
        # 'bconv5':init_biases([256]),
        # 'bconv6':init_biases([256]),
        # 'bconv7':init_biases([512]),
        'woutput':init_weights([512*3, FLAGS.z_size]),
        'boutput':init_biases([FLAGS.z_size])
        }

handle_male = open('./final_z_male/recon_z_male.pickle', 'rb')
test_male = pickle.load(handle_male)

handle_fmale = open('./final_z_male/recon_z_fmale.pickle', 'rb')
test_fmale = pickle.load(handle_fmale)

zz = open('./z_5000.pickle', 'rb')
zz = pickle.load(zz)

# def denorm_z(y):
# 	y_mean = y.mean(axis=1).reshape(1,FLAGS.bn)
# 	y_std = y.std(axis=1).reshape(1, FLAGS.bn)
# 	for i in range(len(y[0])):
# 		y[:,i] = y_std*y[:,i] - y_mean
# 	return y

# def norm_z(y):

# 	y_train = np.concatenate((y_train1[:,:], y_train2[:,:]), axis=0)
# 	y_mean = y_train.mean(axis=1).reshape(1, 32)
# 	y_std = y_train.std(axis=1).reshape(1, 32)
# 	for j in range(100):
# 		y_train[:,j] = (y_train[:,j] - y_mean)/y_std




def batch_norm(x, n_out, phase_train, scope='bn'):
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moment')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed

def cnn_1d(melspectrogram, weights, keep_prob, phase_train):

	# melspectrogram = tf.transpose(melspectrogram, perm=[0,2,1])
	print (melspectrogram.shape)

	# x = tf.reshape(melspectrogram, [-1, 1, 157, 128])
	# x = batch_norm(melspectrogram, 157, phase_train)
	
	x = tf.reshape(melspectrogram, [-1, 128, 157])
	conv2_1 = tf.add(tf.layers.conv1d(x, filters=128, kernel_size=4, padding='SAME'), weights['bconv1'])
	conv2_1 = tf.nn.relu(batch_norm(conv2_1, 128, phase_train))
	mpool_1 = tf.layers.max_pooling1d(conv2_1, pool_size=4, strides=4)
	dropout_1 = tf.nn.dropout(mpool_1, keep_prob)
	print (dropout_1.shape)

	conv2_2 = tf.add(tf.layers.conv1d(dropout_1, filters=256, kernel_size=4, padding='SAME'), weights['bconv2'])
	conv2_2 = tf.nn.relu(batch_norm(conv2_2, 256, phase_train))
	mpool_2 = tf.layers.max_pooling1d(conv2_2, pool_size=4,  strides=4)
	dropout_2 = tf.nn.dropout(mpool_2, keep_prob)
	print (dropout_2.shape)

	conv2_3 = tf.add(tf.layers.conv1d(dropout_2, filters=256, kernel_size=4, padding='SAME'), weights['bconv3'])
	conv2_3 = tf.nn.relu(batch_norm(conv2_3, 256, phase_train))
	mpool_3 = tf.layers.max_pooling1d(conv2_3, pool_size=2,  strides=2)
	dropout_3 = tf.nn.dropout(mpool_3, keep_prob)
	print (dropout_3.shape)

	conv2_4 = tf.add(tf.layers.conv1d(dropout_3, filters=512, kernel_size=4, padding='SAME'), weights['bconv4'])
	conv2_4 = tf.nn.relu(batch_norm(conv2_4, 512, phase_train))
	mpool_4 = tf.layers.max_pooling1d(conv2_4, pool_size=2,  strides=2)
	dropout_4 = tf.nn.dropout(mpool_4, keep_prob)
	print (dropout_4.shape)

	# flat = tf.reshape(dropout_4, [-1, weights['woutput'].get_shape().as_list()[0]])
	conv2_5 = tf.add(tf.layers.conv1d(dropout_4, filters=512, kernel_size=4, padding='SAME'), weights['bconv5'])
	conv2_5 = tf.nn.relu(batch_norm(conv2_5, 512, phase_train))
	global_pool1 = tf.layers.max_pooling1d(conv2_5, pool_size=2,  strides=2)
	global_pool2 = tf.layers.average_pooling1d(conv2_5, pool_size=2,  strides=2)
	global_pool3 = tf.pow(tf.layers.average_pooling1d(tf.pow(conv2_5, 2), pool_size=2, strides=2), 0.5)
	print(global_pool3.shape)
	global_pool = tf.concat([global_pool1, global_pool2, global_pool3], axis=2)
	print(global_pool.shape)

	out1 = tf.reshape(global_pool, [-1, 1536])
	out1 = tf.layers.dense(inputs=out1, units=1536, activation=None)
	dout_out1 = tf.nn.dropout(out1, keep_prob)		
	print(dout_out1.shape)
	# out2 = tf.reshape(dout_out1, [-1, 1536])
	# out2 = tf.layers.dense(inputs=out2, units=1536, activation=None)
	# dout_out2 = tf.nn.dropout(out2, keep_prob)		
	# print(dout_out2.shape)
	out3 = tf.add(tf.matmul(dout_out1, weights['woutput']),weights['boutput'])
	print(out3.shape)	
	# out3 = tf.layers.dense(inputs=out2, units=2048, activation=tf.nn.relu)

	# x = tf.reshape(melspectrogram, [-1, 128, 157])
	# conv1 = tf.add(tf.nn.conv1d(x, filters=[4,128], stride=1, padding='SAME'))
	# conv1 = tf.nn.relu(conv1)
	# mpool1 = tf.nn.max_pool(conv1, ksize=[4,2], strides=1, padding='VALID')

	# conv2 = tf.add(tf.nn.conv1d(mpool1, filters=[4,256], stride=1, padding='SAME'))
	# conv2 = tf.nn.relu(conv2)
	# mpool2 = tf.nn.max_pool(conv2, ksize=[2,1], stride=1, padding='VALID')

	# conv3 = tf.add(tf.nn.conv1d(mpool2, filters=[4,256], stride=1, padding='SAME'))
	# conv3 = tf.nn.relu(conv3)
	# mpool3 = tf.nn.max_pool(conv3, ksize=[2,2], stride=1, padding='VALID')
	
	# flat = tf.reshape(mpool3, [-1, weights['woutput'].get_shape().as_list()[0]])
	# p_y_X = tf.layers.dense(inputs=flat, units = 512, activation=None)
	# p_y_X = tf.layers.dense(inputs=p_y_X, units = 512, activation=None)	
	# # p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(p_y_X,weights['woutput']),weights['boutput']))
	# p_y_X = tf.add(tf.matmul(out3, weights['woutput']),weights['boutput'])
	return out3





def cnn_withoutb(melspectrogram, weights, keep_prob):

	melspectrogram = tf.transpose(melspectrogram, perm=[0,2,1,3])
	print (melspectrogram.shape)
	# x = tf.reshape(melspectrogram, [-1, 1, 157, 128])
	# x = batch_norm(melspectrogram, 157, phase_train)
	
	x = tf.reshape(melspectrogram, [-1, 157, 128, 1])

	conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
	conv2_1 = tf.nn.relu(conv2_1)
	conv2_1 = tf.add(tf.nn.conv2d(conv2_1, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
	conv2_1 = tf.nn.relu(conv2_1)
	mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='VALID')
	dropout_1 = tf.nn.dropout(mpool_1, keep_prob)
	print (dropout_1.shape)

	conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
	conv2_2 = tf.nn.relu(conv2_2)
	conv2_2 = tf.add(tf.nn.conv2d(conv2_2, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
	conv2_2 = tf.nn.relu(conv2_2)	
	mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
	dropout_2 = tf.nn.dropout(mpool_2, keep_prob)
	print (dropout_2.shape)

	conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
	conv2_3 = tf.nn.relu(conv2_3)
	conv2_3 = tf.add(tf.nn.conv2d(conv2_3, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
	conv2_3 = tf.nn.relu(conv2_3)	
	mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
	dropout_3 = tf.nn.dropout(mpool_3, keep_prob)
	print (dropout_3.shape)

	conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
	conv2_4 = tf.nn.relu(conv2_4)
	conv2_4 = tf.add(tf.nn.conv2d(conv2_4, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
	conv2_4 = tf.nn.relu(conv2_4)
	mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_4 = tf.nn.dropout(mpool_4, keep_prob)
	print (dropout_4.shape)

	conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
	conv2_5 = tf.nn.relu(conv2_5)
	conv2_5 = tf.add(tf.nn.conv2d(conv2_5, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
	conv2_5 = tf.nn.relu(conv2_5)
	mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_5 = tf.nn.dropout(mpool_5, keep_prob)
	print (dropout_5.shape)

	conv2_6 = tf.add(tf.nn.conv2d(dropout_5, weights['wconv6'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv6'])
	conv2_6 = tf.nn.relu(conv2_6)	
	conv2_6 = tf.add(tf.nn.conv2d(conv2_6, weights['wconv6'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv6'])
	conv2_6 = tf.nn.relu(conv2_6)
	mpool_6 = tf.nn.max_pool(conv2_6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_6 = tf.nn.dropout(mpool_6, keep_prob)
	print (dropout_6.shape)

	conv2_7 = tf.add(tf.nn.conv2d(dropout_6, weights['wconv7'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv7'])
	conv2_7 = tf.nn.relu(conv2_7)
	conv2_7 = tf.add(tf.nn.conv2d(conv2_7, weights['wconv7'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv7'])
	conv2_7 = tf.nn.relu(conv2_7)
	mpool_7 = tf.nn.max_pool(conv2_7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_7 = tf.nn.dropout(mpool_7, keep_prob)
	print (dropout_7.shape)

	flat = tf.reshape(dropout_7, [-1, weights['woutput'].get_shape().as_list()[0]])
	p_y_X = tf.layers.dense(inputs=flat, units = 512, activation=None)
	p_y_X = tf.layers.dense(inputs=p_y_X, units = 512, activation=None)	
	# p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(p_y_X,weights['woutput']),weights['boutput']))
	p_y_X = tf.add(tf.matmul(p_y_X, weights['woutput']),weights['boutput'])
	return p_y_X

def cnn(melspectrogram, weights, phase_train, keep_prob):

	melspectrogram = tf.transpose(melspectrogram, perm=[0,2,1,3])
	print (melspectrogram.shape)
	x = tf.reshape(melspectrogram, [-1, 1, 157, 128])
	x = batch_norm(melspectrogram, 157, phase_train)

	x = tf.reshape(melspectrogram, [-1, 157, 128, 1])

	conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
	conv2_1 = tf.nn.relu(batch_norm(conv2_1, 64, phase_train))
	# conv2_1 = tf.nn.relu(conv2_1)
	mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_1 = tf.nn.dropout(mpool_1, keep_prob)
	print (dropout_1.shape)

	conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
	conv2_2 = tf.nn.relu(batch_norm(conv2_2, 128, phase_train))
	# conv2_2 = tf.nn.relu(conv2_2)
	mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_2 = tf.nn.dropout(mpool_2, keep_prob)
	print (dropout_2.shape)

	conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
	conv2_3 = tf.nn.relu(batch_norm(conv2_3, 128, phase_train))
	# conv2_3 = tf.nn.relu(conv2_3)
	mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_3 = tf.nn.dropout(mpool_3, keep_prob)
	print (dropout_3.shape)

	conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
	conv2_4 = tf.nn.relu(batch_norm(conv2_4, 192, phase_train))
	# conv2_4 = tf.nn.relu(conv2_4)
	mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_4 = tf.nn.dropout(mpool_4, keep_prob)
	print (dropout_4.shape)

	conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
	conv2_5 = tf.nn.relu(batch_norm(conv2_5, 192, phase_train))
	# conv2_5 = tf.nn.relu(conv2_5)
	mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_5 = tf.nn.dropout(mpool_5, keep_prob)
	print (dropout_5.shape)

	conv2_6 = tf.add(tf.nn.conv2d(dropout_5, weights['wconv6'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv6'])
	conv2_6 = tf.nn.relu(batch_norm(conv2_6, 256, phase_train))
	# conv2_6 = tf.nn.relu(conv2_6)	
	mpool_6 = tf.nn.max_pool(conv2_6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_6 = tf.nn.dropout(mpool_6, keep_prob)
	print (dropout_6.shape)

	conv2_7 = tf.add(tf.nn.conv2d(dropout_6, weights['wconv7'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv7'])
	conv2_7 = tf.nn.relu(batch_norm(conv2_7, 100, phase_train))
	# conv2_7 = tf.nn.relu(conv2_7)
	mpool_7 = tf.nn.max_pool(conv2_7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_7 = tf.nn.dropout(mpool_7, keep_prob)
	print (dropout_7.shape)

	flat = tf.reshape(dropout_7, [-1, weights['woutput'].get_shape().as_list()[0]])
	# p_y_X = tf.layers.dense(inputs=flat, units = 100, activation=None)

	# p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(p_y_X,weights['woutput']),weights['boutput']))
	p_y_X = tf.add(tf.matmul(flat,weights['woutput']),weights['boutput'])
	return p_y_X


def sort_result(tags, preds, r):
    # result = zip(tags, preds)
    # argx = preds[1].argsort()
    print(y)
    # print(tags)
    print(preds)
    # print(result)
    # sorted_result = result[argx]
    sorted_result = sorted(result, key=(lambda x: x[1]), reverse=True)
    print (sorted_result)
    return [(name, '%5.3f' % score) for name, score in sorted_result]








def main(_):

	"""
	Run main function
	"""

	#___________________________________________Layer info_____________________________________________________

	"""
	Prepare Image Loader
	"""
	import pickle 
	handle_male = open('./z_audio.pickle', 'rb')
	test = pickle.load(handle_male)

	key_train = []
	key_test = []

	for k in test.keys():
	    if 'fask' not in k and 'fcft' not in k and 'mdbb' not in k and 'mwbt' not in k:
	        key_train.append(k)
	    else:
	        key_test.append(k)

	Xtrain = []
	Xtest = []
	ytrain = []
	ytest = []

	for i, j in zip(test.values(), test.keys()):
	    if j in key_train:
	        Xtrain.append(i.values()[0])
	        ytrain.append(i.values()[1])
	    else:
	        Xtest.append(i.values()[0])
	        ytest.append(i.values()[1])       



	# root1 = "./melspectrograms_female/"
	# root2 = "./melspectrograms_male/"
	
	batch_size = 16
	scale_size = [128, 157]
	data_format = "NHWC"
	loader1 = Spectrogram_Loader(root1, batch_size, scale_size, data_format)
	loader2 = Spectrogram_Loader(root2, batch_size, scale_size, data_format)
	loader3 = Spectrogram_Loader(root2, 5000, scale_size, data_format)
	loader4 = Spectrogram_Loader(root2, 5000, scale_size, data_format)
	
	spec_holder = tf.placeholder(tf.float32, shape=(None, 128, 157))#FLAGS.bn
	# spec_holder = tf.placeholder(tf.float32, shape=(None, 128, 157, 1))#FLAGS.bn
	y = tf.placeholder(tf.float32, shape = (None, FLAGS.z_size)) #FLAGS.bn
	# lr = tf.Variable(FLAGS.lr_reverse, name='lr')
	lr = tf.placeholder(tf.float32)
	lam1 = tf.placeholder(tf.float32)
	lam2 = tf.placeholder(tf.float32)

	phase_train = tf.placeholder(tf.bool, name='phase_train')

	y_ = cnn_1d(spec_holder, weights, keep_prob=0.5, phase_train=phase_train)
	# y_ = cnn_withoutb(spec_holder, weights, keep_prob=0.7)
	# y_ = cnn(spec_holder, weights, phase_train, keep_prob=0.7)
	# y_ = E.encode(spec_holder)

	# pre_mean, pre_var = tf.nn.moments(y_, axes=[1])

	# predict_op = tf.reshape(tf.sqrt(pre_var), [FLAGS.bn,-1])*y_ - tf.reshape(pre_mean, [FLAGS.bn,-1])
	predict_op = y_
	theta = tf.Variable(tf.random_normal([128, 157, 1, FLAGS.z_size], stddev=0.01), name="Theta")
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
	L1_norm = tf.reduce_mean(tf.abs(tf.subtract(y, predict_op)))

	# L1_norm = tf.reduce_sum(tf.pow(predict_op - y, 2))/(2 * 128) #L2 loss
	# +tf.constant(0.1)*tf.reduce_mean(tf.abs(theta))
	# L1_norm = tf.nn.l1_loss(predict_op - y)
	rweights = tf.Variable(tf.truncated_normal([128*157, 100]))
	regularizer = tf.nn.l2_loss(rweights)
	L2_norm = tf.reduce_sum(tf.square(tf.subtract(y, predict_op))/2)/100
	# + 0.01*regularizer
	y_mean = tf.placeholder(tf.float32, shape = (None, FLAGS.z_size))
	# L2_norm1 = tf.reduce_sum(tf.square(tf.subtract(y, predict_op)))/tf.reduce_sum(tf.square(tf.subtract(y_mean, predict_op)))
	# L2_norm2 = tf.reduce_sum(tf.square(tf.subtract(y, predict_op)))/tf.reduce_sum(tf.square(tf.subtract(y, y_mean)))
 	# L2_norm3 = tf.reduce_sum(tf.square(tf.subtract(y_mean, predict_op)))/tf.reduce_sum(tf.square(tf.subtract(y_mean, y)))
 	# L2_norm_n = tf.reduce_sum(tf.square(tf.subtract(y, predict_op)))
 	# L2_norm = tf.sqrt(L2_norm1) + 0.2*tf.sqrt(L2_norm2)

 	 	# - tf.sqrt(L2_norm3)
 	 # + 0.000001*regularizer

 	# L2_norm = tf.sqrt(L2_norm3)

	# L2_norm = tf.abs(tf.subtract(y, predict_op))/tf.abs(tf.subtract(y, y_mean))
 # 	L2_norm = tf.reduce_sum(L2_norm) 
	# + (tf.reduce_sum(tf.square(tf.subtract(y, predict_op))))*0.000001
	# L2_norm = (tf.reduce_sum(y, predict_op)/tf.abs(tf.reduce_sum((-1)*tf.abs(tf.subtract(y, root_mean)))))

	# L2_norm = (tf.nn.l2_loss(tf.subtract(predict_op,y))+tf.constant(0.1)*tf.nn.l2_loss(theta))
	# L2_norm = tf.nn.l2_loss(tf.subtract(predict_op, y)) + L2regularizationLoss
	# normalize_a = tf.nn.l2_normalize(y,0)        
	# normalize_b = tf.nn.l2_normalize(predict_op,0)
	norm_z = tf.sqrt(tf.reduce_sum(tf.square(y), 1, keep_dims=True))
	normalized_z = y/norm_z
	norm_pre = tf.sqrt(tf.reduce_sum(tf.square(predict_op), 1, keep_dims=True))
	normalized_pre = predict_op/norm_pre
	cos_loss = tf.contrib.losses.cosine_distance(normalized_z, normalized_pre, dim=0)
	# valid_z = tf.nn.embedding_lookup(normalized_z, predict_op)
	# cos_loss = tf.matmul(valid_z, normalized_z, transpose_b = True)
	# Cos_loss = tf.abs(tf.reduce_sum(tf.multiply(normalize_a,normalize_b)))
	# Cos_loss = tf.abs(tf.contrib.losses.cosine_distance(y, predict_op, dim=0))
	# com_loss = 2/(1/(10*L1_norm) + 1/Cos_loss)
	com_loss = L2_norm*0.8 +  0.2*cos_loss

	correct_pred = tf.equal(tf.argmax(predict_op, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# train_op = tf.train.RMSPropOptimizer(learning_rate = lr).minimize(com_loss)
	train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(com_loss)

	# predict_op = y_

	# L1_norm = tf.reduce_mean(tf.abs(tf.subtract(predict_op, y)))
	# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# y_train1 = np.array(test_fmale).reshape(10000, 64)
	# np.random.shuffle(y_train1)
	# y_train1 = y_train1[:5000,:]
	# y_train2 = np.array(test_male).reshape(10000, 64)
	# np.random.shuffle(y_train2)
	# y_train2 = y_train2[:5000,:]

	# y_train1 = np.ones(10000).tolist()
	# y_train1.extend(np.zeros(10000).tolist())
	# y_train2 = np.zeros(10000).tolist()
	# y_train2.extend(np.ones(10000).tolist())
	# y_train = np.array(pd.concat([pd.DataFrame(y_train1), pd.DataFrame(y_train2)], axis = 1))
	# y_train1 = np.array(y_train[:10000,:])
	# y_train2 = np.array(y_train[10000:20000,:])
	# print(y_train.shape)
	saver = tf.train.Saver()

	NUM_THREADS=1
	config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
						intra_op_parallelism_threads=NUM_THREADS,\
						allow_soft_placement=True,\
						device_count = {'CPU': 1},\
						)

	with tf.Session(config=config) as sess:
		tf.initialize_all_variables().run()

		coord = tf.train.Coordinator() # Set Coordinator to Manage Queue Runners
		threads = tf.train.start_queue_runners(sess, coord=coord) # Set Threads
		# writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'		

		# y_label1 = loader3.get_label_from_loader(sess)
		# y_label2 = loader4.get_label_from_loader(sess)
		# print(zz.shape)
		# y_label = np.concatenate((y_label1[:,:], y_label2[:,:]), axis=0)
		y_av = np.average(zz, axis=0).reshape(1,100)


		for i in range(2000):
			# training_batch = zip(range(0, 5000, batch_size), range(batch_size, 5000, batch_size))
			# for start, end in training_batch:
				# X_train = loader1.get_spectrogram_from_loader(sess)
			X_train1 = loader1.get_spectrogram_from_loader(sess)
			X_train2 = loader2.get_spectrogram_from_loader(sess)
				# X_train1 = loader1.get_spectrogram_from_loader(sess)
				# X_train2 = loader2.get_spectrogram_from_loader(sess)
			
			y_train1 = loader1.get_label_from_loader(sess)
			y_train2 = loader2.get_label_from_loader(sess)

				# X_train = np.concatenate((X_train1[:,:,:,:], X_train2[:,:,:,:]), axis=0)
			X_train = np.concatenate((X_train1[:,:,:,:], X_train2[:,:,:,:]), axis=0).reshape(batch_size*2, 128, 157)
			y_train = np.concatenate((y_train1[:,:], y_train2[:,:]), axis=0)

			# X_mean = X_train1.mean(axis=1).reshape(1, FLAGS.bn, 157, 1)
			# X_std = X_train1.std(axis=1).reshape(1, FLAGS.bn, 157, 1)
			# for j in range(128):
			# 	X_train1[:,j,:,:] = (X_train1[:,j,:,:] - X_mean)/X_std

			# X_mean = X_train2.mean(axis=1).reshape(1, FLAGS.bn, 157, 1)
			# X_std = X_train2.std(axis=1).reshape(1, FLAGS.bn, 157, 1)
			# for j in range(128):
			# 	X_train2[:,j,:,:] = (X_train2[:,j,:,:] - X_mean)/X_std

				# y_train = np.concatenate((y_train1[start:end][:,:], y_train2[start:end][:,:]), axis=0)
			# y_mean = y_train1.mean(axis=1).reshape(1, FLAGS.bn)
			# y_std = y_train1.std(axis=1).reshape(1, FLAGS.bn)
			# for j in range(FLAGS.z_size):
			# 	y_train1[:,j] = (y_train1[:,j] - y_mean)/y_std

			# y_mean = y_train2.mean(axis=1).reshape(1, FLAGS.bn)
			# y_std = y_train2.std(axis=1).reshape(1, FLAGS.bn)
			# for j in range(FLAGS.z_size):
			# 	y_train2[:,j] = (y_train2[:,j] - y_mean)/y_std

			# y_std*x-y_mean
			# X_row_sums = X_train.sum(axis=1, keepdims=True)
			# X_std = X_train.std(axis=(1), keepdims=True)
			# X_train = (X_train - X_mean)/X_std
			# y_train = np.concatenate((y_train1[start:end][:,:], y_train2[start:end][:,:]), axis=0)
			train_input_dict = {spec_holder: X_train, 
								y: y_train,
								y_mean: y_av,
								lr: 0.0011,
								lam1: 0.5,
								lam2: 0.5,
								phase_train: True}
			sess.run(train_op, feed_dict=train_input_dict)
			# print(sess.run(y_mean), feed_dict={y_mean: y_av})
			loss1, loss2, cosine_loss = sess.run([L1_norm, L2_norm, cos_loss], feed_dict=train_input_dict)
			print('iter : ', i, 'loss1 : ', loss1, 'loss2 : ', loss2, 'cos_loss : ', cosine_loss)

			# loss1, loss2, cosine_loss, l2norm3 = sess.run([L2_norm1, L2_norm2, cos_loss, L2_norm3], feed_dict=train_input_dict)
			# print('iter : ', i, 'loss1 : ', loss1, 'loss2 : ', loss2, 'loss3',l2norm3,'cos_loss : ', cosine_loss)
			if i % 200 ==0:
				print("z : ", y_train)
				print("pre : ", sess.run(predict_op, feed_dict=train_input_dict))

			# train_input_dict = {spec_holder: X_train1, 
			# 					y: y_train1,
			# 					lr: 0.0007,
			# 					lam1: 0.5,
			# 					lam2: 0.5,
			# 					phase_train: True}
			# sess.run(train_op, feed_dict=train_input_dict)
			# loss1, loss2, cosine_loss = sess.run([L1_norm, L2_norm, cos_loss], feed_dict=train_input_dict)
			# print('iter : ', i, 'loss1 : ', loss1, 'loss2 : ', loss2, 'cos_loss : ', cosine_loss)
			# if i %  2000==0:
			# 	print ("z : ",y_train1)		
			# 	print ("pre : ",sess.run(predict_op, feed_dict=train_input_dict))


			# train_input_dict = {spec_holder: X_train2, 
			# 					y: y_train2,
			# 					lr: 0.0007,
			# 					lam1: 0.5,
			# 					lam2: 0.5,
			# 					phase_train: True}
			# sess.run(train_op, feed_dict=train_input_dict)
			# loss1, loss2, cosine_loss = sess.run([L1_norm, L2_norm, cos_loss], feed_dict=train_input_dict)
			# print('iter : ', i, 'loss1 : ', loss1, 'loss2 : ', loss2, 'cos_loss : ', cosine_loss)			
			# 	# print ("x : ",X_train)
			# if i % 2000 ==0:
			# 	print ("z : ",y_train2)		
			# 	print ("pre : ",sess.run(predict_op, feed_dict=train_input_dict))
					
					# X_train = loader2.get_spectrogram_from_loader(sess)
					# X_train2 = loader2.get_spectrogram_from_loader(sess)
					# train_input_dict = {spec_holder: X_train,
					# 					y: y_train2[start:end],
					# 					lr:0.005,
					# 					phase_train: True}
					# sess.run(train_op, feed_dict=train_input_dict)
					# loss = sess.run([L1_norm], feed_dict=train_input_dict)
					# print('iter : ', start, 'accuracy : ', loss, 'l1 norm : ', loss)
			if i % 200 ==0:
				save_path = saver.save(sess, 'Cnn_Check_Point/model_t')

			# test_pre = []
			# mean_test_loss = []
	  #       y, sr = librosa.core.load("./hyeongseoksample.wav", sr = 16000, mono=True, offset=0 , duration=2.5)
	  #       D = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=int(1024/4), fmax=None)
	  #       D = np.abs(D).reshape(1, 128, 157, 1)
	  #       D = np.concatenate((D[:,:,:,:], D[:,:,:,:]), axis=0)
	  #       sample_test_feed_dict = {spec_holder:D,
	  #       							lr:0.0001,
	  #       							phase_train:True}
	  #       sample_t_pred = sess.run(predict_op, feed_dict=sample_test_feed_dict)
	  #       test_pre.append(sample_t_pred)
	  #       sample_t_pred = sess.run(predict_op, feed_dict=sample_test_feed_dict)
	  #       test_pre.append(sample_t_pred)
	  #       sample_t_pred = sess.run(predict_op, feed_dict=sample_test_feed_dict)
	  #       test_pre.append(sample_t_pred)
	  #       file_name = "./pre_zsample.pickle"
	  #       with open(file_name, 'wb') as handle:
	  #       	pickle.dump(test_pre, handle, protocol=pickle.HIGHEST_PROTOCOL)


			# for start, end in zip(range(0, 1000, 16), range(16, 1000, 16)):
			# 	X_test1 = loader1.get_spectrogram_from_loader(sess)
			# 	X_test2 = loader2.get_spectrogram_from_loader(sess)
			# 	X_test = np.concatenate((X_test1[:,:,:,:], X_test2[:,:,:,:]), axis=0)
			# 	y_test = np.concatenate((y_train1[4000+start:4000+end][:,:], y_train2[4000+start:4000+end][:,:]),axis=0)
			# 	test_input_dict = {spec_holder: X_test, 
			# 						y: y_test,
			# 						lr: 0.0001,
			# 						phase_train: True}
			# 	t_pre, loss1 = sess.run([predict_op, L1_norm], feed_dict=test_input_dict)

			# 	# test_pre.extend(t_pre)
			# 	# mean_test_loss.append(loss1)
			# 	# print('iter : ', start, 'test loss1 : ', loss1)
			# 	file_name = "./pre_z.pickle"
			# 	with open(file_name, 'wb') as handle:
			# 		pickle.dump(test_pre, handle, protocol=pickle.HIGHEST_PROTOCOL)
			# save_path = saver.save(sess, 'model_f')

# 	"""
# 	Make Saving Directories
# 	"""
# 	os.makedirs("./Check_Point", exist_ok=True)
# 	os.makedirs("./Generator_Check_Point", exist_ok=True)
# 	os.makedirs("./logs", exist_ok=True) # make logs directories to save summaries
# 	os.makedirs("./Real_Images", exist_ok=True)
# 	os.makedirs("./Generated_Images", exist_ok=True)
# 	os.makedirs("./Decoded_Generated_Images", exist_ok=True)




# 	#----------------------------------------------------------------------------------------------------



# 	#____________________________________Model composition________________________________________

# 	k = tf.Variable(0.0, name = "k_t", trainable = False, dtype = tf.float32) #init value of k_t = 0
# 	lr_g = tf.Variable(FLAGS.lr_G, name='lr_g')
# 	lr_d = tf.Variable(FLAGS.lr_D, name='lr_d')

# 	lr_g_update = tf.assign(lr_g, lr_g*0.6, name='lr_g_update')
# 	lr_d_update = tf.assign(lr_d, lr_d*0.6, name='lr_d_update')
	
	# mel_spec = loader1.queue # Get image batch tensor
# 	image = norm_img(batch) # Normalize Image
	# print (mel_spec.shape)
# 	#z_G = tf.Variable(tf.random_uniform(shape=(FLAGS.bn,FLAGS.z_size), minval=-1, maxval=1, dtype=tf.float32), trainable=False, name = "z_G") # Sample embedding vector batch from uniform distribution
# 	#z_D = tf.Variable(tf.random_uniform(shape=(FLAGS.bn,FLAGS.z_size), minval=-1, maxval=1, dtype=tf.float32), trainable=False, name = "z_D") # Sample embedding vector batch from uniform distribution
# 	z_G = generate_z("uniform") # Sample embedding vector batch from uniform distribution
# 	z_D = generate_z("uniform") # Sample embedding vector batch from uniform distribution


# 	E = Encoder("Encoder", Encoder_infos)
# 	D = Decoder("Decoder", Decoder_infos)
# 	G = Decoder("Generator", Generator_infos)

# 	#Generator
# 	generated_image = G.decode(z_G)
# 	generated_image_for_disc = G.decode(z_D, reuse = True)


# 	#Discriminator (Auto-Encoder)	

# 	#image <--AutoEncoder--> reconstructed_image_real
# 	embedding_vector_real = E.encode(image)
# 	reconstructed_image_real = D.decode(embedding_vector_real)

# 	#generated_image_for_disc <--AutoEncoder--> reconstructed_image_fake
# 	embedding_vector_fake_for_disc = E.encode(generated_image_for_disc, reuse=True)
# 	reconstructed_image_fake_for_disc = D.decode(embedding_vector_fake_for_disc, reuse=True)

# 	#generated_image <--AutoEncoder--> reconstructed_image_fake
# 	embedding_vector_fake = E.encode(generated_image, reuse=True)
# 	reconstructed_image_fake = D.decode(embedding_vector_fake, reuse=True)


# 	#-----------------------------------------------------------------------------------------------



# 	#_________________________________Loss & Summary_______________________________________________


# 	"""
# 	Define Loss
# 	"""
# 	real_image_loss = get_loss(image, reconstructed_image_real)
# 	generator_loss_for_disc = get_loss(generated_image_for_disc, reconstructed_image_fake_for_disc)
# 	discriminator_loss = real_image_loss - tf.multiply(k, generator_loss_for_disc)

# 	generator_loss = get_loss(generated_image, reconstructed_image_fake)
# 	global_measure = real_image_loss + tf.abs(tf.multiply(FLAGS.gamma,real_image_loss) - generator_loss)


# 	"""
# 	Summaries
# 	"""
# 	tf.summary.scalar('Real image loss', real_image_loss)
# 	tf.summary.scalar('Generator loss for discriminator', generator_loss_for_disc)
# 	tf.summary.scalar('Discriminator loss', discriminator_loss)
# 	tf.summary.scalar('Generator loss', generator_loss)
# 	tf.summary.scalar('Global_Measure', global_measure)
# 	tf.summary.scalar('k_t', k)
	

# 	merged_summary = tf.summary.merge_all() # merege summaries, no more summaries under this line

# 	#-----------------------------------------------------------------------------------------------



# 	#_____________________________________________Train_______________________________________________

# 	discriminator_parameters = []
# 	generator_parameters = []

# 	for v in tf.trainable_variables():
# 		if 'Encoder' in v.name:
# 			discriminator_parameters.append(v)
# 			print("Discriminator parameter : ", v.name)
# 		elif 'Decoder' in v.name:
# 			discriminator_parameters.append(v)
# 			print("Discriminator parameter : ", v.name)			
# 		elif 'Generator' in v.name:
# 			generator_parameters.append(v)
# 			print("Generator parameter : ", v.name)
# 		else:
# 			print("None of Generator and Discriminator parameter : ", v.name)

# 	optimizer_D = tf.train.AdamOptimizer(lr_d,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(discriminator_loss,var_list=discriminator_parameters)
# 	optimizer_G = tf.train.AdamOptimizer(lr_g,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(generator_loss,var_list=generator_parameters)

# 	with tf.control_dependencies([optimizer_D, optimizer_G]):
# 		k_update = tf.assign(k, tf.clip_by_value(k + FLAGS.lamb * (FLAGS.gamma*real_image_loss - generator_loss), 0, 1)) #update k_t

# 	init = tf.global_variables_initializer()	


# 	NUM_THREADS=2
# 	config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
# 						intra_op_parallelism_threads=NUM_THREADS,\
# 						allow_soft_placement=True,\
# 						device_count = {'CPU': 1},\
# 						)

# 	# config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_portion



# 	with tf.Session(config=config) as sess:

# 		sess.run(init) # Initialize Variables

# 		coord = tf.train.Coordinator() # Set Coordinator to Manage Queue Runners
# 		threads = tf.train.start_queue_runners(sess, coord=coord) # Set Threads
# 		writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'		

# #_______________________________Restore____________________________________

# 		saver = tf.train.Saver(max_to_keep=1000)
# 		ckpt = tf.train.get_checkpoint_state(checkpoint_dir="./Check_Point")

# 		# generator_saver = tf.train.Saver(generator_parameters, max_to_keep=1000)
# 		# generator_ckpt = tf.train.get_checkpoint_state(checkpoint_dir="./Generator_Check_Point")

# 		# try :	
# 		# 	if ckpt and ckpt.model_checkpoint_path:
# 		# 		print("check point path : ", ckpt.model_checkpoint_path)
# 		# 		saver.restore(sess, ckpt.model_checkpoint_path)	
# 		# 		print('Restored!')
# 		# except AttributeError:
# 		# 		print("No checkpoint")	

# 		Real_Images = sess.run(denorm_img(image))
# 		save_image(Real_Images, '{}.png'.format("./Real_Images/Real_Image"))

# #---------------------------------------------------------------------------
# 		for t in range(FLAGS.iteration): # Mini-Batch Iteration Loop

# 			if coord.should_stop():
# 				break
			
# 			_, _, l_D, l_G, l_Global, k_t = sess.run([\
# 													optimizer_D,\
# 													optimizer_G,\
# 													discriminator_loss,\
# 													generator_loss,\
# 													global_measure,\
# 													k_update,\
# 											   		])

# 			print(
# 				 " Step : {}".format(t),
# 				 " Global measure of convergence : {}".format(l_Global),
# 				 " Generator Loss : {}".format(l_G),
# 				 " Discriminator Loss : {}".format(l_D),
# 				 " k_{} : {}".format(t,k_t) 
# 				 )


# 	       #________________________________Save____________________________________


# 			if t % 200 == 0:

# 				summary = sess.run(merged_summary)
# 				writer.add_summary(summary, t)


# 				Generated_Images, Decoded_Generated_Images = sess.run([denorm_img(generated_image), denorm_img(reconstructed_image_fake)])
# 				save_image(Generated_Images, '{}/{}{}.png'.format("./Generated_Images", "Generated", t))
# 				save_image(Decoded_Generated_Images, '{}/{}{}.png'.format("./Decoded_Generated_Images", "AutoEncoded", t))
# 				print("-------------------Image saved-------------------")


# 			if t % 2000 == 0:
# 				print("Save model {}th".format(t))
# 				saver.save(sess, "./Check_Point/model.ckpt", global_step = t)
# 				# generator_saver.save(sess, "./Generator_Check_Point/model.ckpt", global_step = t)

# 			if t % 50000 == 0:
# 				lr_d, lr_g = sess.run([lr_d_update,lr_g_update])
# 				print("learning rate update")
# 				print("lr_d : {}".format(lr_d), "lr_g : {}".format(lr_g))

	

# 	       #--------------------------------------------------------------------
		
# 		writer.close()
# 		coord.request_stop()
# 		coord.join(threads)


# #-----------------------------------Train Finish---------------------------------



if __name__ == "__main__" :
	tf.app.run()


