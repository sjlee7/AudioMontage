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
import load_spec

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
	return tf.Variable(tf.zeros(shape))

n = FLAGS.hidden_n

					
Encoder_infos = {
					"outdim":[n,n,2*n,2*n,2*n,3*n,3*n, 3*n, 3*n, 4*n, 4*n, 4*n, 4*n, 4*n, 4*n, 6*n, 6*n, 6*n, 8*n, 8*n, 8*n],\
					"kernel":[ \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
								[3, 3], \
							], \
					"stride":[ \
								[1, 1], \
								[1, 1], \
								[1, 1], \
								[4, 4], \
								[1, 1], \
								[1, 1], \
								[2, 2], \
								[1, 1], \
								[1, 1], \
								[2, 2], \
								[1, 1], \
								[1, 1], \
								[2, 2], \
								[1, 1], \
								[1, 1], \
								[2, 2], \
								[1, 1], \
								[1, 1], \
								[3, 3], \
								[1, 1], \
								[1, 1], \
							], \
				} 

E = spec_encoder("Encoder", Encoder_infos)

weights = load_spec.weights

# weights = {
#         'wconv1':init_weights([3, 3, 1, 64]),
#         'wconv2':init_weights([3, 3, 64, 64]),
#         'wconv3':init_weights([3, 3, 64, 128]),
#         'wconv4':init_weights([3, 3, 128, 192]),
#         'wconv5':init_weights([3, 3, 128, 192]),
#         'wconv6':init_weights([3, 3, 192, 256]),
#         'wconv7':init_weights([3, 3, 256, 512]),
#         'bconv1':init_biases([64]),
#         'bconv2':init_biases([64]),
#         'bconv3':init_biases([128]),
#         'bconv4':init_biases([192]),
#         'bconv5':init_biases([192]),
#         'bconv6':init_biases([256]),
#         'bconv7':init_biases([512]),
#         'woutput':init_weights([512, 64]),
#         'boutput':init_biases([64])
#         }


handle_male = open('./final_z_male/recon_z_male.pickle', 'rb')
test_male = pickle.load(handle_male)

handle_fmale = open('./final_z_male/recon_z_fmale.pickle', 'rb')
test_fmale = pickle.load(handle_fmale)

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

def cnn(melspectrogram, weights, phase_train):

	x = tf.reshape(melspectrogram, [-1, 1, 128, 157])
	x = batch_norm(melspectrogram, 157, phase_train)
	x = tf.reshape(melspectrogram, [-1, 128, 157, 1])

	conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
	conv2_1 = tf.nn.relu(batch_norm(conv2_1, 64, phase_train))
	mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_1 = tf.nn.dropout(mpool_1, 1.0)
	print (dropout_1.shape)

	conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
	conv2_2 = tf.nn.relu(batch_norm(conv2_2, 128, phase_train))
	mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_2 = tf.nn.dropout(mpool_2, 1.0)
	print (dropout_2.shape)

	conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
	conv2_3 = tf.nn.relu(batch_norm(conv2_3, 128, phase_train))
	mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_3 = tf.nn.dropout(mpool_3, 1.0)
	print (dropout_3.shape)

	conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
	conv2_4 = tf.nn.relu(batch_norm(conv2_4, 256, phase_train))
	mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_4 = tf.nn.dropout(mpool_4, 1.0)
	print (dropout_4.shape)

	conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
	conv2_5 = tf.nn.relu(batch_norm(conv2_5, 384, phase_train))
	mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_5 = tf.nn.dropout(mpool_5, 1.0)
	print (dropout_5.shape)

	conv2_6 = tf.add(tf.nn.conv2d(dropout_5, weights['wconv6'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv6'])
	conv2_6 = tf.nn.relu(batch_norm(conv2_6, 512, phase_train))
	mpool_6 = tf.nn.max_pool(conv2_6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_6 = tf.nn.dropout(mpool_6, 1.0)
	print (dropout_6.shape)

	conv2_7 = tf.add(tf.nn.conv2d(dropout_6, weights['wconv7'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv7'])
	conv2_7 = tf.nn.relu(batch_norm(conv2_7, 1024, phase_train))
	mpool_7 = tf.nn.max_pool(conv2_7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	dropout_7 = tf.nn.dropout(mpool_7, 1.0)
	print (dropout_7.shape)

	flat = tf.reshape(dropout_7, [-1, weights['woutput'].get_shape().as_list()[0]])
	p_y_X = tf.layers.dense(inputs=flat, units = 1024, activation=None)

	# p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(p_y_X,weights['woutput']),weights['boutput']))
	p_y_X = tf.add(tf.matmul(p_y_X,weights['woutput']),weights['boutput'])
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
	root1 = "./audio_sample/f5000"
	root2 = "./audio_sample/m5000"
	
	batch_size = 16 #FLAGS.bn
	scale_size = [128, 157]
	data_format = "NHWC"
	loader1 = Spectrogram_Loader(root1, batch_size, scale_size, data_format)
	loader2 = Spectrogram_Loader(root2, batch_size, scale_size, data_format)

	spec_holder = tf.placeholder(tf.float32, shape=(None, 128, 157, 1))#FLAGS.bn
	y = tf.placeholder(tf.float32, shape = (None, FLAGS.z_size))#FLAGS.bn
	# lr = tf.Variable(FLAGS.lr_reverse, name='lr')
	lr = tf.placeholder(tf.float32)
	lam1 = tf.placeholder(tf.float32)
	lam2 = tf.placeholder(tf.float32)

	phase_train = tf.placeholder(tf.bool, name='phase_train')

	y_ = load_spec.cnn_1d(spec_holder, weights, keep_prob=1.0)
	# y_ = load_spec.cnn_withoutb(spec_holder, weights, keep_prob=1.0)
	# y_ = load_spe.E.encode(spec_holder)
	pre_mean, pre_var = tf.nn.moments(y_, axes=[1])

	predict_op = tf.reshape(tf.sqrt(pre_var), [32,-1])*y_ - tf.reshape(pre_mean, [32,-1])
	# y_ = E.encode(spec_holder)

	predict_op = y_


	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
	L1_norm = tf.reduce_mean(tf.abs(tf.subtract(y, predict_op)))
	# L1_norm = tf.nn.l1_loss(predict_op - y)
	# L2_norm = tf.sqrt(tf.nn.l2_loss(predict_op - y))
	Cos_loss = tf.abs(tf.losses.cosine_distance(y, predict_op, dim=1))
	com_loss = lam1*L1_norm + lam2*Cos_loss



	correct_pred = tf.equal(tf.argmax(predict_op, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(L1_norm)

	# predict_op = y_


	# L1_norm = tf.reduce_mean(tf.abs(tf.subtract(predict_op, y)))
	# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# y_train1 = np.array(test_fmale).reshape(6000, 64)
	# y_train2 = np.array(test_male).reshape(6000, 64)
	# y_train1 = np.ones(10000).tolist()
	# y_train1.extend(np.zeros(10000).tolist())
	# y_train2 = np.zeros(10000).tolist()
	# y_train2.extend(np.ones(10000).tolist())
	# y_train = np.array(pd.concat([pd.DataFrame(y_train1), pd.DataFrame(y_train2)], axis = 1))
	# y_train1 = np.array(y_train[:10000,:])
	# y_train2 = np.array(y_train[10000:20000,:])
	# print(y_train.shape)
	saver = tf.train.Saver()
	
	print(" Reading checkpoints...")
	cnn_parameters = []

	for v in tf.trainable_variables():
		print(v.name)
		cnn_parameters.append(v)
		
	print("cnn variables : {}".format(cnn_parameters))

	NUM_THREADS=1
	config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
						intra_op_parallelism_threads=NUM_THREADS,\
						allow_soft_placement=True,\
						device_count = {'CPU': 1},\
						)

	with tf.Session(config=config) as sess:
		tf.initialize_all_variables().run()

		# cnn_saver = tf.train.Saver(cnn_parameters)
		# ckpt = tf.train.get_checkpoint_state(checkpoint_dir="./Cnn_Check_Point")
		# saver.restore(sess, checkpoint_path)

		# try :	
		# 	if ckpt and ckpt.model_checkpoint_path:
		# 		print("check point path : ", ckpt.model_checkpoint_path)
		# 		saver.restore(sess, ckpt.model_checkpoint_path)
		# 		print('Restored!')
		# except AttributeError:
		# 		print("No checkpoint")

		# saver = tf.train.Saver(max_to_keep=1000)
		# ckpt = tf.train.get_checkpoint_state(checkpoint_dir="./Cnn_Check_Point")
				

		coord = tf.train.Coordinator() # Set Coordinator to Manage Queue Runners
		threads = tf.train.start_queue_runners(sess, coord=coord) # Set Threads
		# writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'		
		
		restorer = tf.train.Saver(cnn_parameters)
		checkpoint_path = tf.train.latest_checkpoint("./Cnn_Check_Point")
		restorer.restore(sess, checkpoint_path)


		test_pre = []
		# mean_test_loss = []

		# for start, end in zip(range(0, 32, 16), range(16, 32, 16)):
		for i in range(1):
			X_test = loader1.get_spectrogram_from_loader_test(sess)
			# X_test = np.concatenate((X_test1[:,:,:,:], X_test2[:,:,:,:]), axis=0)
			# y_test = np.concatenate((y_train1[4000+start:4000+end][:,:], y_train2[4000+start:4000+end][:,:]),axis=0)
			test_input_dict = {spec_holder: X_test, 
								lr: 0.0001,
								phase_train: True}
			t_pre = sess.run([predict_op], feed_dict=test_input_dict)

			test_pre.extend(t_pre)
			print("fmale")
			print(t_pre)
			X_test = loader2.get_spectrogram_from_loader_test(sess)

			test_input_dict = {spec_holder: X_test, 
								lr: 0.0001,
								phase_train: True}
			t_pre = sess.run([predict_op], feed_dict=test_input_dict)
			print("male")
			print(t_pre)
			test_pre.extend(t_pre)

			# mean_test_loss.append(loss1)

		file_name = "./pre_zsample.pickle"
		with open(file_name, 'wb') as handle:
			pickle.dump(test_pre, handle, protocol=pickle.HIGHEST_PROTOCOL)
		

# import and transform audio file 
		test_pre = []
		ls = os.listdir("audio_sample/mf5000")
		ls.sort()
		# # print (ls)
		for lsi in ls:
			print(lsi)
			y, sr = librosa.core.load("audio_sample/mf5000/"+lsi, sr = 16000, mono=True, duration=2.5)
			D = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, n_mels=128, hop_length=int(1024/4), fmax=None)
			# print ("mel : ",D)

			D = np.log10(D + np.finfo(float).eps)
			# D = np.expand_dims(D, axis=2)

			# D = np.expand_dims(D, axis=2)
			D = D.reshape(1, 128, 157, 1)
			# D = D[:,:,:,:]
			# D = D.transpose(axes=[0,2,1,3])

			# D = np.concatenate((X_test1[:,:,:,:], X_test2[:,:,:,:]), axis=0)
			sample_test_feed_dict = {spec_holder:D,
						lr:0.0001,
						phase_train:True}

			sample_t_pred = sess.run([predict_op], feed_dict=sample_test_feed_dict)
			# # # print ('mel :', de1)
			# # # print ('conv_1 : ', de2)
			# # # print ('dropout_5 : ', de2)
			print (sample_t_pred)
			test_pre.extend(sample_t_pred)

			# y = sample_t_pred[0][0]
			# y_mean = np.average(y)
			# y_std = np.std(y)
			# for i in range(FLAGS.z_size):
			# 	# print (y[i])
			# 	y[i] = y_std*y[i] - y_mean
			# test_pre.append(y)

		file_name = "./pre_z.pickle"
		with open(file_name, 'wb') as handle:
			pickle.dump(test_pre, handle, protocol=pickle.HIGHEST_PROTOCOL)




		# sample_t_pred, de1, de2, de3 = sess.run([predict_op, d1, d2, d3], feed_dict=sample_test_feed_dict)
		# # print ('mel :', de1)
		# # print ('conv_1 : ', de2)
		# # print ('dropout_5 : ', de2)
		# test_pre.append(sample_t_pred)
		# sample_t_pred, de1, de2, de3 = sess.run([predict_op, d1, d2, d3], feed_dict=sample_test_feed_dict)
		# # print ('mel :', de1)
		# # print ('conv_1 : ', de2)
		# # print ('dropout_5 : ', de2)
		# test_pre.append(sample_t_pred)

		# # sample_t_pred = sess.run(predict_op, feed_dict=sample_test_feed_dict)
		# # test_pre.append(sample_t_pred)
		# # sample_t_pred = sess.run(predict_op, feed_dict=sample_test_feed_dict)
		# # test_pre.append(sample_t_pred)
		# file_name = "./pre_zsample.pickle"

		# with open(file_name, 'wb') as handle:
		# 	pickle.dump(test_pre, handle, protocol=pickle.HIGHEST_PROTOCOL)


		# for start, end in zip(range(0, 1000, 16), range(16, 1000, 16)):
		# 	X_test1 = loader1.get_spectrogram_from_loader(sess)
		# 	X_test2 = loader2.get_spectrogram_from_loader(sess)
		# 	X_test = np.concatenate((X_test1[:,:,:,:], X_test2[:,:,:,:]), axis=0)
		# 	# y_test = np.concatenate((y_train1[4000+start:4000+end][:,:], y_train2[4000+start:4000+end][:,:]),axis=0)
		# 	test_input_dict = {spec_holder: X_test, 
		# 						# y: y_test,
		# 						lr: 0.0001,
		# 						phase_train: True}
		# 	t_pre = sess.run([predict_op], feed_dict=test_input_dict)

		# 	test_pre.extend(t_pre)
		# 	# mean_test_loss.append(loss1)
		# 	# print('iter : ', start, 'test loss1 : ', loss1)
		# 	file_name = "./pre_z.pickle"
		# 	with open(file_name, 'wb') as handle:
		# 		pickle.dump(test_pre, handle, protocol=pickle.HIGHEST_PROTOCOL)
			# save_path = saver.save(sess, 'model_f')


if __name__ == "__main__" :
	tf.app.run()


