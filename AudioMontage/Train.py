"""
Python2 & Python3 
Version Compatible
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Utils import generate_z, FLAGS, get_loss, norm_img, denorm_img
from Decoder import Decoder
from Encoder import Encoder
from Loader import Image_Loader, save_image
from PIL import Image

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def main(_):

	"""


	Run main function
	"""

	#___________________________________________Layer info_____________________________________________________
	n = FLAGS.hidden_n

						
	Encoder_infos = {
						"outdim":[n,n,2*n,2*n,2*n,3*n,3*n, 3*n, 3*n, 4*n, 4*n, 4*n],\
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
								], \
						"stride":[ \
									[1, 1], \
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
								], \
					} 


	Decoder_infos = {
						"outdim":[n,n,n,n,n,n,n,n,3], \
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
								], \
						"stride":[ \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
								], \
					} 

	Generator_infos = {
						"outdim":[n,n,n,n,n,n,n,n,3], \
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
								], \
						"stride":[ \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
								], \
					} 


	"""
	Prepare Image Loader
	"""
	root = "../../workspace/DCGAN-tensorflow-master/data/celebA/"
	batch_size = FLAGS.bn
	scale_size = [FLAGS.scale_h,FLAGS.scale_w]
	data_format = "NHWC"
	loader = Image_Loader(root, batch_size, scale_size, data_format, file_type="jpg")



	"""
	Make Saving Directories
	"""
	# os.makedirs("./Check_Point")
	# os.makedirs("./logs") # make logs directories to save summaries
	# os.makedirs("./Real_Images")
	# os.makedirs("./Generated_Images")
	# os.makedirs("./Decoded_Generated_Images")




	#----------------------------------------------------------------------------------------------------



	#____________________________________Model composition________________________________________

	k = tf.Variable(0.0, name = "k_t", trainable = False, dtype = tf.float32) #init value of k_t = 0
	
	
	batch = loader.queue # Get image batch tensor
	image = norm_img(batch) # Normalize Imgae
	z_G = generate_z() # Sample embedding vector batch from uniform distribution
	z_D = generate_z() # Sample embedding vector batch from uniform distribution


	E = Encoder("Encoder", Encoder_infos)
	D = Decoder("Decoder", Decoder_infos)
	G = Decoder("Generator", Generator_infos)

	#Generator
	generated_image = G.decode(z_G)
	generated_image_for_disc = G.decode(z_D, reuse = True)


	#Discriminator (Auto-Encoder)	

	#image <--AutoEncoder--> reconstructed_image_real
	embedding_vector_real = E.encode(image)
	reconstructed_image_real = D.decode(embedding_vector_real)

	#generated_image_for_disc <--AutoEncoder--> reconstructed_image_fake
	embedding_vector_fake_for_disc = E.encode(generated_image_for_disc, reuse=True)
	reconstructed_image_fake_for_disc = D.decode(embedding_vector_fake_for_disc, reuse=True)

	#generated_image <--AutoEncoder--> reconstructed_image_fake
	embedding_vector_fake = E.encode(generated_image, reuse=True)
	reconstructed_image_fake = D.decode(embedding_vector_fake, reuse=True)


	#-----------------------------------------------------------------------------------------------



	#_________________________________Loss & Summary_______________________________________________


	"""
	Define Loss
	"""
	real_image_loss = get_loss(image, reconstructed_image_real)
	generator_loss_for_disc = get_loss(generated_image_for_disc, reconstructed_image_fake_for_disc)
	discriminator_loss = real_image_loss - tf.multiply(k, generator_loss_for_disc)

	generator_loss = get_loss(generated_image, reconstructed_image_fake)
	global_measure = real_image_loss + tf.abs(tf.multiply(FLAGS.gamma,real_image_loss) - generator_loss)


	"""
	Summaries
	"""
	tf.summary.scalar('Real image loss', real_image_loss)
	tf.summary.scalar('Generator loss for discriminator', generator_loss_for_disc)
	tf.summary.scalar('Discriminator loss', discriminator_loss)
	tf.summary.scalar('Generator loss', generator_loss)
	tf.summary.scalar('Global_Measure', global_measure)
	tf.summary.scalar('k_t', k)
	

	merged_summary = tf.summary.merge_all() # merege summaries, no more summaries under this line

	#-----------------------------------------------------------------------------------------------







	#_____________________________________________Train_______________________________________________

	discriminator_parameters = []
	generator_parameters = []

	for v in tf.trainable_variables():
		if 'Encoder' in v.name:
			discriminator_parameters.append(v)
			print("Discriminator parameter : ", v.name)
		elif 'Decoder' in v.name:
			discriminator_parameters.append(v)
			print("Discriminator parameter : ", v.name)			
		elif 'Generator' in v.name:
			generator_parameters.append(v)
			print("Generator parameter : ", v.name)
		else:
			print("None of Generator and Discriminator parameter : ", v.name)

	optimizer_D = tf.train.AdamOptimizer(FLAGS.lr,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(discriminator_loss,var_list=discriminator_parameters)
	optimizer_G = tf.train.AdamOptimizer(FLAGS.lr,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(generator_loss,var_list=generator_parameters)

	with tf.control_dependencies([optimizer_D, optimizer_G]):
		k_update = tf.assign(k, tf.clip_by_value(k + FLAGS.lamb * (FLAGS.gamma*real_image_loss - generator_loss), 0, 1)) #update k_t

	init = tf.global_variables_initializer()	


	NUM_THREADS=2
	config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
						intra_op_parallelism_threads=NUM_THREADS,\
						allow_soft_placement=True,\
						device_count = {'CPU': 1},\
						)

	# config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_portion



	with tf.Session(config=config) as sess:

		sess.run(init) # Initialize Variables

		coord = tf.train.Coordinator() # Set Coordinator to Manage Queue Runners
		threads = tf.train.start_queue_runners(sess, coord=coord) # Set Threads
		writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'		

#_______________________________Restore____________________________________

		saver = tf.train.Saver(max_to_keep=1000)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir="./Check_Point")

		
		# try :	
		# 	if ckpt and ckpt.model_checkpoint_path:
		# 		print("check point path : ", ckpt.model_checkpoint_path)
		# 		saver.restore(sess, ckpt.model_checkpoint_path)	
		# 		print('Restored!')
		# except AttributeError:
		# 		print("No checkpoint")	

		Real_Images = sess.run(denorm_img(image))
		save_image(Real_Images, '{}.png'.format("./Real_Images/Real_Image"))

#---------------------------------------------------------------------------
		for t in range(FLAGS.iteration): # Mini-Batch Iteration Loop

			if coord.should_stop():
				break
			
			_, _, l_D, l_G, l_Global, k_t = sess.run([\
													optimizer_D,\
													optimizer_G,\
													discriminator_loss,\
													generator_loss,\
													global_measure,\
													k_update,\
											   		])

			if t % 1000 == 0:
				print(
					 " Step : {}".format(t),
					 " Global measure of convergence : {}".format(l_Global),
					 " Generator Loss : {}".format(l_G),
					 " Discriminator Loss : {}".format(l_D),
					 " k_{} : {}".format(t,k_t) 
					 )


			

			
	       #________________________________Save____________________________________


			if t % 10000 == 0:

				summary = sess.run(merged_summary)
				writer.add_summary(summary, t)


				Generated_Images, Decoded_Generated_Images = sess.run([denorm_img(generated_image), denorm_img(reconstructed_image_fake)])
				save_image(Generated_Images, '{}/{}{}.png'.format("./Generated_Images", "Generated", t))
				save_image(Decoded_Generated_Images, '{}/{}{}.png'.format("./Decoded_Generated_Images", "AutoEncoded", t))
				# save_image(Rimage, '{}/{}{}.png'.format("./Real_Images", "Real_Images", t))
				print("-------------------Image saved-------------------")


			if t % 10000 == 0:
				print("Save model {}th".format(t))
				saver.save(sess, "./Check_Point/BEGAN.model", global_step = t)


	       #--------------------------------------------------------------------
		
		writer.close()
		coord.request_stop()
		coord.join(threads)


#-----------------------------------Train Finish---------------------------------



if __name__ == "__main__" :
	tf.app.run()


