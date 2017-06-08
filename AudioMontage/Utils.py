import tensorflow as tf
import numpy as np
import librosa
import scipy

from PIL import Image
from glob import glob
from os import walk, mkdir



#Get image information  
image = Image.open("./sample_image.jpg")
image_w, image_h = image.size
image_c = 3


#Get spectrogram information  
offset = 0
duration = 3
sampling_rate = 16000
fft_size = 1024

y,sr = librosa.load("./sample_audio.wav", offset=offset, duration=duration, sr=sampling_rate) # load audio
D = librosa.stft(y, n_fft=fft_size, hop_length=int(fft_size/2), win_length=fft_size, window='hann') # make spectrogram
spectrogram_h = D.shape[0] # height of spectrogram
spectrogram_w = D.shape[1] # width of spectrogram
spectrogram_c = 1 # channel of spectrogram


#Define constant
flags = tf.app.flags
FLAGS = flags.FLAGS

#training parameters
flags.DEFINE_float('lr', 0.00008, 'Initial learning rate.')
flags.DEFINE_float('B1', 0.5, 'Beta1')
flags.DEFINE_float('B2', 0.99, 'Beta2')
flags.DEFINE_integer('epochs', 100000, 'Maximum epochs to iterate.')
flags.DEFINE_integer('bn', 16, "Batch number")


#data parameters
flags.DEFINE_integer('spec_h', spectrogram_h, "Height of spectrogram" )
flags.DEFINE_integer('spec_w', spectrogram_w, "Width of spectrogram" )
flags.DEFINE_integer('spec_c', spectrogram_c, "Channel of spectrogram" )
flags.DEFINE_integer('img_h', image_h, "Height of image" )
flags.DEFINE_integer('img_w', image_w, "Width of image" )
flags.DEFINE_integer('img_c', image_c, "Channel of image" )
flags.DEFINE_integer('scale_w', 32, "Width Scaling Factor" )
flags.DEFINE_integer('scale_h', 32, "Height Scaling Factor" )

#model parameters
flags.DEFINE_integer('hidden_n', 64, "Hidden convolution number")
flags.DEFINE_integer('output_channel', 3, "Output channel number")
flags.DEFINE_float("gamma", 0.5, "Gamma : Diversity ratio")
flags.DEFINE_float("lamb", 0.001, "Lambda : Learning rate of k_t")
flags.DEFINE_float("iteration", 200000, "Maximum iteration number")

#gpu parameters
flags.DEFINE_float("gpu_portion", 0.4, "Limit the GPU portion")

#---------------------------------------------------------------------------#

#Functions

def generate_z(size=FLAGS.hidden_n):
	return tf.random_uniform(shape=(FLAGS.bn,size), minval=-1, maxval=1, dtype=tf.float32)

def get_loss(image, decoded_image):
	L1_norm = tf.reduce_mean(tf.abs(tf.subtract(image,decoded_image)))
	return L1_norm

def norm_img(image):
	image = image/127.5 - 1.
	return image

def denorm_img(norm):
	return tf.clip_by_value((norm + 1.)*127.5, 0, 255)


def upsample(images, size):
	"""    
	images : image having shape with [batch, height, width, channels], 
	size : output_size with [new_height, new_width]
	"""
	return tf.image.resize_nearest_neighbor(images=images, size=size)  