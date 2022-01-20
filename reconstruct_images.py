import numpy
import tensorflow as tf
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import math
import pickle
import glob
import scipy.interpolate
import imageio
import os
import copy

import preproc
import model

#this file uses a trained model to convert oscilloscope traces of drive signals and APD recordings into images

folder = 'M30D-1977-1980/'
weights_file = folder+'weights-20-5.pickle'
n_periods = 2000
imsize_x = 512
imsize_y = 512


def image_postproc(image, niters=1):
	#cleans up holes left in scanning patterns
	for ii in range(niters): #iterate more than once if there are extra large gaps
		new_image = numpy.zeros(image.shape)
		for i in range(1,image.shape[0]-1):
			for j in range(1,image.shape[1]-1):
				if image[i,j] > 0:
					new_image[i,j] = image[i,j]
				elif numpy.sum(image[i-1:i+1,j-1:j+1]>0) > 0:
					new_image[i,j] = numpy.sum(image[i-1:i+1,j-1:j+1])/numpy.sum(image[i-1:i+1,j-1:j+1]>0)
		image = new_image

	image_min = numpy.min(image)
	image_max = numpy.max(image)
	image = (image-image_min)/(image_max-image_min)*255.
	image = image.astype(numpy.uint8)
	return image


#start by getting list of all numpy files in given folder: will do reconstruction for all of them
if not os.path.exists(folder + 'frame_sequence_reconstructions'):
			os.makedirs(folder + 'frame_sequence_reconstructions')
			os.makedirs(folder + 'gradual_reconstruction_sequence')

#===========meat of the algorithm: converts scope traces into x and y positions
with open(weights_file, 'rb') as f:
	weights = pickle.loads(f.read())

for file_name in glob.glob(folder+'*.npy'):

	base_name = file_name[:-4]
	name = base_name[len(folder):]

	image_data = numpy.load(base_name + '.npy')

	phases = preproc.traces_to_phases(image_data[:,1], image_data[:,2], n_periods, image_data[:,0])

	x_positions = model.predict_position(phases, weights[0])
	x_positions = x_positions.numpy()

	y_positions = model.predict_position(phases, weights[1])
	y_positions = y_positions.numpy()


	#============form images by assigning avalanche photodiode measurements to array locations

	#first, scale positions
	image = numpy.zeros([imsize_x,imsize_y])

	xmax = numpy.max(x_positions)
	xmin = numpy.min(x_positions)
	x_positions = numpy.int32((imsize_x-1)*(x_positions-xmin)/(xmax-xmin))
	ymax = numpy.max(y_positions)
	ymin = numpy.min(y_positions)
	y_positions = numpy.int32((imsize_y-1)*(y_positions-ymin)/(ymax-ymin))

	#somehow some NaNs got in here... let's try to filter them out:
	image_data[:,3] = numpy.nan_to_num(image_data[:,3])
	image_data[:,3] = image_data[:,3]*(numpy.abs(image_data[:,3]) < 0.23)
	# print(numpy.max(image_data[:,3]))


	#=====================================================first, plot frames that start where the beat phase is zero

	#find where the beat phase is zero (thus finding starting points for each frame)
	#identify which way the beat slope is:
	zero_beat_indices1 = numpy.nonzero(numpy.diff(phases[2]) > 0)
	zero_beat_indices2 = numpy.nonzero(numpy.diff(phases[2]) < 0)
	if zero_beat_indices1[0].shape[0] < zero_beat_indices2[0].shape[0]:
		beat_slope_sign = 1
	else:
		beat_slope_sign = -1

	zero_beat_indices = numpy.nonzero(beat_slope_sign*numpy.diff(phases[2]) > 0)
	zero_beat_indices = zero_beat_indices[0]

	#depending on how many beat edges were found, dice up the data trace in different ways.  Create a list of (start, stop) indices for frames
	if zero_beat_indices.shape[0] <= 1: #not guarenteed to even have a full frame
		middle = int(x_positions.shape[0]/2)-1
		frame_edges = [(0,middle),(middle,-1)]
		gradual_frame = [0,-1]

	elif zero_beat_indices.shape[0] == 2:
		#between 1 and three frames.  Take bits from the start and end that overlap for the sequential reconstruction
		start_beat_indices = numpy.nonzero(beat_slope_sign*numpy.diff((phases[2]+math.pi*2.-phases[2][4])%(math.pi*2.)) > 0)
		end_beat_indices = numpy.nonzero(beat_slope_sign*numpy.diff((phases[2]+math.pi*2.-phases[2][-5])%(math.pi*2.)) > 0)
		# print(end_beat_indices[0].shape[0])
		if end_beat_indices[0].shape[0] == 2:
			frame_edges = [(4,end_beat_indices[0][0]),(start_beat_indices[0][-1],-5)]
		if end_beat_indices[0].shape[0] == 3:
			frame_edges = [(4,end_beat_indices[0][1]),(start_beat_indices[0][-2],-5)]
		gradual_frame = [0,-1]

	elif zero_beat_indices.shape[0] > 7:
		#if it beats a lot, then this is probably from an unstable type pattern, so just split in half:
		middle = int(x_positions.shape[0]/2)-1
		frame_edges = [(0,middle),(middle,-1)]
		gradual_frame = [0,-1]

	else:
		#in this case there are several full frames that can be plotted:
		frame_edges = []
		for i in range(zero_beat_indices.shape[0]-1):
			frame_edges.append((zero_beat_indices[i],zero_beat_indices[i+1]))
		gradual_frame = frame_edges[0]

	color_image = numpy.zeros([imsize_x, imsize_y, 3])
	for i in range(len(frame_edges)):
		image = numpy.zeros([imsize_x,imsize_y])
		edges = frame_edges[i]

		frame_x_positions = x_positions[edges[0]:edges[1]]
		frame_y_positions = y_positions[edges[0]:edges[1]]
		frame_image_data = image_data[edges[0]:edges[1],3]

		for x, y, value in zip(frame_x_positions, frame_y_positions, frame_image_data):
			image[x, y] = value

		image = image_postproc(image,2)
		if (i<2):
			color_image[:,:,i] = image
		imageio.imwrite(folder + 'frame_sequence_reconstructions/' + name + '_%i.png'%i, image)


	color_image = color_image.astype(numpy.uint8)
	imageio.imwrite(folder + 'frame_sequence_reconstructions/' + name + '_biframe.png', color_image)


	frame_x_positions = x_positions[gradual_frame[0]:gradual_frame[1]]
	frame_y_positions = y_positions[gradual_frame[0]:gradual_frame[1]]
	frame_image_data = image_data[gradual_frame[0]:gradual_frame[1],3]

	#next, plot incrimental for last frame from above
	nsubframes = 10
	counter = 0
	image = numpy.zeros(image.shape) #reset to zeros
	save_interval = int(frame_y_positions.shape[0]/nsubframes)
	for x, y, value in zip(frame_x_positions, frame_y_positions, frame_image_data):
		image[x, y] = value
		counter = counter + 1
		if counter%save_interval == 0:
			imageio.imwrite(folder + 'gradual_reconstruction_sequence/' + name + '_%i.png'%(counter/save_interval), image_postproc(copy.copy(image),0))
