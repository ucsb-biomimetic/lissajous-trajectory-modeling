import numpy
import tensorflow as tf
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import math
import pickle
import random
import copy
import os

import model
import preproc

set_names = ['M30D-1977-1980'] #list of scanning configurations to fit (names of folders that have calibration data)
n_periods = 2000 #expected number of mirror oscillations in dataset; used for filtering in frequency space

def train(weights, train_weights, loss_fn, phases, positions, niter, phase_adj, include_all, name=None, set_name=None, lr=0.00005):
	#training, with many options.  This is called once for each recorded calibration image
	#weights: list of weights to use during inference pass of training.  A list of tensorflow weights, as defined in the model file
	#train weights: list of weights to update in backward pass of training, must be a subset of weights, and point to weight objects shared with the inference weights
	#loss fn: a tensorflow function that reduces the predicted position and the measured position data to a loss value.  This is adjustable to accomodate switching between positional and periodic loss functions
	#phases: list of drive phases that are used to predict position.
	#positions: ground truth position measurements
	#niter: number of times training is applied to data (data does not change, gradient descent is not stochastic)
	#phase_adj: list data struct with instructions on enforcing the addition of extra periods per beat phase for any harmonic
	#include_all: whether or not only the lower harmonic weights are used, see the model inference function for more details
	#name: identifier string for which phase of training, used to save diagnostic plots
	#set_name: identifier string for which scanning condition is being trained, also the file path to calibration data
	#lr: learning rate

	print("starting training for: " + name)

	#diagnostic plots for before training
	if name is not None and set_name is not None: #check where extracted points lie before training
		predicted_position = model.predict_position(phases, weights)
		plt.figure(figsize=(30, 20))
		plt.scatter(phases[2], predicted_position)
		# plt.show()
		plt.savefig(set_name + "/training_diagnostic_plots/" + name + "_position_predictions_before_training.png")

	optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.99)

	niter = int(niter)

	@tf.function
	def train_iter():
		with tf.GradientTape() as tape:
			predicted_position = model.predict_position(phases, weights, include_all)
			loss, residuals = loss_fn(predicted_position, positions)

		gradients = tape.gradient(loss, train_weights)
		optimizer.apply_gradients(zip(gradients, train_weights))

		return loss


	for i in range(niter): #iterative gradient descent steps
		loss = train_iter()

		if i%10 == 0:
			print(loss.numpy(), end='\r')

		if (i+1)%(niter/5) == 0:
			print('')#new line

		#fix specific phase value in phase weights, since gradient decent can't seem to get it right:
		if i == 0:
			phase_weights = weights[1].numpy()
			for adj in phase_adj:
				phase_weights[0,0,adj[0]] = adj[1]
			weights[1].assign(phase_weights, read_value=False)

	#===========diagnostic plots after training
	if name is not None and set_name is not None: #check where extracted points lie after training
		predicted_position = model.predict_position(phases, weights)
		plt.figure(figsize=(30, 20))
		plt.scatter(phases[2], predicted_position)
		# plt.show()
		plt.savefig(set_name + "/training_diagnostic_plots/" + name + "_position_predictions_after_training.png")

		#plot predicted phase and amplitude adjustments after training
		positions_pred, phases_adj, amplitudes_adj = model.predict_position(phases, weights, return_extra=True)
		positions = positions_pred.numpy()
		phases_adj = phases_adj.numpy()
		amplitudes_adj = amplitudes_adj.numpy()

		plt.figure()
		plt.subplot(2,1,1)
		for i in range(model.fast_harmonics):
			#each fast harmonic has a different amplitude and phase correction
			plt.plot(phases[2], amplitudes_adj[:,:,i])
		plt.subplot(2,1,2)
		for i in range(model.fast_harmonics):
			#each fast harmonic has a different amplitude and phase correction
			plt.plot(phases[2], phases_adj[:,:,i])
		# plt.show()
		plt.savefig(set_name + "/training_diagnostic_plots/" + name + "_phase_amplitude_adjustments.png")

def main():
	#=============================================================== begin actual execution
	#get data
	for set_name in set_names: #name of folder where calibration data lies

		if not os.path.exists(set_name + "/training_diagnostic_plots"):
			os.makedirs(set_name + "/training_diagnostic_plots")
		
		weights_file = '%s/weights-%i-%i.pickle'%(set_name, model.slow_harmonics, model.fast_harmonics)

		save_weights = []

		#list of pairs: each pair is (zero-indexed harmonic #, # of extra periods per beat)
		phase_adj = {'x': [], 'y': []} #example with no phase adjustments
		phase_adj = {'x': [], 'y': [[1,1]]} #i.e. second harmonic of y oscillations gets one extra period per beat

		for direction in ['x', 'y']: #model is fit twice, once for each of the orthogonal axes of oscillation

			#get data:
			starter_data = numpy.load('%s/scope_%s_starter.npy'%(set_name, direction))
			grating_data = numpy.load('%s/scope_%s_grating.npy'%(set_name, direction))
			fine_data = numpy.load('%s/scope_%s_grating_fine.npy'%(set_name, direction))

			weights = model.weight_set()
			# second_order_phase_adj = 

			#================================================================================================================train on starter data
			times, positions, phases = preproc.traces_to_points(starter_data, n_periods=n_periods, names=[set_name, "%s initial two-position fit"%direction])

			#scale so that period is roughly right to match grating period
			positions = positions * 3.15

			def loss_fn(predicted_position, positions):
				residuals = tf.math.abs(predicted_position-positions)
				loss = tf.math.reduce_mean(residuals)
				return loss, residuals

			train_weights = weights.copy()
			train_weights.pop(4)

			train(weights, train_weights, loss_fn, phases, positions, 5e3, phase_adj[direction], False, "%s initial two-position fit"%direction, set_name)
			

			#=============================================retrain on periodic data, using the same coarse parameters; ensures that data gets locked correctly
			train_weights = [weights[2]]

			times, positions, phases = preproc.traces_to_points(grating_data, n_periods = n_periods)
			def loss_fn(predicted_position, positions):
				residuals = tf.math.abs(tf.math.sin(predicted_position)-positions)
				loss = tf.math.reduce_mean(residuals)
				return loss, residuals
			train(weights, train_weights, loss_fn, phases, positions, 1e3, phase_adj[direction], False, "%s Grating Alignment"%direction, set_name)
			#=======================================retrain on periodic data, using all weights
			train(weights, weights, loss_fn, phases, positions, 0.5e4, phase_adj[direction], True, "%s Grating fit"%direction, set_name)


			#=============================================================================================================retrain on fine periodic data
			times, positions, phases = preproc.traces_to_points(fine_data, n_periods=n_periods)
			def loss_fn(predicted_position, positions):
				residuals = tf.math.abs(tf.math.sin(3*predicted_position)-positions)
				loss = tf.math.reduce_mean(residuals)
				return loss, residuals	
			train(weights, train_weights, loss_fn, phases, positions, 1e3, phase_adj[direction], True, "%s Fine Grating Alignment"%direction, set_name)
			#=======================================retrain on periodic data, using all weights
			train(weights, weights, loss_fn, phases, positions, 2e5, phase_adj[direction], True, "%s Fine grating fit"%direction, set_name)


			#place weights for further saving
			save_weights.append(weights)


		with open(weights_file, 'wb') as f:
			f.write(pickle.dumps(save_weights))

if __name__ == '__main__':
	main()