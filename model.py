import numpy
import tensorflow as tf
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import math
import pickle
import random

#============================This file defines the model used to describe the mirror motion

#parameters: number of harmonics, allows control over complexity and fine-tunability of model but does not affect fundamental operation
#set1 are fit during coarse training before additionally fitting set2, to prevent getting suck in particular local minima
slow_harmonics_set1 = 4
slow_harmoincs_set2 = 16
slow_harmonics = slow_harmonics_set1 + slow_harmoincs_set2
fast_harmonics_set1 = 2
fast_harmonics_set2 = 3
fast_harmonics = fast_harmonics_set1 + fast_harmonics_set2

def weight_set():
	#generates initialized model weights
	fast_weights_set1 = tf.Variable(initial_value = numpy.zeros((1,4,fast_harmonics_set1)), dtype=tf.float32)
	fast_weights_set2 = tf.Variable(initial_value = numpy.zeros((1,4,fast_harmonics_set2)), dtype=tf.float32)
	phase_weights = tf.Variable(initial_value = numpy.zeros((1,slow_harmonics*2+1, fast_harmonics)), dtype=tf.float32) #extra one for full phase rotation
	amplitude_weights = tf.Variable(initial_value = numpy.zeros((1,slow_harmonics*2+1, fast_harmonics)), dtype=tf.float32) #extra one for constant amplitude
	slow_weights = tf.Variable(initial_value = numpy.zeros((1,slow_harmonics*2+1)), dtype=tf.float32) #extra one for constant amplitude
	return [fast_weights_set1, phase_weights, slow_weights, amplitude_weights, fast_weights_set2]

#@tf.function
def predict_position(phases, weights, include_all=True, return_extra=False):
	#this is the core function that defines how positions are predicted from the drive signals.
	#phases: list of phase traces, in order: drive phase 1, drive phase 2, and beat phase
	#weights: a set of weight values to use in the model
	#include all: weather to use all basis functions (True), or to only use set 1 (False)
	#return extra: if True, function also returns the predicted phase and amplitude adjustments in addition to the predicted position
	
	#=========================================== assemble basis functions
	slow_waves = []
	slow_waves_phase = []
	slow_waves_phase.append(phases[2]) #direct phase is needed as a predictor for the following reason:
	#the phase adjustment signal needs to be able to wrap around (2 pi to 0) which is physically low frequency but
	#requires high frequency sine components, which we do not want to have to include in the model
	for i in range(slow_harmonics_set1):
		slow_waves.append(tf.math.cos(phases[2]*(i+1)))
		slow_waves.append(tf.math.sin(phases[2]*(i+1)))
	for i in range(slow_harmonics_set1, slow_harmonics):
		if include_all:
			slow_waves.append(tf.math.cos(phases[2]*(i+1)))
			slow_waves.append(tf.math.sin(phases[2]*(i+1)))
		else:
			slow_waves.append(phases[2]*0)
			slow_waves.append(phases[2]*0)

	slow_waves_phase = slow_waves_phase + slow_waves

	slow_waves = tf.stack(slow_waves, axis=1)
	slow_waves = tf.cast(slow_waves, dtype=tf.float32)

	slow_waves_phase = tf.stack(slow_waves_phase, axis=1)
	slow_waves_phase = tf.cast(slow_waves_phase, dtype=tf.float32)

	#still do sin/cos for slow waves, since gradients should be more direct than for fitting an explicit phase
	slow_waves = tf.concat((slow_waves, tf.ones((phases[0].shape[0],1))), axis=1)
	slow_waves_extra_dim = tf.expand_dims(slow_waves, axis=2)

	slow_waves_phase_extra_dim = tf.expand_dims(slow_waves_phase, axis=2)

	#================================================= actual model inference
	phase_adjustment = tf.math.reduce_sum(weights[1]*slow_waves_phase_extra_dim, keepdims=True, axis=1);
	#phase_adjustment = tf.math.floormod(phase_adjustment, 2*math.pi)
	#print(tf.math.reduce_mean(phase_adjustment,axis=0).numpy())
	amplitude_adjustment = tf.math.reduce_sum(weights[3]*slow_waves_extra_dim, keepdims=True, axis=1) + tf.ones((1,1,1));
	amplitude_adjustment = tf.math.abs(amplitude_adjustment)

	fast_phases = []
	fast_phases_base = tf.stack((phases[0], phases[1]), axis=1)
	for i in range(fast_harmonics):
		fast_phases.append((i+1)*fast_phases_base)
	fast_phases = tf.stack(fast_phases, axis=2)
	fast_phases = tf.cast(fast_phases, dtype=tf.float32)

	#print(fast_phases.shape)
	#print(amplitude_adjustment.shape)
	#print(phase_adjustment.shape)

	fast_phases = fast_phases + phase_adjustment
	fast_waves = tf.concat((tf.math.sin(fast_phases), tf.math.cos(fast_phases)), axis=1)*amplitude_adjustment

	#print(waves.shape)
	fast_weights = tf.concat((weights[0], weights[4]), axis=2);
	fast_prediction = fast_waves*fast_weights
	#print(fast_prediction[0:2000:100,:,1].numpy())
	fast_prediction = tf.math.reduce_sum(fast_prediction, axis=(1,2))
	slow_prediction = tf.math.reduce_sum(slow_waves*weights[2], axis=1)
	#print(position.shape)
	if return_extra:
		return fast_prediction + slow_prediction, phase_adjustment, amplitude_adjustment
	else:
		return fast_prediction + slow_prediction