import numpy
import tensorflow as tf
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('TkAgg') #needed to run graphics on our particular server
from matplotlib import pyplot as plt
import math
import pickle
import random

def traces_to_phases(drive1, drive2, n_periods, time=None, names=None):
	#extracts phase of two signals, along with beat phase (difference between the two signals)
	#drive1: oscilloscope trace of first square wave used to drive MEMS mirror
	#drive2: trace of second drive signal
	#n_periods: number, expected number of mirror oscillations in the data, used for filter cutoff frequencies
	#time: time vector for drive signals, only used for plotting
	#names: string used to save diagnostic plots, if supplied

	index_vector = numpy.array(range(drive1.shape[0]))
	phase_vectors = []

	for drive in [drive1, drive2]:
		#first, filter out some high freq stuff - i.e. ringing at square wave edges that might create multiple crossings
		drive = numpy.fft.fft(drive)
		drive[n_periods*100:-n_periods*100] = 0;
		drive = numpy.real(numpy.fft.ifft(drive))
		drive = drive - numpy.mean(drive)

		crossings = numpy.argwhere(numpy.diff(numpy.sign(drive)))
		pos_crossings = numpy.argwhere(numpy.diff(numpy.sign(drive))>0)
		pos_crossings = pos_crossings[2:-2] #get rid of stuff at ends
		crossings = crossings[2:-2]

		drive_period = 2*(crossings[-1]-crossings[0])/(crossings.shape[0]-1) #exact frequency of this drive signal
		start_index = numpy.mean(pos_crossings%drive_period)#phase alignment

		phase_vectors.append(((index_vector-start_index)*2.*math.pi/drive_period)%(2.*math.pi)) #phase of this drive signal at each point in the data

	beat_phase = (phase_vectors[0]-phase_vectors[1])%(2.*math.pi)
	phase_vectors.append(beat_phase)

	if time is None:
		time = index_vector

	if names is not None:
		plt.figure()
		time = time*1000
		plt.plot(time, phase_vectors[0])
		plt.plot(time, phase_vectors[1])
		plt.plot(time, phase_vectors[2])
		plt.plot(time, drive1)
		plt.plot(time, drive2)
		plt.xlabel("time, ms")
		plt.ylabel("Phase, radians")
		plt.savefig("set" + names[0] + "/training_diagnostic_plots/" + names[1] + "_drive_signals_and_phases.png")

	return phase_vectors


def traces_to_points(data, n_periods, names=None):
	#extracts the sharp peaks and valleys from the data that indicate exact positions
	#data: Nx4 array of time, drive signal 1, drive signal 2, and APD response traces
	#n_periods: number, expected number of mirror oscillations in the data, used for filter cutoff frequencies
	#names: an informative string, used to name saved diagnostic plots if supplied

	phase_vectors = traces_to_phases(data[:,1], data[:,2], n_periods, names=names)

	#first, filter signal
	signal = numpy.fft.fft(data[:,3])
	signal[:n_periods*2] = 0
	signal[-n_periods*2:] = 0
	signal = numpy.real(numpy.fft.ifft(signal))

	#next, find greatest peaks in data
	top_bottom_peak_indx = []
	for sig in [signal, -signal]: #find maxima, then minima
		peak_indices, _ = find_peaks(sig)
		peaks = sig[peak_indices]
		top_fifth_threshold = numpy.max(sig)-(numpy.max(sig)-numpy.min(sig))/4.
		top_peak_indx_indx = numpy.nonzero(peaks>top_fifth_threshold)

		#if there are more than 10K points, randomly reduce to 10K:
		if top_peak_indx_indx[0].shape[0] > 10000:
			top_peak_indx_indx = (numpy.random.choice(top_peak_indx_indx[0], (10000), replace=False),"NA")

		top_bottom_peak_indx.append(peak_indices[top_peak_indx_indx[0]])
		print("found %i alignment points"%top_peak_indx_indx[0].shape[0])

	peak_indx = numpy.concatenate((top_bottom_peak_indx[0],top_bottom_peak_indx[1]))
	positions = peak_indx*0-1
	positions[:top_bottom_peak_indx[0].shape[0]] = 1
	times = data[peak_indx,0]
	phases = [phase_vectors[i][peak_indx] for i in range(3)]

	#one final sort, to get everything back in chronological order:
	time_indx = numpy.argsort(times)
	times = times[time_indx]
	positions = positions[time_indx]
	phases = [phases[i][time_indx] for i in range(3)]


	if names is not None:
		#plots to check that we got it all right:
		plt.figure()
		for i in range(1,2):
			plt.plot(data[:,0], phase_vectors[i])
			plt.scatter(times, phases[i])	
		#plt.plot(times, positions)
		plt.savefig("set" + names[0] + "/training_diagnostic_plots/" + names[1] + "_drive_phases_and_found_points.png")


		plt.figure()
		plt.plot(data[:,0], signal*25)
		#plt.plot(times, positions)
		plt.savefig("set" + names[0] + "/training_diagnostic_plots/" + names[1] + "_signal.png")

	return times, positions, phases #this is all we need from the data: the phases will predict the positions


