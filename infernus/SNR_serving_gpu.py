print("Starting SNR serving job")

import time
start = time.time()

#import sys
#sys.path.append('/fred/oz016/alistair/GWSamplegen/')

from GWSamplegen.noise_utils import fetch_noise_loaded, load_noise, load_psd
from typing import List, Tuple
import numpy as np
import os
from numpy.lib.stride_tricks import as_strided
from GWSamplegen.waveform_utils import load_pycbc_templates
from GWSamplegen.snr_utils_np import np_get_cutoff_indices

from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.types import TimeSeries
from pycbc.filter import highpass
import argparse
import gc

from GWSamplegen.snr_utils_np import numpy_matched_filter, mf_in_place, np_sigmasq
from GWSamplegen.noise_utils import get_valid_noise_times

noise_dir = "/fred/oz016/alistair/GWSamplegen/noise/O3_first_week_1024"
duration = 1024
sample_rate = 2048
delta_t = 1/sample_rate
f_lower = 30
f_final = 1024
delta_f = 1/duration
approximant = "TaylorF2"

import multiprocessing as mp
n_cpus = 2

WINDOW_SIZE = 2048
STEP = 128


def make_windows_2d(time_series_list, window_size=WINDOW_SIZE, step_size=STEP):
	"""
	Turns a list of 1D arrays into a 3D array of sequential labelled windows of window_size with horizon size label.
	"""
	# Convert the list of time series into a 2D numpy array
	time_series_array = np.array(time_series_list)

	num_series, series_length = time_series_array.shape

	#print("WINDOW STATS",num_series, series_length)

	# Calculate the number of windows for each time series
	num_windows = (series_length - window_size) // step_size + 1
	#print(num_windows)

	# Calculate the strides for creating the windowed view
	strides = time_series_array.strides + (time_series_array.strides[-1] * step_size,)

	#print("strides:", strides)

	# Use as_strided to create a view of the data
	windowed_array = as_strided(
		time_series_array,
		shape=(num_series, num_windows, window_size),
		strides=(strides[0], strides[2], strides[1])
	)
    

	return windowed_array


def noise_generator(valid_times, paths, file_list, duration, sample_rate):
	file_idx = 0
	
	for i in range(len(valid_times)):
		#print(i)
		if i == 0:
			noise = np.load(file_list[file_idx])
		
		if valid_times[i] + duration > int(paths[file_idx][1]) + int(paths[file_idx][2]):
			file_idx += 1
			print("loading new file")
			noise = np.load(file_list[file_idx])
			
		if int(paths[file_idx][1]) <= valid_times[i]:
			#print("start time good")
			if int(paths[file_idx][1]) + int(paths[file_idx][2]) >= valid_times[i] + duration:
				start_idx = int((valid_times[i] - int(paths[file_idx][1])) * sample_rate)
				end_idx = int(start_idx + duration * sample_rate)
				#print(start_idx, end_idx)
				yield noise[:,start_idx:end_idx]



valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, 900)


templates, metricParams, aXis= load_pycbc_templates("PyCBC_98_aligned_spin", "/fred/oz016/alistair/GWSamplegen/template_banks/")




N = int(duration/delta_t)
kmin, kmax = np_get_cutoff_indices(f_lower, None, delta_f, N)


ifos = ["H1", "L1"]

psds = load_psd(noise_dir, duration,ifos , f_lower, int(1/delta_t))

#new numpy code time!

for psd in psds:
	psds[psd] = psds[psd][kmin:kmax]



hp, _ = get_td_waveform(mass1 = templates[0,1], mass2 = templates[0,2], 
						delta_t = delta_t, f_lower = f_lower, approximant = 'SpinTaylorT4')

max_waveform_length = len(hp)/sample_rate
max_waveform_length = max(12, int(np.ceil(max_waveform_length/10)*10))




parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int)
parser.add_argument('--totaljobs', type=int, default=1)
args = parser.parse_args()

n_jobs = args.totaljobs
job_id = args.index

print("starting job {} of {}".format(job_id, n_jobs))

#MAKING JOB SMALLER
templates = templates[:30]

total_templates = len(templates)
templates_per_job = int(np.ceil((len(templates)/n_jobs)))
last_job_templates = total_templates - templates_per_job * (n_jobs - 1)


template_start = templates_per_job * job_id

if job_id == n_jobs - 1:
	templates_per_job = last_job_templates
	print("last job, only doing {} templates".format(templates_per_job))

print("templates per job:", templates_per_job)
#templates_per_batch is the target number of templates to do per batch.
#the last batch will have equal or fewer templates.
templates_per_batch = 30

n_batches = int(np.ceil(templates_per_job/templates_per_batch))

print("batches per job (SHOULD BE 1):", n_batches)

n_noise_segments = len(valid_times)
total_noise_segments = n_noise_segments
#WARNING! SET TO A SMALL VALUE FOR TESTING
n_noise_segments = 50
#n_noise_segments = 10


window_size = 2048
sample_rate = 2048
stride = 128

start_cutoff = max_waveform_length
end_cutoff = duration - 24 #set so total length is a nice number
slice_duration = (end_cutoff - start_cutoff)

print("start cutoff: {}, end cutoff: {}, slice duration: {}".format(start_cutoff, end_cutoff, slice_duration))

n_windows = (slice_duration*sample_rate - window_size)//stride +1

print("n_windows:", n_windows)


print("initialisation took", time.time() - start, "seconds")

template_time = 0
noise_time = 0
mf_time = 0
window_time = 0
strain_time = 0
pred_time = 0
timeslide_time = 0

n_templates = templates_per_batch
t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)


from model_utils import split_models

ifo_dict = split_models()

#adding tensorflow stuff
import tensorflow as tf


global windowed_SNR
#global strain_np
global template_conj
#global template_norm


def strain2SNR(ifo):
	strain = TimeSeries(noise[ifos.index(ifo)], delta_t=delta_t, copy=False)
	strain = highpass(strain,f_lower).to_frequencyseries(delta_f=delta_f).data
	strain = np.array([strain])[:,kmin:kmax]
	strain_np = np.repeat(strain, n_templates, axis=0)
	#strain_time += time.time() - start

	#start = time.time()
	template_norm = np_sigmasq(t_templates, psds[ifo], N, kmin, kmax, delta_f)
	y = mf_in_place(strain_np, psds[ifo], N, kmin, kmax, template_conj, template_norm)

	print("y shape:", y.shape)
	#mf_time += time.time() - start

	#start = time.time()
	windowed = np.array(make_windows_2d(np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32), 
							window_size, stride))
	#window_time += time.time() - start
	#print("in the function, {} windowed_SNR starts with {}".format(ifo, windowed[0][0][:10]))
	return windowed



for i in range(n_batches):
	print("batch", i)
	template_start_idx = i*templates_per_batch + template_start

	start = time.time()
	if i == n_batches - 1:
		print("loading templates in range",template_start_idx, templates_per_job + template_start)
		n_templates =  (templates_per_job + template_start) - template_start_idx
		print("last templates for this job, only {}".format(n_templates))
	else:
		print("loading templates in range",template_start_idx, (i+1)*templates_per_batch + template_start)
		

	#load this batch's templates
	
	for j in range(n_templates):
		t_templates[j] = get_fd_waveform(mass1 = templates[template_start_idx + j,1], 
								   mass2 = templates[template_start_idx + j,2],
								   spin1z = templates[template_start_idx + j,3], 
								   spin2z = templates[template_start_idx + j,4],
								   approximant = approximant, f_lower = f_lower, 
								   delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]
	template_time += time.time() - start
	#reload the noise from the start

	noise_gen = noise_generator(valid_times, paths, file_list, duration, sample_rate)
	
	#windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)
	#strain_np = np.empty((n_templates, kmax-kmin), dtype=np.complex128)

	template_conj = np.conjugate(t_templates)
	#template_norm = np_sigmasq(t_templates, psds[ifos[0]], N, kmin, kmax, delta_f)

	preds = {"H1": [], "L1": []}

	for j in range(n_noise_segments):

		windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)

		start = time.time()
		print("noise segment", j)
		noise = next(noise_gen)
		noise_time += time.time() - start

		#template_norm = np_sigmasq(t_templates, psds[ifos[ifo]], N, kmin, kmax, delta_f)
		start = time.time()
		#with mp.Pool(n_cpus) as p:
		#	windowed_SNR = np.array(p.map(strain2SNR, ifos))
		#	#p.map(strain2SNR, ifos)

		mf_time += time.time() - start

		
		for ifo in range(len(ifos)):
			
			start = time.time()
			template_norm = np_sigmasq(t_templates, psds[ifos[ifo]], N, kmin, kmax, delta_f)

			strain = TimeSeries(noise[ifo], delta_t=delta_t, copy=False)
			strain = highpass(strain,f_lower).to_frequencyseries(delta_f=delta_f).data
			strain = np.array([strain])[:,kmin:kmax]
			strain_np = np.repeat(strain, n_templates, axis=0)
			strain_time += time.time() - start

			start = time.time()
			y = mf_in_place(strain_np, psds[ifos[ifo]], N, kmin, kmax, template_conj, template_norm)

			print("y shape:", y.shape)
			mf_time += time.time() - start

			start = time.time()
			windowed_SNR[ifo] = np.array(make_windows_2d(np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32), 
									window_size, stride))
			window_time += time.time() - start
			
		print("windowed_SNR starts with", windowed_SNR[:, 0, 0, :10])

		if j > 0 and (valid_times[j] - valid_times[j-1]) < slice_duration:
			print(int((valid_times[j] - valid_times[j-1]) * sample_rate/stride), "windows need to be discarded from start of sample", j)
			chop_index = int((valid_times[j] - valid_times[j-1]) * sample_rate/stride)
			chop_index = 0 #TODO: for now we're just ignoring this. If using ONNX we can't chop anyway because we need to fix the input size
		else:
			chop_index = 0

		start = time.time()
		#with mp.Pool(n_cpus) as p:
		#	results = p.map(distribute_preds, [[windowed_SNR[ifos.index(ifo)], ifo] for ifo in ifos] )
		
		with tf.device('/GPU:0'):
			for ifo in ifos:
				predstemp = ifo_dict[ifo].predict(windowed_SNR[ifos.index(ifo), :, chop_index: ].reshape(-1,2048), batch_size = 4096, verbose = 2)
				#the 2 needs to be the shape of the output of the 1 detector sub-models
				preds_reshaped = predstemp.reshape((windowed_SNR.shape[1], windowed_SNR.shape[2] - chop_index, 2))
				preds[ifo].append(preds_reshaped)
				#np.save("preds_{}_{}.npy".format(ifo, j), )
			
			#print("preds are using {} GB".format((j+1)*preds_reshaped.nbytes*2/1024**3))

		del windowed_SNR
		#del strain
		#del strain_np
		gc.collect()

		#np.save("preds_L1_{}.npy".format(j), results[1])

		pred_time += time.time() - start

	#timeslide stuff should actually go outside of the j loop i.e. we should run it only after we've collected
	#the whole week. Should only be ~1.1 GB for 25 templates
	predsh1 = np.concatenate((preds["H1"]), axis = 1)
	predsl1 = np.concatenate((preds["L1"]), axis = 1)

	print(predsh1.shape)
	start = time.time()
	
	combopreds = []
	#with tf.device('/GPU:0'):
	#	for j in range(100):
	#		predsl1roll = np.roll(predsl1, axis = 1, shift = j * 100)
	#		for k in range(len(predsh1)):
	#			combopreds.append(ifo_dict['combiner'].predict([predsh1[k],predsl1roll[k]], verbose = 0, batch_size = 32000))
	#	np.roll(preds["L1"], k, axis=1)
	#time.sleep(10)
	timeslide_time += time.time() - start


	print("pretending to send to triton")
	#time.sleep(1)

print("template loading took", template_time, "seconds")
print("noise loading took", noise_time, "seconds")
print("matched filtering took", mf_time, "seconds")
print("windowing took", window_time, "seconds")
print("prediction took", pred_time, "seconds")
print("timesliding took", timeslide_time, "seconds")

total_time = template_time + noise_time + mf_time + window_time + pred_time + timeslide_time

print("total time:", total_time, "seconds")

print("actual run would take {} hours".format((total_noise_segments/n_noise_segments) *total_time/3600))

