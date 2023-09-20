import time
start = time.time()

import sys
sys.path.append('/fred/oz016/alistair/GWSamplegen/')

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

from GWSamplegen.snr_utils_np import numpy_matched_filter, mf_in_place, np_sigmasq


noise_dir = "/fred/oz016/alistair/GWSamplegen/noise/O3_first_week_1024"
duration = 1024
sample_rate = 2048
delta_t = 1/sample_rate
f_lower = 30
f_final = 1024
delta_f = 1/duration
approximant = "TaylorF2"

def get_valid_noise_times(
	noise_dir: str,
	noise_len: int,
	min_step: int = 1,
	start_time: int = None,
	end_time: int = None,
) -> (List[int], np.ndarray, List[str]):
	"""multipurpose function to return a list of valid start times, list of noise file paths and deconstructed file names 
	
	noise_dir: directory containing noise files
	noise_len: minimum length of noise segments to consider
	start_time: if specified, the start of the time window to consider. Otherwise, all noise in noise_dir will be used.
	end_time: if specified, the end of the time window to consider

	returns:

	valid_times: list of valid start times for noise segments
	paths: array of deconstructed file names, giving detector info, segment start time and duration
	file_list: list of noise file paths in chronological order
	"""

	valid_times = np.array([])
	
	#get all strain file paths from the noise directory, then extract their start time and duration
	paths = os.listdir(noise_dir)
	paths = [path.split("-") for path in paths if len(path.split("-")) == 3]

	#paths[0] is the interferometer list
	#paths[1] is the start time
	#paths[2] is the duration

	ifo_list = paths[0][0]
	
	valid_paths = []
	for path in paths:
		if int(path[2][:-4]) >= noise_len:
			if start_time is not None and end_time is not None:
				if int(path[1]) <= start_time and int(path[1]) + int(path[2][:-4]) - start_time >= noise_len:
					valid_paths.append(path)
					print("path valid, starts before", path)
				
				elif int(path[1]) >= start_time and int(path[1]) + int(path[2][:-4]) <= end_time:
					valid_paths.append(path)
					print("path valid, contained", path)

				
				elif int(path[1]) < end_time and int(path[1]) + int(path[2][:-4]) - end_time >= noise_len:
					
					valid_paths.append(path)
					print("path valid, ends after", path)

				else:
					pass
					#print("path not valid", path)
			
			else:
				valid_paths.append(path)

	paths = valid_paths
	for path in paths:
		path[1] = int(path[1])
		path[2] = int(path[2][:-4])

		#print(path[1], path[2])

		times = np.arange(path[1], path[1]+path[2] - noise_len, min_step)
		if path[1] + path[2] - noise_len not in times:

			times = np.append(times, path[1] + path[2] - noise_len)

		valid_times = np.concatenate((valid_times,times))

		if start_time is not None and end_time is not None:
			valid_times = valid_times[(valid_times >= start_time) & (valid_times + noise_len <= end_time) ]
		
	#ensure the file paths are in chronological order
	paths = np.array(paths)
	paths = paths[np.argsort(paths[:,1])]

	valid_times = np.sort(valid_times)

	#reconstruct the file paths from the start times and ifo_list
	file_list = [noise_dir +"/"+ ifo_list +"-"+ path[1] +"-"+ path[2] +".npy" for path in paths]

	return valid_times, paths, file_list

WINDOW_SIZE = 2048
STEP = 128


def make_windows_2d(time_series_list, window_size=WINDOW_SIZE, step_size=STEP):
	"""
	Turns a list of 1D arrays into a 3D array of sequential labelled windows of window_size with horizon size label.
	"""
	# Convert the list of time series into a 2D numpy array
	time_series_array = np.array(time_series_list)

	num_series, series_length = time_series_array.shape

	#print(num_series, series_length)

	# Calculate the number of windows for each time series
	num_windows = (series_length - window_size) // step_size + 1
	#print(num_windows)

	# Calculate the strides for creating the windowed view
	strides = time_series_array.strides + (time_series_array.strides[-1] * step_size,)

	# Use as_strided to create a view of the data
	windowed_array = as_strided(
		time_series_array,
		shape=(num_series, num_windows, window_size),
		strides=(strides[0], strides[2], strides[1])
	)

	#     print()

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
templates = templates[:10000]

total_templates = len(templates)
templates_per_job = int(len(templates)/n_jobs)
last_job_templates = total_templates - templates_per_job * (n_jobs - 1)


template_start = templates_per_job * job_id

if job_id == n_jobs - 1:
	templates_per_job = last_job_templates
	print("last job")

print("templates per job:", templates_per_job)
#templates_per_batch is the target number of templates to do per batch.
#the last batch will have equal or fewer templates.
templates_per_batch = 50

n_batches = int(np.ceil(templates_per_job/templates_per_batch))

print("batches per job:", n_batches)

n_noise_segments = len(valid_times)
#WARNING! SET TO A SMALL VALUE FOR TESTING
n_noise_segments = 20



window_size = 2048
sample_rate = 2048
stride = 128

start_cutoff = max_waveform_length
end_cutoff = duration - 24 #set so total length is a nice number
slice_duration = (end_cutoff - start_cutoff)

n_windows = (slice_duration*sample_rate - window_size)//stride +1


print("initialisation took", time.time() - start, "seconds")





template_time = 0
noise_time = 0
mf_time = 0
window_time = 0
strain_time = 0

n_templates = templates_per_batch
t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)

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
	
	windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)
	strain_np = np.empty((n_templates, kmax-kmin), dtype=np.complex128)

	template_conj = np.conjugate(t_templates)
	template_norm = np_sigmasq(t_templates, psds[ifos[0]], N, kmin, kmax, delta_f)

	for j in range(n_noise_segments):

		start = time.time()
		print("noise segment", j)
		noise = next(noise_gen)
		noise_time += time.time() - start

		
		if j > 0 and (valid_times[j] - valid_times[j-1]) < slice_duration:
			print(int((valid_times[j] - valid_times[j-1]) * sample_rate/stride), "windows need to be discarded from start of sample", j)

		for ifo in range(len(ifos)):
			start = time.time()

			strain = TimeSeries(noise[ifo], delta_t=delta_t, copy=False)
			strain = highpass(strain,f_lower).to_frequencyseries(delta_f=delta_f).data
			strain = np.array([strain])[:,kmin:kmax]
			strain_np = np.repeat(strain, n_templates, axis=0)
			strain_time += time.time() - start

			start = time.time()
			y = mf_in_place(strain_np, psds[ifos[ifo]], N, kmin, kmax, template_conj, template_norm)
				   
			#y = numpy_matched_filter(strain_np, t_templates, psds[ifos[ifo]], 
			#				N, kmin, kmax, duration, delta_t = delta_t, flow = f_lower)
			#y = np.random.uniform(0,8,(n_templates, 1024*2048)).astype(np.complex128) + 1j*np.random.uniform(0,8,(n_templates, 1024*2048)).astype(np.complex128)
			#mf_save[j] = y[:,100*2048:1000*2048]
			#if ifo == 0:
			#	temp.append(np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32))
			#	noisetemp.append(noise[0])

			mf_time += time.time() - start

			start = time.time()
			windowed_SNR[ifo] = np.array(make_windows_2d(np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32), 
									window_size, stride))
			window_time += time.time() - start

		print("pretending to send to triton")
		time.sleep(1)

print("template loading took", template_time, "seconds")
print("noise loading took", noise_time, "seconds")
print("matched filtering took", mf_time, "seconds")
print("windowing took", window_time, "seconds")
