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






#REGULAR SNR SERIES STUFF


noise_dir = "/fred/oz016/alistair/GWSamplegen/noise/O3_first_week_1024"
duration = 1024
sample_rate = 2048
delta_t = 1/sample_rate
f_lower = 30
f_final = 1024
delta_f = 1/duration
approximant = "TaylorF2"

#import multiprocessing as mp
#n_cpus = 2

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
parser.add_argument('--jobindex', type=int)
parser.add_argument('--workerid', type=int, default=0)
parser.add_argument('--totalworkers', type=int, default=1)
parser.add_argument('--totaljobs', type=int, default=1)
parser.add_argument('--node', type=str, default="john108")
parser.add_argument('--port', type=int, default=8001)

args = parser.parse_args()

print(args)
job_id = args.jobindex #job id in job array
worker_id = args.workerid #worker number of a server
n_workers = args.totalworkers
n_jobs = args.totaljobs
gpu_node = args.node
grpc_port = args.port + 1 #GRPC port is always 1 more than HTTP port

print("starting job {} of {}".format(job_id, n_jobs))
print("I am worker {} of {} for this server".format(worker_id, n_workers))
job_id = worker_id + job_id*n_workers
print("my unique index is {}".format(job_id))
n_jobs = n_jobs * n_workers
print("there are {} jobs in total".format(n_jobs))



#ADDING TRITON STUFF





from functools import partial
import time
import os
from queue import Empty, Queue
from tqdm import tqdm
from typing import Optional
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.grpc._infer_result import InferResult
from tritonclient.utils import InferenceServerException

#gpu_node = "john108"
#gpu_node = 
print("connecting to {} on port {}".format(gpu_node, grpc_port))

#batch size for triton model
batch_size = 512
model = "test-bns-512"

# Setting up client
triton_client = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port))

dummy_data = np.random.normal(size=(batch_size, 2048,1)).astype(np.float32)
inputh = grpcclient.InferInput("h", dummy_data.shape, datatype="FP32")
inputl = grpcclient.InferInput("l", dummy_data.shape, datatype="FP32")
output = grpcclient.InferRequestedOutput("concatenate")


callback_q = Queue()

def onnx_callback(
    queue: Queue,
    result: InferResult,
    error: Optional[InferenceServerException]
) -> None:
    """
    Callback function to manage the results from 
    asynchronous inference requests and storing them to a  
    queue.

    Args:
        queue: Queue
            Global variable that points to a Queue where 
            inference results from Triton are written to.
        result: InferResult
            Triton object containing results and metadata 
            for the inference request.
        error: Optional[InferenceServerException]
            For successful inference, error will return 
            `None`, otherwise it will return an 
            `InferenceServerException` error.
    Returns:
        None
    Raises:
        InferenceServerException:
            If the connected Triton inference request 
            returns an error, the exception will be raised 
            in the callback thread.
    """
    try:
        if error is not None:
            raise error

        request_id = str(result.get_response().id)

        # necessary when needing only one number of 2D output
        #np_output = {}
        #for output in result._result.outputs:
        #    np_output[output.name] = result.as_numpy(output.name)[:,1]

        # only valid when one output layer is used consistently
        output = list(result._result.outputs)[0].name
        np_outputs = result.as_numpy(output)

        response = (np_outputs, request_id)

        if response is not None:
            queue.put(response)

    except Exception as ex:
        print("Exception in callback")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)





















#MAKING JOB SMALLER
templates = templates[:n_jobs*30*10]

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

print("batches per job:", n_batches)

n_noise_segments = len(valid_times)
total_noise_segments = n_noise_segments
#WARNING! SET TO A SMALL VALUE FOR TESTING
#n_noise_segments = 50
n_noise_segments = 10


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
reshape_time = 0

n_templates = templates_per_batch
t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)


#from model_utils import split_models

#ifo_dict = split_models()

#adding tensorflow stuff
#import tensorflow as tf


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

	#print("y shape:", y.shape)
	#mf_time += time.time() - start

	#start = time.time()
	windowed = np.array(make_windows_2d(np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32), 
							window_size, stride))
	#window_time += time.time() - start
	#print("in the function, {} windowed_SNR starts with {}".format(ifo, windowed[0][0][:10]))
	return windowed

#flush prints
import sys
sys.stdout.flush()

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

			#print("y shape:", y.shape)
			mf_time += time.time() - start

			start = time.time()
			windowed_SNR[ifo] = np.array(make_windows_2d(np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32), 
									window_size, stride))
			window_time += time.time() - start
			
		#print("windowed_SNR starts with", windowed_SNR[:, 0, 0, :10])

		if j > 0 and (valid_times[j] - valid_times[j-1]) < slice_duration:
			print(int((valid_times[j] - valid_times[j-1]) * sample_rate/stride), "windows need to be discarded from start of sample", j)
			chop_index = int((valid_times[j] - valid_times[j-1]) * sample_rate/stride)
			#chop_index = 0 #TODO: for now we're just ignoring this. If using ONNX we can't chop anyway because we need to fix the input size
		else:
			chop_index = 0

		if i == 0 and j == 0:
			while True:
				try:
					if triton_client.is_server_ready():
						print("job {} will sleep for {} seconds".format(job_id, job_id*5))
						time.sleep(job_id*5)
						break
					else:
						print("waiting for server to be live")
						time.sleep(1)
				except Exception as e:
					
					print("waiting for server to be live")
					time.sleep(1)

		start = time.time()
		#with mp.Pool(n_cpus) as p:
		#	results = p.map(distribute_preds, [[windowed_SNR[ifos.index(ifo)], ifo] for ifo in ifos] )
		
		#print("triton send shape:", windowed_SNR[0]")
		#print("example windowed SNR:", windowed_SNR[0, 0, 0:batch_size, :])
		print("example shape:", windowed_SNR[0, 0, 0:batch_size, :].shape)
		total_batches = 0

		windowed_SNR = windowed_SNR[:, :, chop_index:]

		newshape = windowed_SNR.shape

		#print("pre reshaping:", windowed_SNR[:,0,0,0])
		#reshape into (2, flattened_windowed_SNR, 2048)
		windowed_SNR = windowed_SNR.reshape(2, -1, 2048)

		#print("post reshaping:", windowed_SNR[:,0,0])
		reshape_time += time.time() - start
		start = time.time()

		bufsize = 0
		for k in range(int((np.ceil(windowed_SNR.shape[1]/batch_size)))):
			if k == int((np.ceil(windowed_SNR.shape[1]/batch_size))) - 1:
				#we may need to pad with zeroes to get to batch_size
				bufsize = batch_size - windowed_SNR.shape[1] % batch_size
				print("padding with", bufsize, "windows")
				hbuf = np.pad(windowed_SNR[0, k*batch_size:], ((0, bufsize), (0,0)), 'constant', constant_values = 0)
				lbuf = np.pad(windowed_SNR[1, k*batch_size:], ((0, bufsize), (0,0)), 'constant', constant_values = 0)
			
			else:
				hbuf = windowed_SNR[0, k*batch_size: (k+1)*batch_size]
				lbuf = windowed_SNR[1, k*batch_size: (k+1)*batch_size]
			
			inputh.set_data_from_numpy(np.expand_dims(hbuf, -1))
			inputl.set_data_from_numpy(np.expand_dims(lbuf, -1))

			request_id = str(k) + "_" + str(bufsize)

			triton_client.async_infer(model_name=model, inputs=[inputh, inputl], outputs=[output],
							 request_id=request_id, callback=partial(onnx_callback,callback_q))
			
			total_batches += 1
			
				
			print("taking a small break from sending")
			print("queue size:",triton_client.get_inference_statistics().model_stats[0].inference_stats.queue.count)
			time.sleep(0.5)


		"""
		bufsize = 0
		for k in range(n_templates):
			for l in range(bufsize, windowed_SNR.shape[2], batch_size):
				#print(l)
				if bufsize != 0:
					print("some of this batch was processed previously. starting from {}".format(bufsize))
				hbuf = np.expand_dims(windowed_SNR[0, k, chop_index + l: chop_index + l+batch_size, :], -1)
				lbuf = np.expand_dims(windowed_SNR[1, k, chop_index + l: chop_index + l+batch_size, :], -1)
				#pad end of output_buffer to a multiple of batch_size if necessary
				bufsize =  hbuf.shape[0] % batch_size
				print("bufsize:", bufsize)
				if bufsize != 0:
					#instead of padding with zeroes, grab the next template's data and pad to batch_size
					hbuf = np.concatenate((hbuf, np.expand_dims(windowed_SNR[0, k+1, chop_index: chop_index+bufsize, :], -1)), axis=0)
					lbuf = np.concatenate((lbuf, np.expand_dims(windowed_SNR[1, k+1, chop_index: chop_index+bufsize, :], -1)), axis=0)
					print("padding with {} windows from next template".format(bufsize))
					#hbuf = np.pad(hbuf, ((0, bufsize), (0,0), (0,0)), 'constant', constant_values = 0)
					#lbuf = np.pad(lbuf, ((0, bufsize), (0,0), (0,0)), 'constant', constant_values = 0)
					print("hbuf shape:", hbuf.shape)
				inputh.set_data_from_numpy(hbuf)
				inputl.set_data_from_numpy(lbuf)
				request_id = str(k) + "_" + str(l)
				#TODO make request ID more informative
				triton_client.async_infer(model_name=model, inputs=[inputh, inputl], outputs=[output], 
											  request_id=request_id, callback=partial(onnx_callback,callback_q))
				#triton_client.async_infer(model_name=model, inputs=[inputh], outputs=[output], 
				#							  request_id=request_id, callback=partial(onnx_callback,callback_q))
				#time.sleep(0.0001)
				total_batches += 1
		"""
		del windowed_SNR
		#del strain
		#del strain_np
		gc.collect()

		print("total batches sent:", total_batches)
		all_responses = []
		count = 0
		while True:
			count += 1
			if count <= total_batches:
				#print(count)
				response = callback_q.get()
				all_responses.append(response)
		#         print(response)
			else:
				break
		print(f"infer time: {time.time() - start}")

		del all_responses
		gc.collect()
		#triton_client.get_inference_statistics().model_stats[0].inference_stats.success.count
		print("queue size:",triton_client.get_inference_statistics().model_stats[0].inference_stats.queue.count)

		#output_buffer = windowed_SNR[ifos.index(ifo), :, chop_index: ].reshape(-1,2048)
		#pad end of output_buffer to a multiple of 512
		#print("output buffer shape:", output_buffer.shape)
		#bufsize = 512 - output_buffer.shape[0] % 512
		#print("padding with", bufsize, "zeros")

		#output_buffer = np.pad(output_buffer, ((0, bufsize), (0,0)), 'constant', constant_values = 0)


		#with tf.device('/GPU:0'):
		#	for ifo in ifos:
		#		predstemp = ifo_dict[ifo].predict(windowed_SNR[ifos.index(ifo), :, chop_index: ].reshape(-1,2048), batch_size = 4096, verbose = 2)
		#		#the 2 needs to be the shape of the output of the 1 detector sub-models
		#		preds_reshaped = predstemp.reshape((windowed_SNR.shape[1], windowed_SNR.shape[2] - chop_index, 2))
		#		preds[ifo].append(preds_reshaped)
		#		#np.save("preds_{}_{}.npy".format(ifo, j), )
		#	
		#	#print("preds are using {} GB".format((j+1)*preds_reshaped.nbytes*2/1024**3))

		#del windowed_SNR
		#del strain
		#del strain_np
		#gc.collect()

		#np.save("preds_L1_{}.npy".format(j), results[1])

		pred_time += time.time() - start

		sys.stdout.flush()

	#timeslide stuff should actually go outside of the j loop i.e. we should run it only after we've collected
	#the whole week. Should only be ~1.1 GB for 25 templates
	#predsh1 = np.concatenate((preds["H1"]), axis = 1)
	#predsl1 = np.concatenate((preds["L1"]), axis = 1)

	#print(predsh1.shape)
	start = time.time()
	
	#combopreds = []
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
print("reshaping took", reshape_time, "seconds")

total_time = template_time + noise_time + mf_time + window_time + pred_time + timeslide_time + reshape_time

print("total time:", total_time, "seconds")

print("actual run would take {} hours".format((total_noise_segments/n_noise_segments) *total_time/3600))

