import time
import os
import numpy as np
import argparse
import gc
from queue import Queue
from functools import partial
import tritonclient.grpc as grpcclient
from GWSamplegen.noise_utils import load_psd, get_valid_noise_times
from GWSamplegen.waveform_utils import load_pycbc_templates
from GWSamplegen.snr_utils_np import np_get_cutoff_indices, mf_in_place, np_sigmasq
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.types import TimeSeries
from pycbc.filter import highpass



#infernus imports
from noise_utils import noise_generator
from SNR_utils import make_windows_2d
from model_utils import onnx_callback
start = time.time()


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
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--injectionfile', type=str, default=None)

args = parser.parse_args()

job_id = args.jobindex #job id in job array
worker_id = args.workerid #worker number of a server
n_workers = args.totalworkers
n_jobs = args.totaljobs
gpu_node = args.node
grpc_port = args.port + 1 #GRPC port is always 1 more than HTTP port
n_gpus = args.ngpus
injfile = args.injectionfile

myfolder = os.path.join(os.environ["JOBFS"], "job_" +str(job_id), "worker_"+str(worker_id))
print("my folder is", myfolder)

print("starting job {} of {}".format(job_id, n_jobs))
old_job_id = job_id
print("I am worker {} of {} for this server".format(worker_id, n_workers))
job_id = worker_id + job_id*n_workers
print("my unique index is {}".format(job_id))
n_jobs = n_jobs * n_workers
print("there are {} jobs in total".format(n_jobs))
print("I am using {} GPUs".format(n_gpus))








#ADDING TRITON STUFF


print("connecting to {} on port {}".format(gpu_node, grpc_port))
print("new GRPC port on", grpc_port+3)

#batch size for triton model
if n_gpus == 1:
	batch_size = 1024
else:
	batch_size = 512

model = "test-bns-1024" #the model used by a 1 gpu server
model = "new-hl-1024"

modelh = "test-h-512"   #the models used by a 2 gpu server
modelh = "new-h-1024"

modell = "test-l-512"
modell = "new-l-1024"

# Setting up client

triton_client = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port))
if n_gpus == 2:
	triton_client2 = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port+3))


dummy_data = np.random.normal(size=(batch_size, 2048,1)).astype(np.float32)
inputh = grpcclient.InferInput("h", dummy_data.shape, datatype="FP32")
inputl = grpcclient.InferInput("l", dummy_data.shape, datatype="FP32")

#handle both one and two model cases
output = grpcclient.InferRequestedOutput("concat")
outputh = grpcclient.InferRequestedOutput("h_out")
outputl = grpcclient.InferRequestedOutput("l_out")


callback_q = Queue()


#MAKING JOB SMALLER
#templates = templates[:n_jobs*30*10]
#templates = templates[:1487]
templates = templates[:]

total_templates = len(templates)

print("total templates:", total_templates)
templates_per_job = int(np.ceil((len(templates)/n_jobs)))
main_job_templates = templates_per_job
last_job_templates = total_templates - templates_per_job * (n_jobs - 1)

total_lastjob = last_job_templates + templates_per_job
print("total templates in last job:", total_lastjob)


template_start = templates_per_job * job_id

#if job_id == n_jobs - 1:
#	templates_per_job = last_job_templates
#	print("last job, only doing {} templates".format(templates_per_job))

if old_job_id == n_jobs/n_workers - 1:
	print("I'm a worker in the last job.")
	if worker_id == 0:
		templates_per_job = int(np.ceil(total_lastjob/2))
		template_start = total_templates - total_lastjob
		
	if worker_id == 1:
		templates_per_job = int(np.floor(total_lastjob/2))
		template_start = total_templates - templates_per_job


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


#TODO: we should save a .json file with all the configs that the cleanup job needs
#this includes template IDs, inference rate (i.e. 16 Hz), and the number of noise segments

json_dict = {
	"template_start": template_start,
	"inference_rate": sample_rate//stride,
	"n_noise_segments": n_noise_segments,
	"n_batches": n_batches,
}

import json
with open(os.path.join(myfolder, "args.json"), "w") as f:
	json.dump(json_dict, f)
	

print("initialisation took", time.time() - start, "seconds")

template_time = 0
noise_time = 0
mf_time = 0
window_time = 0
strain_time = 0
pred_time = 0
timeslide_time = 0
reshape_time = 0
wait_time = 0

n_templates = templates_per_batch
t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)


global windowed_SNR
global template_conj

#flush prints
import sys
sys.stdout.flush()

total_batches_sent = 0

for i in range(n_batches):
	print("batch", i)
	template_start_idx = i*templates_per_batch + template_start

	start = time.time()
	if i == n_batches - 1:
		print("loading templates in range",template_start_idx, templates_per_job + template_start)
		n_templates =  (templates_per_job + template_start) - template_start_idx
		t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)
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
		n_windows = (slice_duration*sample_rate - window_size)//stride +1
		nonwindowed_SNR = np.empty((len(ifos), n_templates, slice_duration*sample_rate), dtype=np.float32)
		windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)

		start = time.time()
		print("noise segment", j)
		noise = next(noise_gen)
		noise_time += time.time() - start
		
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
			nonwindowed_SNR[ifo, :, :] = np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32)

			#print("y shape:", y.shape)
			mf_time += time.time() - start


			

		#print("windowed_SNR starts with", windowed_SNR[:, 0, 0, :10])

		if j > 0 and (valid_times[j] - valid_times[j-1]) < slice_duration:
			chop_time = int((valid_times[j] - valid_times[j-1]) * sample_rate)
			print("chopping", chop_time/sample_rate, "seconds")

			chop_index = int(((valid_times[j] - valid_times[j-1]) * sample_rate/stride))
			print("only windows {} onwards of sample {} are needed".format(chop_index, j))

			nonwindowed_SNR = nonwindowed_SNR[:, :, chop_time:]
			n_windows = (nonwindowed_SNR.shape[-1] - window_size)//stride +1
			print("n_windows:", n_windows)
			windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)
			start = time.time()
		
			for ifo in range(len(ifos)):
				windowed_SNR[ifo] = np.array(make_windows_2d(nonwindowed_SNR[ifo, :, :], window_size, stride))
			window_time += time.time() - start
				
		else:
			chop_index = 0
			chop_time = 0

			start = time.time()
			for ifo in range(len(ifos)):
				windowed_SNR[ifo] = np.array(make_windows_2d(nonwindowed_SNR[ifo, :, :], window_size, stride))
			window_time += time.time() - start

		if i == 0 and j == 0:
			while True:
				try:
					if (n_gpus == 1 and triton_client.is_server_ready()) or \
					   (n_gpus == 2 and triton_client.is_server_ready() and triton_client2.is_server_ready()) :

						#print("worker {} will sleep for {} seconds".format(worker_id, worker_id*5))
						#time.sleep(worker_id*5)
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

		#windowed_SNR = windowed_SNR[:, :, chop_index:]

		

		#print("pre reshaping:", windowed_SNR[:,0,0,0])
		#reshape into (2, flattened_windowed_SNR, 2048)
		windowed_SNR = windowed_SNR.reshape(2, -1, 2048)
		newshape = windowed_SNR.shape

		#print("post reshaping:", windowed_SNR[:,0,0])
		reshape_time += time.time() - start

		tritonbatches = int(np.ceil(windowed_SNR.shape[1]/batch_size))
		total_batches_sent += n_workers* tritonbatches #to account for 2 workers


		#workers wait before sending their first batch so that the server processes them in the correct order

		start = time.time()

		if n_gpus == 1:
			successes = triton_client.get_inference_statistics(model).model_stats[0].inference_stats.success.count
		else:
			successes = triton_client.get_inference_statistics(modelh).model_stats[0].inference_stats.success.count
		
		if worker_id == 0:
			reqbatches = int(total_batches_sent - 2.3 * tritonbatches)
		if worker_id == 1:
			reqbatches = int(total_batches_sent - 1.3 * tritonbatches)
		#if worker_id == 2:
		#	reqbatches = int(total_batches_sent - 0.3 * tritonbatches)
		if old_job_id == n_jobs/n_workers - 1 and i == n_batches - 1 and worker_id == 0:
			#if worker_id == 0:
			reqbatches -= tritonbatches * 0.5
			print("removing a small amount of required batches from worker 0")
			#if i >= int(np.ceil(last_job_templates/templates_per_batch)):
			#	#in this case, worker 0 will have to send more batches than worker 1 and so we can send immediately
			#	reqbatches = 0
			#	print("this worker has free reign to send as many batches as it wants now")
			#elif i == int(np.ceil(last_job_templates/templates_per_batch)) - 1:
			#	#in this case, worker 1 has fewer templates and so we need to reduce worker 0's sending requirement
			#	reqbatches -= tritonbatches * 0.5
			#print("sending in whatever order, as one of these jobs may have fewer templates")

					
		while successes < reqbatches:
			print("worker waiting, only {} successes. we need {}".format(successes, reqbatches))
			#print("queue size:",triton_client.get_inference_statistics(model).model_stats[0].inference_stats.nv_inference_pending_request_count.count)
			sys.stdout.flush()
			time.sleep(1)
			if n_gpus == 1:
				successes = triton_client.get_inference_statistics(model).model_stats[0].inference_stats.success.count
			else:
				successes = triton_client.get_inference_statistics(modelh).model_stats[0].inference_stats.success.count

		wait_time += time.time() - start


		predstart = time.time()

		bufsize = 0
		for k in range(tritonbatches):
			if k == tritonbatches - 1:
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
			request_id_h = str(k) + "_" + str(bufsize) + '_h'
			request_id_l = str(k) + "_" + str(bufsize) + '_l'
			
			if n_gpus == 1:
				triton_client.async_infer(model_name=model, inputs=[inputh, inputl], outputs=[output],
								request_id=request_id, callback=partial(onnx_callback,callback_q))
				total_batches += 1

			else:
				triton_client.async_infer(model_name=modelh, inputs=[inputh], outputs=[outputh],
								request_id=request_id_h, callback=partial(onnx_callback,callback_q))
				
				triton_client2.async_infer(model_name=modell, inputs=[inputl], outputs=[outputl],
								request_id=request_id_l, callback=partial(onnx_callback,callback_q))
				total_batches += 2

			
			
				
			#print("taking a small break from sending")
			#print("queue size:",triton_client.get_inference_statistics().model_stats[0].inference_stats.queue.count)
			#time.sleep(0.1)
		
		print("sending time:", time.time()- predstart)
		start = time.time()

		del windowed_SNR
		#del strain
		#del strain_np
		#gc.collect()

		print("total batches sent:", total_batches)
		all_responses = []
		#responses_h  = []
		#responses_l  = []

		count = 0
		while True:
			count += 1
			if count <= total_batches:
				#print(count)
				response = callback_q.get()
				all_responses.append(response)
				#responseh = queue_h.get()
				#responses_h.append(responseh)
				#responsel = queue_l.get()
				#responses_l.append(responsel)
			else:
				break
		print(f"infer time: {time.time() - start}")

		pred_time += time.time() - predstart

		det_output_shape = all_responses[0][0].shape[-1]
		if i == 0 and j == 0:
			print("det output shape:", det_output_shape)

		start = time.time()
		#newshape
		#should have shape (n_templates, n_windows, 4). TODO: handle arbitrary prediction length, not just 2 per det
		if n_gpus == 1:
			predbuf = np.empty((newshape[1], det_output_shape), dtype=np.float32)
		else:
			predbuf = np.empty((newshape[1], det_output_shape*2), dtype=np.float32)

		for r in range(len(all_responses)):
			response_header = all_responses[r][1].split("_")
			idx = int(response_header[0])
			bufsize = int(response_header[1])
			if n_gpus == 1:
				if bufsize == 0:
					predbuf[idx*batch_size : (idx+1)*batch_size] = all_responses[r][0]
				else:
					predbuf[idx*batch_size :] = all_responses[r][0][:batch_size-bufsize]

			elif n_gpus == 2:
				ifo = response_header[2]
				if bufsize == 0:
					if ifo == "h":
						predbuf[idx*batch_size : (idx+1)*batch_size, :det_output_shape] = all_responses[r][0]
					else:
						predbuf[idx*batch_size : (idx+1)*batch_size, det_output_shape:] = all_responses[r][0]
				else:
					if ifo == "h":
						predbuf[idx*batch_size : (idx+1)*batch_size, :det_output_shape] = all_responses[r][0][:batch_size-bufsize]
					else:
						predbuf[idx*batch_size : (idx+1)*batch_size, det_output_shape:] = all_responses[r][0][:batch_size-bufsize]

					

		
		predbuf = predbuf.reshape(n_templates, -1, predbuf.shape[-1])
		print("predbuf shape",predbuf.shape)
		
		

		#save to my folder
		np.save(os.path.join(myfolder, "SNR_batch_{}_segment_{}.npy".format(i, j)), nonwindowed_SNR)
		np.save(os.path.join(myfolder, "preds_batch_{}_segment_{}.npy".format(i, j)), predbuf)

		#np.save(os.path.join("/fred/oz016/alistair/infernus/timeslides/", "SNR_batch_{}_segment_{}_worker{}.npy".format(i, j,job_id)), nonwindowed_SNR)
		#np.save(os.path.join("/fred/oz016/alistair/infernus/timeslides/", "preds_batch_{}_segment_{}_worker{}.npy".format(i, j,job_id)), predbuf)
		
		
		timeslide_time += time.time() - start
		#print("timeslide time:", time.time() - start)
		

		del all_responses
		#del responses_h
		#del responses_l

		gc.collect()

		sys.stdout.flush()


print("template loading took", template_time, "seconds")
print("noise loading took", noise_time, "seconds")
print("strain generation took", strain_time, "seconds")
print("matched filtering took", mf_time, "seconds")
print("windowing took", window_time, "seconds")
print("prediction took", pred_time, "seconds")
print("timesliding took", timeslide_time, "seconds")
print("reshaping took", reshape_time, "seconds")
print("waiting took", wait_time, "seconds")

total_time = template_time + noise_time + mf_time + window_time + strain_time + pred_time + timeslide_time + reshape_time + wait_time

print("total time:", total_time, "seconds")

print("actual run would take {} hours".format((total_noise_segments/n_noise_segments) *total_time/3600))

