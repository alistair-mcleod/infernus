import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import argparse
import gc
import json
import h5py
from queue import Queue
from functools import partial
import tritonclient.grpc as grpcclient
from GWSamplegen.noise_utils import load_psd, get_valid_noise_times, load_gps_blacklist
from GWSamplegen.waveform_utils import load_pycbc_templates
from GWSamplegen.snr_utils_np import np_get_cutoff_indices, mf_in_place, np_sigmasq
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.types import TimeSeries
from pycbc.filter import highpass


#adding memory debugging
import tracemalloc
from collections import Counter
import linecache

def display_top(snapshot, key_type='lineno', limit=5):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f MiB"
              % (index, filename, frame.lineno, stat.size / (1024)**2))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

#tracemalloc.start()

#infernus imports
from noise_utils import noise_generator
from SNR_utils import make_windows_2d
from model_utils import onnx_callback
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--jobindex', type=int)
parser.add_argument('--workerid', type=int, default=0)
parser.add_argument('--totalworkers', type=int, default=1)
parser.add_argument('--totaljobs', type=int, default=1)
parser.add_argument('--node', type=str, default="john108")
parser.add_argument('--port', type=int, default=8001)
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--argsfile', type=str, default=None)

args = parser.parse_args()

job_id = args.jobindex #job id in job array
worker_id = args.workerid #worker number of a server
n_workers = args.totalworkers
n_jobs = args.totaljobs
gpu_node = args.node
grpc_port = args.port + 1 #GRPC port is always 1 more than HTTP port
n_gpus = args.ngpus
#injfile = args.injfile
argsfile = args.argsfile

#maxnoisesegs = args.maxnoisesegs

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

#read args from file

args = json.load(open(argsfile, "r"))
noise_dir = args["noise_dir"]
maxnoisesegs = args["max_noise_segments"]
template_bank_dir = args["template_bank_dir"]
template_bank_name = args["template_bank_name"]
duration = args["duration"]
sample_rate = args["sample_rate"]
f_lower = args["f_lower"]
fd_approximant = args["fd_approximant"]
td_approximant = args["td_approximant"]

#injfile can take 3 valid values: "None", which leads to a background run, 
# "noninj" which leads to a foreground run with no injections, 
# and a path to an injection file, which leads to a foreground run with injections.

injfile = args["injfile"]

if noise_dir == "None":
	#exit
	print("no noise dir specified, breaking")
	sys.exit(1)

if injfile == "None":
	injfile = None



#REGULAR SNR SERIES STUFF

print(noise_dir)
delta_t = 1/sample_rate
f_final = sample_rate//2
delta_f = 1/duration


#WINDOW_SIZE = 2048
#STEP = 128

valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, 900, blacklisting = False)
templates, _, _= load_pycbc_templates(template_bank_name, template_bank_dir)

N = int(duration/delta_t)
kmin, kmax = np_get_cutoff_indices(f_lower, None, delta_f, N)


ifos = ["H1", "L1"]

psds = load_psd(noise_dir, duration,ifos , f_lower, int(1/delta_t))


for psd in psds:
	psds[psd] = psds[psd][kmin:kmax]



hp, _ = get_td_waveform(mass1 = templates[0,1], mass2 = templates[0,2], 
						delta_t = delta_t, f_lower = f_lower, approximant = td_approximant)

max_waveform_length = len(hp)/sample_rate
max_waveform_length = max(32, int(np.ceil(max_waveform_length/10)*10))


#injection file setup stuff

from pycbc.detector import Detector
from GWSamplegen.waveform_utils import t_at_f

all_detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

if injfile is not None and injfile != "noninj":
	print("using injection file", injfile)
	f = h5py.File(injfile, 'r')
	mask = (f['injections']['gps_time'][:] > valid_times[0]) & (f['injections']['gps_time'][:] < valid_times[-1] + duration)
	n_injs = np.sum(mask)

	gps = f['injections']['gps_time'][mask]
	mass1 = f['injections']['mass1_source'][mask] * (1 + f['injections']['redshift'][mask]) #TODO: simplify by replacing with detector frame masses
	mass2 = f['injections']['mass2_source'][mask] * (1 + f['injections']['redshift'][mask])
	spin1x = f['injections']['spin1x'][mask]
	spin1y = f['injections']['spin1y'][mask]
	spin1z = f['injections']['spin1z'][mask]
	spin2x = f['injections']['spin2x'][mask]
	spin2y = f['injections']['spin2y'][mask]
	spin2z = f['injections']['spin2z'][mask]
	distance = f['injections']['distance'][mask]
	inclination = f['injections']['inclination'][mask]
	polarization = f['injections']['polarization'][mask]
	right_ascension = f['injections']['right_ascension'][mask]
	declination = f['injections']['declination'][mask]
	optimal_snr_h = f['injections']['optimal_snr_h'][mask]
	optimal_snr_l = f['injections']['optimal_snr_l'][mask]

	startgps = []
	for i in range(n_injs):
		startgps.append(np.floor(gps[i] - t_at_f(mass1[i], mass2[i], f_lower)))

	startgps = np.array(startgps)

	lgps = gps + all_detectors['L1'].time_delay_from_detector(all_detectors['H1'], 
												right_ascension, 
												declination, 
												gps)

	gps_dict = {'H1': gps, 'L1': lgps}



#ADDING TRITON STUFF


print("connecting to {} on port {}".format(gpu_node, grpc_port))
print("new GRPC port on", grpc_port+3)

#batch size for triton model
if n_gpus == 1:
	batch_size = 1024
else:
	batch_size = 1024
	#batch_size = 512

#model = "test-bns-1024" #the model used by a 1 gpu server
#model = "new-hl-1024"
model = "model_hl"

#modelh = "test-h-512"   #the models used by a 2 gpu server
#modelh = "new-h-1024"
modelh = 'model_h'

#modell = "test-l-512"
#modell = "new-l-1024"
modell = 'model_l'

# Setting up client

triton_client = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port))
if n_gpus == 2:
	triton_client2 = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port+3))


#dummy_data = np.random.normal(size=(batch_size, 2048,1)).astype(np.float32)

inputh = grpcclient.InferInput("h", (batch_size, 2048, 1), datatype="FP32")
inputl = grpcclient.InferInput("l", (batch_size, 2048, 1), datatype="FP32")

#handle both one and two model cases
output = grpcclient.InferRequestedOutput("concat")
outputh = grpcclient.InferRequestedOutput("h_out")
outputl = grpcclient.InferRequestedOutput("l_out")


callback_q = Queue()


def initialise_server(
	gpu_node: str, 
	grpc_port: str, 
	model: str = 'model_hl', 
	modelh: str = 'model_h', 
	modell: str = 'model_l'
) -> (grpcclient.InferenceServerClient, grpcclient.InferenceServerClient,
	  grpcclient.InferInput, grpcclient.InferInput, grpcclient.InferRequestedOutput,
	  grpcclient.InferRequestedOutput, grpcclient.InferRequestedOutput):
	
	"Return all the triton client and input/output objects needed for running the inference server."
	
	triton_client = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port))
	triton_client2 = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port+3))

	inputh = grpcclient.InferInput("h", (batch_size, 2048, 1), datatype="FP32")
	inputl = grpcclient.InferInput("l", (batch_size, 2048, 1), datatype="FP32")
	output = grpcclient.InferRequestedOutput("concat")
	outputh = grpcclient.InferRequestedOutput("h_out")
	outputl = grpcclient.InferRequestedOutput("l_out")

	return triton_client, triton_client2, inputh, inputl, output, outputh, outputl

#triton_client, triton_client2, inputh, inputl, output, outputh, outputl = initialise_server(gpu_node, grpc_port)


#MAKING JOB SMALLER
if n_jobs <= 40:
	#this is used in testing, as for full runs we use more jobs. 
	#Comment out if you want to run the full template bank with fewer jobs
	templates = templates[:n_jobs*30*5]
	print("only doing {} templates, because this is a test run".format(len(templates)))

#templates = templates[:1487]
templates = templates[:]

total_templates = len(templates)

print("total templates:", total_templates)
templates_per_job = int(np.ceil((len(templates)/n_jobs)))
main_job_templates = templates_per_job
#last_job_templates = total_templates - templates_per_job * (n_jobs - 1)

#total_lastjob = last_job_templates + templates_per_job
total_lastjob = total_templates - templates_per_job * (n_jobs - n_workers)

print("total templates in last job:", total_lastjob)


template_start = templates_per_job * job_id

#if job_id == n_jobs - 1:
#	templates_per_job = last_job_templates
#	print("last job, only doing {} templates".format(templates_per_job))


# if old_job_id == n_jobs/n_workers - 1:
# 	print("I'm a worker in the last job.")
# 	if worker_id == 0:
# 		templates_per_job = int(np.ceil(total_lastjob/2))
# 		template_start = total_templates - total_lastjob
		
# 	if worker_id == 1:
# 		templates_per_job = int(np.floor(total_lastjob/2))
# 		template_start = total_templates - templates_per_job

if old_job_id == n_jobs/n_workers - 1:
	template_start = total_templates - total_lastjob
	for i in range(n_workers):

		templates_per_job = int(np.floor(total_lastjob/n_workers))
		if i < total_lastjob%n_workers:
			templates_per_job += 1
		if worker_id == i:
			print("I'm a worker in the last job.")
			print(i)
			print(templates_per_job)
			print(template_start)
			break
		template_start += templates_per_job


print("templates per job:", templates_per_job)
#templates_per_batch is the target number of templates to do per batch.
#the last batch will have equal or fewer templates.
templates_per_batch = 30

n_batches = int(np.ceil(templates_per_job/templates_per_batch))

print("batches per job:", n_batches)


n_noise_segments = len(valid_times)
total_noise_segments = n_noise_segments
#WARNING! SET TO A SMALL VALUE FOR TESTING
#n_noise_segments = 100
#n_noise_segments = 3
if maxnoisesegs is not None and maxnoisesegs < n_noise_segments:
	n_noise_segments = maxnoisesegs
	print("only doing {} noise segments".format(n_noise_segments))
print("total noise segments:", n_noise_segments)

noise_seg_start = 0
if maxnoisesegs is not None and maxnoisesegs == 123:
	print("Special run, should only be a one-off")
	n_noise_segments = len(valid_times)
	noise_seg_start = 120

window_size = 2048
stride = 128

start_cutoff = max_waveform_length
end_cutoff = duration - 24 #set so total length is a nice number
slice_duration = (end_cutoff - start_cutoff)

print("start cutoff: {}, end cutoff: {}, slice duration: {}".format(start_cutoff, end_cutoff, slice_duration))

n_windows = (slice_duration*sample_rate - window_size)//stride +1

print("n_windows:", n_windows)

#TODO: add the ability to load the gps blacklist from a different file
json_dict = {
	"template_start": template_start,
	"inference_rate": sample_rate//stride,
	"n_noise_segments": n_noise_segments,
	"n_batches": n_batches,
	"injection_file": 1 if injfile else 0,
	"valid_times": valid_times.tolist(),
	"gps_blacklist": load_gps_blacklist(f_lower).tolist()
}



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
								   approximant = fd_approximant, f_lower = f_lower, 
								   delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]
	template_time += time.time() - start
	#reload the noise from the start

	noise_gen = noise_generator(valid_times, paths, file_list, duration, sample_rate)
	
	#windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)
	#strain_np = np.empty((n_templates, kmax-kmin), dtype=np.complex128)

	template_conj = np.conjugate(t_templates)
	#template_norm = np_sigmasq(t_templates, psds[ifos[0]], N, kmin, kmax, delta_f)

	#preds = {"H1": [], "L1": []}

	for j in range(noise_seg_start, n_noise_segments):
		n_windows = (slice_duration*sample_rate - window_size)//stride +1
		nonwindowed_SNR = np.empty((len(ifos), n_templates, slice_duration*sample_rate), dtype=np.float32)
		windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)

		start = time.time()
		print("noise segment", j)
		noise = next(noise_gen)
		noise_time += time.time() - start

		if injfile is not None and injfile != "noninj":
			for k in range(n_injs):
				if startgps[k] > valid_times[j] and gps[k] + 1 < valid_times[j] + end_cutoff:
					print("inserting injection {}".format(k))
					#print(k) 
					#insert into the loaded noise

					hp, hc = get_td_waveform(mass1 = mass1[k], mass2 = mass2[k], 
								spin1x = spin1x[k], spin1y = spin1y[k],
								spin2x = spin2x[k], spin2y = spin2y[k],
								spin1z = spin1z[k], spin2z = spin2z[k],
								inclination = inclination[k], distance = distance[k],
											delta_t = delta_t, f_lower = f_lower, approximant = td_approximant)

					for ifo in ifos:
						f_plus, f_cross = all_detectors[ifo].antenna_pattern(
							right_ascension=right_ascension[k], declination=declination[k],
							polarization=polarization[k],
							t_gps=gps_dict[ifo][k])
						
						detector_signal = f_plus * hp + f_cross * hc

						end_idx = int((gps_dict[ifo][k]-valid_times[j]) * 2048)
						#print(end_idx, end_idx - len(detector_signal))
						#print("inj stats:", mass1[k], mass2[k], spin1z[k], spin2z[k], inclination[k], distance[k])

						#TODO: remove multiplier on detector signal once finished testing !
						noise[ifos.index(ifo),end_idx - len(detector_signal):end_idx] += detector_signal #*10
		
		for ifo in range(len(ifos)):
			
			start = time.time()
			template_norm = np_sigmasq(t_templates, psds[ifos[ifo]], delta_f)

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
			chop_time = int((slice_duration - (valid_times[j] - valid_times[j-1])) * sample_rate)
			print("chopping", chop_time/sample_rate, "seconds")

			chop_index = int(((valid_times[j] - valid_times[j-1]) * sample_rate/stride))
			print("only windows {} onwards of sample {} are needed".format(chop_index, j))

			#nonwindowed_SNR = nonwindowed_SNR[:, :, chop_time:]
			#n_windows = (nonwindowed_SNR.shape[-1] - window_size)//stride +1
			print("n_windows:", n_windows)
			#windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)
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

		
		#print("example shape:", windowed_SNR[0, 0, 0:batch_size, :].shape)
		total_batches = 0

		#reshape into (2, flattened_windowed_SNR, 2048)
		windowed_SNR = windowed_SNR.reshape(2, -1, 2048)
		newshape = windowed_SNR.shape

		#print("post reshaping:", windowed_SNR[:,0,0])
		reshape_time += time.time() - start

		tritonbatches = int(np.ceil(windowed_SNR.shape[1]/batch_size))
		total_batches_sent += n_workers* tritonbatches #to account for n workers


		#workers wait before sending their first batch so that the server processes them in the correct order

		start = time.time()

		if n_gpus == 1:
			successes = triton_client.get_inference_statistics(model).model_stats[0].inference_stats.success.count
		else:
			successes = triton_client.get_inference_statistics(modelh).model_stats[0].inference_stats.success.count
		
		#if worker_id == 0:
		#	reqbatches = int(total_batches_sent - 2.3 * tritonbatches)
		#if worker_id == 1:
		#	reqbatches = int(total_batches_sent - 1.3 * tritonbatches)

		#the 0.3 is to allow the next worker to start sending batches slightly before the previous worker's are finished.
		reqbatches = int(total_batches_sent - (n_workers - worker_id + 0.3) * tritonbatches) 

		#testing new method that doesn't need required batches
		#reqbatches = 0

		#if worker_id == 2:
		#	reqbatches = int(total_batches_sent - 0.3 * tritonbatches)
		if old_job_id == n_jobs/n_workers - 1 and i == n_batches - 1 and worker_id == 0:
			#if worker_id == 0:
			#if there are an odd number of templates, worker 1 processes 1 less. so worker 0 needs to reduce the number of batches it waits for.
			overflow = 4*(n_templates/(n_templates - total_lastjob%2) -1)  
			reqbatches -= int(np.ceil(tritonbatches * overflow))
			print("removing {} required batches from worker 0".format(int(tritonbatches * overflow)))
			#if i >= int(np.ceil(last_job_templates/templates_per_batch)):
			#	#in this case, worker 0 will have to send more batches than worker 1 and so we can send immediately
			#	reqbatches = 0
			#	print("this worker has free reign to send as many batches as it wants now")
			#elif i == int(np.ceil(last_job_templates/templates_per_batch)) - 1:
			#	#in this case, worker 1 has fewer templates and so we need to reduce worker 0's sending requirement
			#	reqbatches -= tritonbatches * 0.5
			#print("sending in whatever order, as one of these jobs may have fewer templates")

		timeout = 0
		while successes < reqbatches:
			print("worker waiting, only {} successes. we need {}".format(successes, reqbatches))
			#print("queue size:",triton_client.get_inference_statistics(model).model_stats[0].inference_stats.nv_inference_pending_request_count.count)
			sys.stdout.flush()
			time.sleep(1)
			if n_gpus == 1:
				successes = triton_client.get_inference_statistics(model).model_stats[0].inference_stats.success.count
			else:
				successes = triton_client.get_inference_statistics(modelh).model_stats[0].inference_stats.success.count
			timeout += 1

			if timeout > 60:
				print("worker has been waiting too long, sending batch anyway")
				reqbatches = 0 
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
				triton_client.async_infer(model_name=model, inputs=[inputh, inputl], outputs=[outputh, outputl],
								request_id=request_id, callback=partial(onnx_callback,callback_q)) 
								#priority = np.constant_uint64(((i*n_noise_segments + j) * n_workers + worker_id)))
				total_batches += 1

			else:
				triton_client.async_infer(model_name=modelh, inputs=[inputh], outputs=[outputh],
								request_id=request_id_h, callback=partial(onnx_callback,callback_q))
								#priority = np.constant_uint64(((i*n_noise_segments + j) * n_workers + worker_id)))
				
				triton_client2.async_infer(model_name=modell, inputs=[inputl], outputs=[outputl],
								request_id=request_id_l, callback=partial(onnx_callback,callback_q))
								#priority = np.constant_uint64(((i*n_noise_segments + j) * n_workers + worker_id)))
				total_batches += 2
		
		print("sending time:", time.time()- predstart)
		start = time.time()

		del windowed_SNR, y

		print("total batches sent:", total_batches)
		all_responses = []


		count = 0
		while True:
			count += 1
			if count <= total_batches:
				#print(count)
				response = callback_q.get()
				all_responses.append(response)
			else:
				break
		print(f"infer time: {time.time() - start}")

		pred_time += time.time() - predstart

		det_output_shape = all_responses[0][0].shape[-1]
		if i == 0 and j == 0:
			print("det output shape:", det_output_shape)
			#print("here's a response:", all_responses[0][0].shape)
			#print(all_responses[0])

		start = time.time()
		#newshape
		#should have shape (n_templates, n_windows, det_output_shape)
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
		
		#if i == 0 and j == 0 and job_id == 0:
		#	np.save(os.path.join("/fred/oz016/alistair/infernus","SNR_batch_{}_segment_{}.npy".format(i, j)), nonwindowed_SNR)
		#	np.save(os.path.join("/fred/oz016/alistair/infernus","preds_batch_{}_segment_{}.npy".format(i, j)), predbuf)

		#save to my folder
		while True:
			try:
				#assert that nonwindowed_SNR has the correct shape
				#assert nonwindowed_SNR.shape == (len(ifos), n_templates, slice_duration*sample_rate)
				np.save(os.path.join(myfolder, "SNR_batch_{}_segment_{}_chop_{}.npy".format(i, j, chop_time//2048)), nonwindowed_SNR)
				break

			except AssertionError as e:
				print("preds array has wrong shape, waiting 10 seconds")
				time.sleep(10)

			except Exception as e:
				print("Run out of disk space, waiting 10 seconds")
				time.sleep(10)	

		while True:
			try:
				#assert that predbuf has the correct shape
				#assert predbuf.shape == (n_templates, n_windows, det_output_shape)
				np.save(os.path.join(myfolder, "preds_batch_{}_segment_{}_chop_{}.npy".format(i, j, chop_time//2048)), predbuf)
				break

			except AssertionError as e:
				print("preds array has wrong shape, waiting 10 seconds")
				time.sleep(10)

			except Exception as e:
				print("Run out of disk space, waiting 10 seconds")
				time.sleep(10)


		
		
		timeslide_time += time.time() - start
		

		del all_responses, nonwindowed_SNR, predbuf, hbuf, lbuf#, noise

		gc.collect()

		sys.stdout.flush()

	#delete batch variables to save memory
	del noise, template_conj

	gc.collect()
	
	#print("getting memory snapshot")
	#snapshot = tracemalloc.take_snapshot()
	#display_top(snapshot, limit = 20)



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


#close the connection to the server(s)
#triton_client.close()
#if n_gpus == 2:
#	triton_client2.close()