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

parser.add_argument('--maxnoisesegs', type=int, default=None)

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
tf_model = args['tf_model']
savedir = args['save_dir']
columns = args['columns']


#TODO: add ifos to args.json
try:
	ifos = args['ifos']

except:
	ifos = ["H1", "L1"]
	print("Oi! fix your args file! add a list of ifos.")



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

inference_rates = np.array([int(x.split("_")[0][:-2]) for x in columns])
window_sizes = [float(x.split("_")[1][:-1]) for x in columns]
window_sizes = np.array(window_sizes * inference_rates, dtype= np.int32)

print("inference rates:", inference_rates)
print("window sizes:", window_sizes)

#REGULAR SNR SERIES STUFF

print(noise_dir)
#noise_dir = "/fred/oz016/alistair/GWSamplegen/noise/O3_third_week_1024"
#duration = 1024
#sample_rate = 2048
delta_t = 1/sample_rate
#f_lower = 30
f_final = sample_rate//2
delta_f = 1/duration
#fd_approximant = "TaylorF2"
#td_approximant = "SpinTaylorT4"

#import multiprocessing as mp
#n_cpus = 2

#WINDOW_SIZE = 2048
#STEP = 128

valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, 900, blacklisting = False)
templates, _, _= load_pycbc_templates(template_bank_name, template_bank_dir)

N = int(duration/delta_t)
kmin, kmax = np_get_cutoff_indices(f_lower, None, delta_f, N)


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

if injfile is not None and injfile != "noninj" and injfile != "real":
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

elif injfile == "noninj":
	print("performing noninj run")

elif injfile == "real":
	print("Performing real event run: no GPS blacklisting")

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
) -> tuple[grpcclient.InferenceServerClient, grpcclient.InferenceServerClient,
	  grpcclient.InferInput, grpcclient.InferInput, grpcclient.InferRequestedOutput,
	  grpcclient.InferRequestedOutput, grpcclient.InferRequestedOutput]:
	
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
if n_jobs <= 100 and injfile != 'real':
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

window_size = 2048
stride = 128

start_cutoff = max_waveform_length
end_cutoff = duration - 24 #set so total length is a nice number
slice_duration = (end_cutoff - start_cutoff)

print("start cutoff: {}, end cutoff: {}, slice duration: {}".format(start_cutoff, end_cutoff, slice_duration))

n_windows = (slice_duration*sample_rate - window_size)//stride +1

print("n_windows:", n_windows)


#TODO: remove requirement that it's in my folder



json_dict = {
	"template_start": template_start,
	"inference_rate": sample_rate//stride,
	"n_noise_segments": n_noise_segments,
	"n_batches": n_batches,
	"injection_file": 1 if injfile else 0,
	"valid_times": valid_times.tolist()
}

gps_blacklist = load_gps_blacklist(f_lower, event_file = "/fred/oz016/alistair/GWSamplegen/noise/segments/event_gpstimes.json").tolist()


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

#moving injection run cleanup code here
from triggering.zerolags import get_zerolags
from model_utils import new_split_models, split_model_stack

import sys
#sys.path.append("/home/amcleod/detnet/utils")
#from train_utils import LogAUC
double_det, combiner, full_model = split_model_stack(tf_model)

#import train_utils from the new system path, so that the linter is happy



def idx_to_gps(idx, start_time):
    return np.floor(idx/2048 + start_time)

def get_windows(start_end_indexes, peak_pos, pad=True, stride = 128):


    if pad:
        ret = np.arange(int(max(peak_pos//stride - sample_rate//stride, -1)), 
                        int(min(peak_pos//stride + 2, len(start_end_indexes)+1)))
    else:
        ret = np.arange(int(max(peak_pos//stride - sample_rate//stride +1, 0)), 
                        int(min(peak_pos//stride + 1, len(start_end_indexes))))

    return ret

inference_rate = sample_rate//stride
f16 = np.ones(inference_rate)/inference_rate
f12 = np.ones(12)/12
f8 = np.ones(8)/8
f4 = np.ones(4)/4
f2 = np.ones(2)/2
light_travel_time = sample_rate//100

ma_kernels = {16: f16, 12: f12, 8: f8, 4: f4, 2: f2}

statusfolder = os.path.join(os.environ["JOBFS"], "job_" +str(old_job_id), "completed") #this folder is used to shut down the triton server

global windowed_SNR
global template_conj

#flush prints
import sys
sys.stdout.flush()


def wait_for_server(n_gpus, triton_client, triton_client2):
	while True:
		try:
			if (n_gpus == 1 and triton_client.is_server_ready()) or \
				(n_gpus == 2 and triton_client.is_server_ready() and triton_client2.is_server_ready()) :

				break
			else:
				print("waiting for server to be live")
				time.sleep(1)
		except Exception as e:
			
			print("waiting for server to be live")
			time.sleep(1)


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

	for j in range(n_noise_segments):
		n_windows = (slice_duration*sample_rate - window_size)//stride +1
		nonwindowed_SNR = np.empty((len(ifos), n_templates, slice_duration*sample_rate), dtype=np.float32)
		windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)

		start = time.time()
		print("noise segment", j)
		noise = next(noise_gen)
		noise_time += time.time() - start

		if injfile is not None and injfile != "noninj" and injfile != "real":
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

		del y

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
		
			#for ifo in range(len(ifos)):
			#	windowed_SNR[ifo] = np.array(make_windows_2d(nonwindowed_SNR[ifo, :, :], window_size, stride))
			window_time += time.time() - start
				
		else:
			chop_index = 0
			chop_time = 0

			start = time.time()
			#for ifo in range(len(ifos)):
			#	windowed_SNR[ifo] = np.array(make_windows_2d(nonwindowed_SNR[ifo, :, :], window_size, stride))
			window_time += time.time() - start

		
		#Now we do the zerolag stuff
		
		#to make things easier, let's add some zeroes to the nonwindowed SNR if there's only one ifo

		if len(ifos) == 1:
			if ifos[0] == "L1":
				nonwindowed_SNR = np.concatenate((np.zeros((1, nonwindowed_SNR.shape[1], nonwindowed_SNR.shape[2])), nonwindowed_SNR), 
									 	axis = 0, dtype=np.float32)
			else:
				nonwindowed_SNR = np.concatenate((nonwindowed_SNR, np.zeros((1, nonwindowed_SNR.shape[1], nonwindowed_SNR.shape[2]))), 
										axis = 0, dtype=np.float32)

		zerolags = get_zerolags(
			data = nonwindowed_SNR,
			snr_thresh = 4,
			offset = 20,
			buffer_length = 2048,
			overlap = int(0.2*2048),
			num_trigs = 1,
			chop_time = chop_time,
		)

		zerolags = np.array(zerolags)
		zerolags = np.concatenate((zerolags, np.zeros((zerolags.shape[0], zerolags.shape[1], 8)) - 1000), axis = -1)

		print("there are {} zerolags".format(len(zerolags)))

		deleted_zerolags = []
		if injfile != "real":
			for gps_time in gps_blacklist:
				if gps_time > valid_times[j] and gps_time < valid_times[j] + slice_duration:
					delete_time = int(gps_time - valid_times[j] - chop_time//2048)
					if delete_time > 0:
						#to handle chopped segments, i.e. if the event is before the chop time, we still need to zero it for
						#timeslide purposes.
						zerolags[delete_time] = -1
					print("deleted zerolag at time", delete_time)
					print("Actual GPS time of deleted event:", gps_time)
					deleted_zerolags.append(gps_time)

		if len(deleted_zerolags) > 0:
			print("deleted zerolags:", deleted_zerolags)

		peak_pos_array = []
		#windows, detectors, templates
		for k in range(0, nonwindowed_SNR.shape[2]-2048+stride, stride):

			peak_pos_array.append(np.argmax(nonwindowed_SNR[:,: , k:k+2048], axis = 2))
		peak_pos_array = np.array(peak_pos_array)
		if i == 0 and j == 0:
			print("peak pos array shape:", peak_pos_array.shape)

		windowed_sample_end_indexes = list(range(sample_rate-1, nonwindowed_SNR.shape[-1], sample_rate//inference_rate))
		windowed_sample_start_indexes = list(np.copy(windowed_sample_end_indexes) - (sample_rate - 1))
		start_end_indexes = list(zip(windowed_sample_start_indexes, windowed_sample_end_indexes))
		
		SNR_send = []
		delta_t_send = []

		pred_arrays = {}
		for rate in inference_rates:
			pred_arrays[rate] = {"h": [], "l": [], "delta_t_array": []}

		rate_sum = sum(rate+2 for rate in pred_arrays)

		for key_k, k in enumerate(zerolags):
				
			# Check this zerolag is valid
			if k[0][0] == -1:
				#print(f"Zerolag {key_i} is invalid")
				continue
			
			
			#process the zerolags

			#TODO: iterate through the different unique sample rates, like in background_timeslides.py.
			#as a quick solution, just append the needed samples to SNR_send and delta_t_send.
			#then when we receive them 

			primary_det = np.argmax(k[0][:2])
			primary_det_pos = k[0][3+primary_det]
			#primary_det_samples = get_windows(start_end_indexes, primary_det_pos)

			primary_det_samples = []
			for rate in pred_arrays:
				pred_arrays[rate]["primary_det_samples"] = get_windows(start_end_indexes, primary_det_pos, stride = sample_rate//rate) * int(16/rate)
			
			primary_det_samples = np.concatenate([pred_arrays[rate]["primary_det_samples"] for rate in pred_arrays])

			if len(primary_det_samples) < rate_sum or min(primary_det_samples) < 0 or \
						max(primary_det_samples) >= len(start_end_indexes):
				print(f"Not enough space either side to get full moving average predictions for primary detector in zerolag {key_k}:")
				print(k)
				zerolags[key_k][0][0] = -1
				continue


			#this is where we normally load primary preds, instead we need to prepare to send the corresponding SNR to triton

			#convert primary_det_samples windows to indicies of nonwindowed_SNR

			for p in primary_det_samples:
				SNR_send.append(nonwindowed_SNR[:, int(k[0][5]), p*stride:p*stride+window_size])

				delta_t_send.append(-np.diff(peak_pos_array[p, :, int(k[0][5])])[0] / light_travel_time)


		if i == 0 and j == 0:
			if n_gpus == 1:
				wait_for_server(n_gpus, triton_client, triton_client)
			else:
				wait_for_server(n_gpus, triton_client, triton_client2)

		start = time.time()

		
		#print("example shape:", windowed_SNR[0, 0, 0:batch_size, :].shape)
		total_batches = 0

		#ready to send the SNR to triton

		#each element in SNR_send has shape (2, window_size)
		#so SNR_send as a whole will have shape (zerolags,2,window_size)

		#if, instead we concatenate it will have shape (2, zerolags, window_size)
		
		#forcing float 32 just in case
		SNR_send = np.array(SNR_send, dtype=np.float32)
		SNR_send = np.swapaxes(SNR_send, 0, 1)

		tritonbatches = int(np.ceil(SNR_send.shape[1]/batch_size))

		predstart = time.time()

		print("SNR send shape:", SNR_send.shape)

		bufsize = 0
		for k in range(tritonbatches):
			if k == tritonbatches - 1 and SNR_send.shape[1] % batch_size != 0:
				#we may need to pad with zeroes to get to batch_size
				bufsize = batch_size - SNR_send.shape[1] % batch_size
				print("padding with", bufsize, "windows")
				hbuf = np.pad(SNR_send[0, k*batch_size:], ((0, bufsize), (0,0)), 'constant', constant_values = 0)
				lbuf = np.pad(SNR_send[1, k*batch_size:], ((0, bufsize), (0,0)), 'constant', constant_values = 0)
			
			else:
				hbuf = SNR_send[0, k*batch_size: (k+1)*batch_size]
				lbuf = SNR_send[1, k*batch_size: (k+1)*batch_size]
			
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

		#del windowed_SNR, y

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
		#NEW SHAPE! since we're only sending the top, it should have an arbitrary shape
		#something like n_zerolags, det_output_shape

		if n_gpus == 1:
			predbuf = np.empty((SNR_send.shape[1], det_output_shape), dtype=np.float32)
		else:
			predbuf = np.empty((SNR_send.shape[1], det_output_shape*2), dtype=np.float32)

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

					
		#next, feed the predictions into the combiner model
		det_output_shape = predbuf.shape[1]//2
		delta_t_send = np.array(delta_t_send)

		if len(ifos) == 1:
			#fill delta_t array with ones.
			#TODO: investigate (briefly) the effect of 
			delta_t_send.fill(1.0)

			#need to wipe the predbuf as well for the missing ifo

			#TODO: these fill values may change depending on requirements.
			#if you use a multiply combine layer, use a fill value of 1.0, else a value of 0.0 should work.

			if ifos[0] == "H1":
				predbuf[:, det_output_shape:].fill(0.0)
			elif ifos[0] == "L1":
				predbuf[:, :det_output_shape].fill(0.0)


		combined_preds = combiner.predict([predbuf[:, :det_output_shape], predbuf[:, det_output_shape:], delta_t_send],
																	verbose = 2, batch_size = 1024)
		
		#TODO: need to split up the predictions by a different amount if we have multiple inference rates.
		#18 needs to be sum(inference_rates) + 2*len(inference_rates)
		#we then split up these pred timeseries into [0:inference_rate[0]], [inference_rate[0]:inference_rate[0]+inference_rate[1]], etc.
		#MAKE SURE EVERYTHING IS IN A DICTIONARY!

		combined_preds = combined_preds.reshape(-1, 1, rate_sum)
		print("combined preds shape: {}".format(combined_preds.shape))

		rate_idx = 0
		for rate in pred_arrays:
			#print(primary_det_samples[rate_idx:rate_idx+rate+2])
			pred_arrays[rate]['preds'] = combined_preds[:, :, rate_idx:rate_idx+rate+2]
			print(pred_arrays[rate]['preds'][0])
			rate_idx += rate + 2
			

		true_idx = 0
		for key_k, k in enumerate(zerolags):
			if k[0, 0] == -1.0:
				#print(f"Zerolag {key_i} is invalid")
				continue
			
			for idx, rate in enumerate(inference_rates):
				ma_prediction = np.max(np.convolve(pred_arrays[rate]["preds"][true_idx][0], ma_kernels[window_sizes[idx]], mode = 'valid'))
				zerolags[key_k][0][6+idx] = ma_prediction
			#ma_prediction_16hz = np.max(np.convolve(combined_preds[true_idx][0], f16, mode = 'valid'))
			#ma_prediction_16hz_12 = np.max(np.convolve(combined_preds[true_idx][0], f12, mode = 'valid'))
			#ma_prediction_16hz_8 = np.max(np.convolve(combined_preds[true_idx][0], f8, mode = 'valid'))
			#ma_prediction_16hz_4 = np.max(np.convolve(combined_preds[true_idx][0], f4, mode = 'valid'))

			#pred_stuff = np.array([ma_prediction_16hz, ma_prediction_16hz_12, ma_prediction_16hz_8, ma_prediction_16hz_4,
			#								np.max(combined_preds[true_idx][0][1:-1]), np.max(combined_preds[true_idx][0][1:-1]),
			#								np.max(combined_preds[true_idx][0][1:-1]), np.max(combined_preds[true_idx][0][1:-1])])


			#zerolags[key_k][0][-8:] = pred_stuff


			true_idx += 1

		zerolags[:,0,5] += template_start_idx
		print("zl_shape:", zerolags.shape)
		#save the zerolags to disk
		print("saving to zerolag file: zerolags_{}-{}_batch_{}_segment_{}.npy".\
				format(template_start_idx, template_start_idx + n_templates, i, j))
		
		np.save(os.path.join(savedir, "zerolags_{}-{}_batch_{}_segment_{}.npy".\
				format(template_start_idx, template_start_idx + n_templates, i, j)), zerolags)


		del all_responses, nonwindowed_SNR, predbuf, hbuf, lbuf, zerolags, combined_preds, pred_arrays, SNR_send, delta_t_send

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

#write a file to the status folder to indicate that this job is done
with open(os.path.join(statusfolder, "worker_{}.txt".format(worker_id)), "w") as f:
	f.write("done")
time.sleep(1)
#print the contents of the status folder

print("status folder contents:")
print(os.listdir(statusfolder))

#close the connection to the server(s)
#triton_client.close()
#if n_gpus == 2:
#	triton_client2.close()