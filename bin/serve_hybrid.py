#Hybrid CPU/GPU serving script for Infernus. If passed a GPU node and port, it will immediately attach to that server.
#If not, it will run only on CPU unless a GPU job makes itself available by writing a port file to the save directory.

import time
import numpy as np
import tritonclient.grpc as grpcclient
import os
import argparse
import json
import onnxruntime as ort
import sys

start = time.time()

from typing import Tuple

from typing import Optional
from tritonclient.grpc._infer_result import InferResult
from tritonclient.utils import InferenceServerException

from functools import partial
from queue import Queue
import concurrent.futures
callback_q = Queue()
import gc
import sys
#sys.path.append("/fred/oz016/alistair/infernus/dev")
#from infernus.jobmem import jobmem
#jmem = jobmem()   # do this once


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
		output = list(result._result.outputs)

		if len(output) == 2 and (output[0].name != 'h_out' or output[1].name != 'l_out'):
			raise ValueError("Output names are not as expected. output labels are: ", output[0].name, " and ", output[1].name)

		for i in range(len(output)):
			if i == 0:
				np_outputs = result.as_numpy(output[i].name)
			else:
				np_outputs = np.concatenate((np_outputs, result.as_numpy(output[i].name)), axis=1)
			
		#np_outputs = result.as_numpy(result._result.outputs)


		response = (np_outputs, request_id)

		if response is not None:
			queue.put(response)

	except Exception as ex:
		print("Exception in callback")
		print("An exception of type occurred. Arguments:")
		#message = template.format(type(ex).__name__, ex.args)
		print(type(ex))
		print(ex)


parser = argparse.ArgumentParser()
parser.add_argument('--jobindex', type=int,default=0)
parser.add_argument('--workerid', type=int, default=0)
parser.add_argument('--totalworkers', type=int, default=1)
parser.add_argument('--totaljobs', type=int, default=1)
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--argsfile', type=str, default=None)
parser.add_argument('--injindex', type = int, default = -1)
parser.add_argument('--gpunode', type=str, default=None)
parser.add_argument('--port', type = int, default=None) #note this is the GRPC port, not the HTTP port.

args = parser.parse_args()

job_id = args.jobindex #job id in job array
#worker_id = args.workerid
n_workers = args.totalworkers
n_jobs = 1

gpu_node = args.gpunode
grpc_port = args.port
print("Node:", gpu_node)
print("Port:", grpc_port)
#gpu_node = 'gina304'
#grpc_port = args.port + 1 #GRPC port is always 1 more than HTTP port
#gpu_node = args.node
#grpc_port = args.port + 1 #GRPC port is always 1 more than HTTP port
if gpu_node is not None and grpc_port is not None:
	print("Running on the same node as the GPUS!")
	streamline = True
else:
	streamline = False

argsfile = args.argsfile
print(argsfile)

print("Job Id: ", job_id)

json_args = json.load(open(argsfile, "r"))
noise_dir = json_args["noise_dir"]
maxnoisesegs = json_args["max_noise_segments"]
#template_bank_dir = json_args["template_bank_dir"]
#template_bank_name = json_args["template_bank_name"]
duration = json_args["duration"]
sample_rate = json_args["sample_rate"]
f_lower = json_args["f_lower"]
#fd_approximant = json_args["fd_approximant"]
#td_approximant = json_args["td_approximant"]
injfile = json_args["injfile"]
save_dir = json_args["save_dir"]
num_time_slides = json_args["n_timeslides"]
num_triggers = json_args["num_triggers"]
batch_size = json_args["batch_size"]
if "bin" in json_args:
	bin = json_args["bin"]
	new_style = True
	print("Using new style of model directory")
else:
	new_style = False
if "injection_SNR_dir" in json_args and injfile not in [None, "noninj", "real"]:
	injection_SNR_dir = json_args["injection_SNR_dir"]
	print("Loading injection SNRs from ", injection_SNR_dir)
else:
	injection_SNR_dir = None

if args.injindex >= 0:
	inj_index = args.injindex
	print("Using injection file index", inj_index)
	print("Need to modify the save directory to include the index")
	save_dir = os.path.join(save_dir, "inj_" + str(inj_index))
	print("New save directory is ", save_dir)

cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
print("running on {} cpus".format(cpus))

try:
	inference_rate = int(json_args["inference_rate"])
	print("inference rate is ", inference_rate)
except:
	inference_rate = 16
	print("inference rate not specified, defaulting to 16 Hz")

try: 
	num_models = int(json_args["final_model_candidates"])
	print("number of models to test is ", num_models)
except:
	num_models = 1
	print("number of models not specified, defaulting to 1")

if injfile == "None":
	injfile = None
else:
	#we set this to 1 so we can keep the same structure
	num_time_slides = 1



print(noise_dir)
delta_t = 1/sample_rate
f_final = sample_rate//2
delta_f = 1/duration

stride = sample_rate//inference_rate


#batch_size = 1024
def initialise_server(
	gpu_node: str, 
	grpc_port: str, 
	model: str = 'model_hl', 
	modelh: str = 'model_h', 
	modell: str = 'model_l'
) -> Tuple[grpcclient.InferenceServerClient, grpcclient.InferenceServerClient,
	  grpcclient.InferInput, grpcclient.InferInput, grpcclient.InferRequestedOutput,
	  grpcclient.InferRequestedOutput, grpcclient.InferRequestedOutput]:
	
	"Return all the triton client and input/output objects needed for running the inference server."
	
	triton_client = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port))
	triton_client2 = grpcclient.InferenceServerClient(url=gpu_node + ":"+ str(grpc_port+3))

	inputh = grpcclient.InferInput("h", (batch_size, 2048, 1), datatype="FP32")
	inputl = grpcclient.InferInput("l", (batch_size, 2048, 1), datatype="FP32")
	input_dt = grpcclient.InferInput("delta_t", (batch_size, 1), datatype="FP32")
	output = grpcclient.InferRequestedOutput("concat")
	outputh = grpcclient.InferRequestedOutput("h_out")
	outputl = grpcclient.InferRequestedOutput("l_out")
	output_full = grpcclient.InferRequestedOutput("full_out")

	return triton_client, triton_client2, inputh, inputl, input_dt, output, outputh, outputl, output_full

if streamline:
	myfolder = save_dir
else:
	if injection_SNR_dir is not None:
		myfolder = injection_SNR_dir
		if args.injindex >= 0:
			myfolder = os.path.join(myfolder, "inj_{}".format(inj_index))
			print("using injection SNR directory:", myfolder)
	else:
		myfolder = os.environ["JOBFS"]


light_travel_time = sample_rate //100


if injfile == "real":
	print("We have to do some extra stuff to put these files in the right place")
	#open myfolder/real_event.txt	
	with open(myfolder + "/real_event.txt", "r") as f:
		path = f.read().strip()
		print("path to save to is ", path)
		save_dir = path


#now merge the timeslides
timeslides = np.load(myfolder + "/timeslides_{}_0.npy".format(job_id))
if streamline:
	SNR = np.load(myfolder + "/SNR_array_{}_0.npy".format(job_id))
	#we can also delete the SNR and timeslides files now
	os.remove(myfolder + "/SNR_array_{}_0.npy".format(job_id))
	os.remove(myfolder + "/timeslides_{}_0.npy".format(job_id))
	print("deleted SNR and timeslides files")
else:
	SNR = np.load(myfolder + "/SNR_array_{}_0.npy".format(job_id), mmap_mode='r')


#now that we've merged the timeslides, we can run the inference

required_rows = num_models + 8

if timeslides.shape[3] < required_rows:
	print("adding entries to accomodate model predictions")
	timeslides = np.concatenate((timeslides, np.zeros((timeslides.shape[0], timeslides.shape[1], timeslides.shape[2], required_rows - timeslides.shape[3]))), axis = 3)
else:
	print("timeslides already has enough entries for model predictions")



#where to add the response to the timeslides array
response_idx = 8

#print the SNR datatype
print("SNR dtype is ", SNR.dtype)

#the models are stored one directory down from the save directory
if os.path.exists(os.path.join(os.path.dirname(save_dir), "model_repositories", "repo_1")):
	modeldir = os.path.join(os.path.dirname(save_dir), "model_repositories", "repo_1")
else:
	print("new style of model dir")
	#get the name of the parent directory
	# pardir = save_dir.split("/")[-1]
	# modeldir = os.path.dirname(os.path.dirname(save_dir))
	# modeldir = os.path.join(modeldir, "models", pardir , "model_repositories", "repo_1")
	modeldir = os.path.join(json_args["jobdir"], "models", bin , "model_repositories", "repo_1")
	print("modeldir is ", modeldir)

sess = None
def run_session(args):
	return (sess.run(None, {input_h: args[0], input_l: args[1], input_dt: args[2]})[0], args[3])
	#return sess.run(None, {input_h: h_data, input_l: l_data, input_dt: dt_data})
#executor = concurrent.futures.ProcessPoolExecutor(max_workers=cpus)

#new code for GPU server selection

#grpc_port = None
index = 0
#gpu_node = None

import atexit

def exit_handler(grpc_port, gpu_node, index):
	if grpc_port is not None:
		with open(os.path.join(save_dir, gpu_node + "_" + str(index) + ".port" ),'w') as f:
			f.write(str(grpc_port))
			print("wrote port to file")
	else:
		print("no port to write to file")

def check_for_server():
	global grpc_port, index, gpu_node, node
	files = os.listdir(save_dir)
	for f in files:
		if f[-5:] == ".port":
			#we've found a free server! read the file's contents and delete it
			try:
				print("found a server!")
				grpc_port = int(open(save_dir + "/" + f, "r").read())
				node = f.split(".")[0]
				gpu_node = node.split("_")[0]
				index = int(node.split("_")[1])
				os.remove(save_dir + "/" + f)
				print("removed file", f, " with port ", grpc_port)
				return True
			except:
				print("failed to remove file, resetting port to None")
				grpc_port = None
	
	return False

if streamline:
	gpu_ready = True
else:
	#might as well check for a server now
	gpu_ready = check_for_server()

if gpu_ready:
	print("immediately found a server")
	for i in range(3):
		try:
			time.sleep(5)
			triton_client, triton_client2, inputh, inputl, input_dt, output, outputh, outputl, output_full = initialise_server(gpu_node, grpc_port)
			print("port, node and index are ", grpc_port, gpu_node, index)
			if not streamline:
				#need to be ready to save the port file
				atexit.register(exit_handler, grpc_port, gpu_node, index)
				print("registered exit handler")
			gpu_ready = True
			break
		except Exception as e:
			print("failed to initialise server")
			print("exception was ", e)
			gpu_ready = False
			#sleep
			time.sleep(5)

#CPU inference variables
flush_count = 50
future_list = []
inference_count = 0
current_cpus = cpus #int(cpus//2)
executor = None
if injfile == 'real':
	unaveraged_response_array = np.zeros((timeslides.shape[0], timeslides.shape[1], timeslides.shape[2], inference_rate))

for n in range(num_models):
	if gpu_ready:
		try:
			modelname = triton_client.get_model_repository_index().models[n].name
			print("model name is ", modelname)
			print("initialising server with model ", modelname)
			triton_client.load_model(modelname)
			print("loading success:",triton_client.is_model_ready(modelname))
		except:
			print("failed to get model name")
			gpu_ready = False
	
	else:
		#running on CPU while we wait for the GPU
		# if executor is not None:
		# 	executor.shutdown(wait=False, cancel_futures=True)
		# 	print("shut down old executor")
		executor = concurrent.futures.ProcessPoolExecutor(max_workers=current_cpus)
		model = os.path.join(modeldir, "model_full_" + str(n), "1", "model.onnx")
		sess_opt = ort.SessionOptions()
		sess_opt.intra_op_num_threads = 2
		sess = ort.InferenceSession(model, sess_opt, providers=['CPUExecutionProvider'])
		input_h = sess.get_inputs()[0].name
		input_l = sess.get_inputs()[1].name
		input_dt = sess.get_inputs()[2].name
		print("model is ", model)


	response_array = np.zeros((timeslides.shape[0], timeslides.shape[1], timeslides.shape[2], inference_rate))
	print("response array shape is ", response_array.shape)

	send_buffer_h = np.zeros((batch_size, 2048, 1)).astype(np.float32)
	send_buffer_l = np.zeros((batch_size, 2048, 1)).astype(np.float32)
	send_buffer_dt = np.zeros((batch_size, 1)).astype(np.float32)

	ts_start = 0
	second_start = 0
	window_start = 0
	trigger_start = 0

	ts_counter = 0
	second_counter = 0
	window_counter = 0
	trigger_counter = 0
	#print("my priority is ", job_id)
	#we send by window, then by timeslide, then by second
	done = False
	count = 0 #note count only increments when we use the GPU
	start = time.time()

	response_list = []
	while not done:
		request_id = str(window_counter) + "_" + str(trigger_counter) + "_" + str(ts_counter) + "_" + str(second_counter) #+ "_" + str(cutoff)
		for i in range(len(send_buffer_h)):
			temp_h = SNR[0,ts_counter,second_counter,trigger_counter,(window_counter+1) * stride: (window_counter+1) * stride+2048].copy()
			temp_l = SNR[1,ts_counter,second_counter,trigger_counter,(window_counter+1) * stride: (window_counter+1) * stride+2048].copy()
			temp_dt = (np.argmax(temp_h) - np.argmax(temp_l))/light_travel_time
			send_buffer_h[i] = temp_h.reshape(2048,1)
			send_buffer_l[i] = temp_l.reshape(2048,1)
			send_buffer_dt[i] = temp_dt
			window_counter = (window_counter + 1)%inference_rate
			if window_counter == 0:
				trigger_counter = (trigger_counter + 1)%num_triggers
				if trigger_counter == 0:
					ts_counter = (ts_counter + 1)%num_time_slides
					if ts_counter == 0:
						second_counter = (second_counter + 1)%timeslides.shape[1]
						if second_counter == 0:
							#need add a flag to indicate that we're done
							done = True
							break

		if gpu_ready:

			inputh.set_data_from_numpy(send_buffer_h)
			inputl.set_data_from_numpy(send_buffer_l)
			input_dt.set_data_from_numpy(send_buffer_dt)

			#success = triton_client.get_inference_statistics('model_full').model_stats[0].inference_stats.success.count
			#q = triton_client.get_inference_statistics('model_full').model_stats[0].inference_stats.queue.count

			triton_client.async_infer(model_name=modelname, inputs=[inputh, inputl, input_dt], outputs=[output_full],
										request_id=str(request_id), callback=partial(onnx_callback,callback_q))#, priority=job_id)
			count += 1
			#sleep for a small amount of time to avoid overloading the server
			#time.sleep(0.001)
			
			if callback_q.qsize() == 0 and count > 300:
				print("waiting for responses", flush=True)
				#this response is blocking
				response = callback_q.get()
				response_list.append(response)
				count -= 1
				time.sleep(0.5)


			if callback_q.qsize() > 0:
				response = callback_q.get()
				response_list.append(response)
				count -= 1

		else:
			future_list.append(executor.submit(run_session, (send_buffer_h.copy(), send_buffer_l.copy(), send_buffer_dt.copy(), request_id)))
			inference_count += 1
			if second_counter%10 == 0 and ts_counter == 0 and trigger_counter == 0 and window_counter == 0:
				print("CPU inference!")
				print(request_id)
			
		if len(future_list) >= flush_count:
			print("flushing requests")
			futures_done, not_done = concurrent.futures.wait(future_list, timeout=300)
			#print("new test")
			if len(not_done) > 0:
				print("some requests did not complete in 300s! restarting this model")
				#need to restart the model
				future_list = []
				#kill the executor and start again
				print("killing executor and starting again")
				#NOTE: this functionality is available in Python 3.14+, so this can be cleanly implemented then.
				print("list of child pids:")
				print(executor._processes.keys())
				for proc in executor._processes.values():
					try:
						if not proc.is_alive():
							continue
					except Exception as e:
						continue
					try:
						proc.kill()
					except Exception as e:
						print("failed to kill process, it may have already exited")
						continue
				print("killed all child processes")
				executor.shutdown(wait=False, cancel_futures=True)
				executor = concurrent.futures.ProcessPoolExecutor(max_workers=current_cpus)
				#also reset the counters
				ts_counter, second_counter, window_counter, trigger_counter = 0, 0, 0, 0
				sess = ort.InferenceSession(model, sess_opt, providers=['CPUExecutionProvider'])
				print("finished resetting the model")
				continue

			for future in future_list:
				response_list.append(future.result())
			future_list = []

			#now we can check if the GPU is ready
			gpu_ready = check_for_server()
			if gpu_ready:
				try:
					triton_client, triton_client2, inputh, inputl, input_dt, output, outputh, outputl, output_full = initialise_server(gpu_node, grpc_port)
					time.sleep(1)
					modelname = triton_client.get_model_repository_index().models[n].name
					print("model name is ", modelname)
					print("initialising server with model ", modelname)
					triton_client.load_model(modelname)
					print("loading success:",triton_client.is_model_ready(modelname))
					print("model name is ", modelname)
					print("found a server! switching to GPU")
					atexit.register(exit_handler, grpc_port, gpu_node, index)
					print("registered exit handler")
				except Exception as e:
					print("failed to initialise server")
					gpu_ready = False
					print("exception was ", e)
			else:
				print("no server found")
				#Doing some Slurm memory wizardry as we don't necessarily have enough memory
				#to spin up 1 model per CPU. This way we don't underutilise CPUs if we have enough memory.
				#current_mem = jmem.read() #read the job memory usage
				#print("current mem (bytes): ", current_mem)
				#mem_remaining = (current_mem['limit'] - current_mem['usage'])/(1024**3)
				# print(mem_remaining, "GB under limit")
				# if mem_remaining > 5 + 0.1*current_mem['limit']/(1024**3) and current_cpus < cpus:
				# 	current_cpus += 1
				# 	print("re-initialising process pool with more cpus")
				# 	executor.shutdown(wait=True)
				# 	executor = concurrent.futures.ProcessPoolExecutor(max_workers=current_cpus)


		sys.stdout.flush()


		#end of sending loop for one model
	if not gpu_ready:
		print("inferences per second: ", batch_size*inference_count/(time.time() - start))

	while count > 0:
		response = callback_q.get()
		response_list.append(response)
		count -= 1

	print("waiting for remaining requests to complete")
	concurrent.futures.wait(future_list)
	for future in future_list:
		response_list.append(future.result())
	future_list = []

	for response in response_list:
		response_data = response[0].ravel()
		response_header = response[1].split("_")
		window_start = int(response_header[0])
		trigger_start = int(response_header[1])
		ts_start = int(response_header[2])
		second_start = int(response_header[3])
		for r in range(len(response_data)):
			response_array[ts_start, second_start, trigger_start, window_start] = response_data[r]
			window_start = (window_start + 1)%inference_rate
			if window_start == 0:
				trigger_start = (trigger_start + 1)%num_triggers
				if trigger_start == 0:
					ts_start = (ts_start + 1)%num_time_slides
					if ts_start == 0:
						second_start = (second_start + 1)%timeslides.shape[1]
						if second_start == 0:
							break


	#success = triton_client.get_inference_statistics('model_full').model_stats[0].inference_stats.success.count
	#queue = triton_client.get_inference_statistics('model_full').model_stats[0].inference_stats.queue.count
	#print("uncompleted requests: ", queue - success)
	if gpu_ready:
		print("unloading model ", modelname)
		triton_client.unload_model(modelname)
		print("unloaded model ", modelname)

	response_array = np.mean(response_array, axis = -1)

	print("first response is ", response_array[0,0,0])
	print("response list length is ", len(response_list))
	triton_time = time.time() - start

	#only add the response if the timeslide is valid
	#for i in range(num_time_slides):
	#	overwrite_criterion = timeslides[i,:,0,2] > -1000

	#	timeslides[i,overwrite_criterion,0,8] = response_array[i, overwrite_criterion]

	#just overwrite everything, but don't use the predictions to tell if a timeslide is valid!

	timeslides[:,:,:,response_idx] = response_array[:,:,:]
	response_idx += 1
	print("response index is now ", response_idx)
	print("Inference time: ", triton_time)
	gc.collect()

#once we exit the loop, we're finished with the GPU

#TODO: check exit handler works correctly
#if gpu_ready:
#	with open(os.path.join(save_dir, gpu_node + "_" + str(index) + ".port" ),'w') as f:
#		f.write(str(grpc_port))
		
#save timeslides to file
#force datatype of timeslides to be float32
timeslides = timeslides.astype(np.float32)

#save the SNR array to file
#np.save(save_dir + "/SNR_array_{}.npy".format(job_id), SNR)


if injfile == "real":
	#Also save the SNR array and the predictions
	np.save(save_dir + "/SNR_array.npy", SNR)
	print("TODO: save the unaveraged prediction time series...")
	np.save(save_dir + "/timeslides.npy", timeslides)
else:
	np.save(save_dir + "/timeslides_{}.npy".format(job_id), timeslides)

print("finished serving job!")
if executor is not None:
	executor.shutdown(wait=False, cancel_futures=True)
	#kill all child processes

	print("shut down executor")
#os._exit(os.EX_OK)
print("Python exiting")
gc.collect()
# PID = os.getpid()
# os.kill(PID, 9)
sys.exit(0)