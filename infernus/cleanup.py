import os
import numpy as np
import argparse
import time
from triggering.zerolags import get_zerolags
import sys

#TODO: change to a better way of splitting models
from model_utils import split_models
ifo_dict = split_models()

parser = argparse.ArgumentParser()
parser.add_argument('--jobindex', type=int)
parser.add_argument('--workerid', type=int, default=0)
parser.add_argument('--totalworkers', type=int, default=1)
parser.add_argument('--totaljobs', type=int, default=1)

args = parser.parse_args()

job_id = args.jobindex #job id in job array
worker_id = args.workerid #worker number of a server
n_workers = args.totalworkers
n_jobs = args.totaljobs



print("JOBFS", os.environ["JOBFS"] )

myfolder = os.path.join(os.environ["JOBFS"], "job_" +str(job_id), "worker_"+str(worker_id))

print("my folder is", myfolder)

print("starting job {} of {}".format(job_id, n_jobs))
print("I am cleanup job {} of {} for this server".format(worker_id, n_workers))
job_id = worker_id + job_id*n_workers
print("my unique index is {}".format(job_id))
n_jobs = n_jobs * n_workers
print("there are {} jobs in total".format(n_jobs))


#change dir into worker's folder
os.chdir(myfolder)

while "args.json" not in os.listdir(myfolder):
	print("waiting for args.json")
	time.sleep(5)

import json
args = json.load(open("args.json", "r"))

print("args are", args)

template_start = args["template_start"]


batch = 0 #template batch count
segment = 0 #noise segment count

while True:
	files = os.listdir(myfolder)
	if 'SNR_batch_{}_segment_{}.npy'.format(batch, segment) in files \
		  and 'preds_batch_{}_segment_{}.npy'.format(batch, segment) in files:
		print("Found files", files)
	
	else:
		print("no files found")
		sys.stdout.flush()
		time.sleep(10)
		continue

	SNR = np.load('SNR_batch_{}_segment_{}.npy'.format(batch, segment))
	preds = np.load('preds_batch_{}_segment_{}.npy'.format(batch, segment))
		
	n_templates = SNR.shape[1]

	#for initial testing just do the timeslides WITHOUT zerolags

	#split the preds along the last axis in half for H and L
	#preds has the shape (n_templates, n_timeslides, ifo_output*2)
	ifo_pred_len = preds.shape[2]//2

	#preds = preds.reshape(-1, ifo_pred_len*2)

	H_preds = preds[:,:,:ifo_pred_len]
	L_preds = preds[:,:,ifo_pred_len:]

	n_windows = H_preds.shape[1]

	combopreds = []

	save_arr = np.zeros((2, n_windows))

	
	start = time.time()
	L_roll = L_preds.copy()
	#roll and concatenate L_preds 10 times
	for i in range(1,100):
		L_roll = np.concatenate((L_roll, np.roll(L_preds, i * n_windows//100, axis=1)), axis=0)
	
	H_preds = np.tile(H_preds, (100,1,1))

	print("finished rolling, took {} seconds".format(time.time() - start))

	start = time.time()

	for j in range(n_templates):
		#L_roll = np.roll(L_preds, i * n_windows//10, axis=1)
		combopreds = ifo_dict['combiner'].predict([H_preds[j], L_roll[j]], batch_size = 4096, verbose = 2)
		#for each window, only save the maximum prediction between templates
		save_arr[0] = np.maximum(save_arr[0], combopreds[0])
		#if we overwrite the maximum, we need to change the template index
		save_arr[1] = np.where(save_arr[0] == combopreds[0], j, save_arr[1])


	print("finished timeslides, took {} seconds".format(time.time() - start))

	#combopreds = np.array(combopreds)
	np.save("/fred/oz016/alistair/infernus/timeslides/combopreds_templates_{}-{}_batch_{}_segment_{}.npy".\
		 format(template_start, template_start + n_templates, batch, segment), save_arr)
	
	os.remove('SNR_batch_{}_segment_{}.npy'.format(batch, segment))
	os.remove('preds_batch_{}_segment_{}.npy'.format(batch, segment))
	
	template_start += n_templates

	if batch == args['n_batches'] - 1 and segment == args['n_noise_segments'] - 1:
		print("main job should have finished")
		break

	elif segment == args['n_noise_segments'] - 1:
		batch += 1
		segment = 0

	else:
		segment += 1

	"""
	if len(files) > 0:

		print("Found files", files)
		i = files[0].split("_")[2]
		j = files[0].split("_")[4]
		print("i is", i)
		print("j is", j)
		os.remove(os.path.join(myfolder, files[0]))
		if i == 9 and j == 9:
			print("main job should have finished")
			break
	"""

print("finished doing cleanup")
# for each file, load it, and get the best zerolags

# save the best zerolags to a file

