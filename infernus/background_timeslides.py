





import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import argparse
import time
import sys
import gc
import numpy as np
import datetime


def println(str):
    print(f"{datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))):%Y-%m-%d %H:%M:%S} - " + str)

from triggering.zerolags import get_zerolags

from model_utils import split_model_stack




def get_windows(start_end_indexes, peak_pos, pad=True, stride = 128, sample_rate = 2048):

    if pad:
        ret = np.arange(int(max(peak_pos//stride - sample_rate//stride, -1)), 
                        int(min(peak_pos//stride + 2, len(start_end_indexes)+1)))
    else:
        ret = np.arange(int(max(peak_pos//stride - sample_rate//stride +1, 0)), 
                        int(min(peak_pos//stride + 1, len(start_end_indexes))))

    return ret

def is_timeslide_valid(second,timeslide, chop_time = 0):
	return timeslide != (second + chop_time) and timeslide != (second + chop_time) - 1


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--jobindex', type=int)
	parser.add_argument('--workerid', type=int, default=0)
	parser.add_argument('--totalworkers', type=int, default=1)
	parser.add_argument('--totaljobs', type=int, default=1)
	parser.add_argument('--cleanupid', type=int, default=0)
	parser.add_argument('--totalcleanups', type=int, default=1)
	parser.add_argument('--argsfile', type=str, default='args.json')
	cmdargs = parser.parse_args()

	job_id = cmdargs.jobindex #job id in job array
	worker_id = cmdargs.workerid #worker number of a server
	n_workers = cmdargs.totalworkers
	n_jobs = cmdargs.totaljobs
	cleanup_id = cmdargs.cleanupid
	n_cleanups = cmdargs.totalcleanups

	args = json.load(open(cmdargs.argsfile, "r"))
	savedir = args["save_dir"]
	num_time_slides = args["n_timeslides"]
	seed = args['seed']
	tf_model = args['tf_model']
	columns = args['columns']
	maxnoisesegs = args["max_noise_segments"]

	inference_rates = np.array([int(x.split("_")[0][:-2]) for x in columns])
	window_sizes = [float(x.split("_")[1][:-1]) for x in columns]
	window_sizes = np.array(window_sizes * inference_rates, dtype= np.int32)

	print("Using the following inference rates for column data:")

	print("using the following network: ", tf_model)

	double_det, combiner, full_model = split_model_stack(tf_model)
	print("num_time_slides is", num_time_slides)

	println("cleanup ID is {} and there are {} cleanups".format(cleanup_id, n_cleanups))

	#time_shift = 2048
	#time_shift_var = 256
	sample_rate = 2048
	duration = 900
	light_travel_time = sample_rate//100  #max light travel time between H and L. In units of datapoints, not ms


	print("JOBFS", os.environ["JOBFS"] )

	statusfolder = os.path.join(os.environ["JOBFS"], "job_" +str(job_id), "completed") #this folder is used to shut down the triton server
	myfolder = os.path.join(os.environ["JOBFS"], "job_" +str(job_id), "worker_"+str(worker_id))
	os.chdir(myfolder)
	print("my folder is", myfolder)

	println("starting job {} of {}".format(job_id, n_jobs))
	print("I am cleanup job {} of {} for this server".format(worker_id, n_workers))
	job_id = worker_id + job_id*n_workers
	print("my unique index is {}".format(job_id))
	n_jobs = n_jobs * n_workers
	print("there are {} jobs in total".format(n_jobs))
     
	#this loop is necessary as there is an edge case where the args.json file has been created by the worker,
	#but the worker has not written its contents yet. 
	while True:
		try:
			args = json.load(open("args.json", "r"))
			print("args are", args)
			break
		except:
			print("waiting for args.json")
			time.sleep(5)

	template_start = args["template_start"]
	batch = 0  # template batch count
	segment = 0  # noise segment count

	segment += cleanup_id
	print("my start segment is", segment)

	injs = True if args["injection_file"] == 1 else False
	print("injs is", injs)
	if injs:
		num_time_slides = 0
		print("switching to injection run mode")
		print("ERROR: should not have gotten here. This code is not used in injection runs")

	else:
		print("switching to background run mode")    

	gps_blacklist = args["gps_blacklist"]
	valid_times = args["valid_times"]

	#convolution kernels for moving average
	f16 = np.ones(args["inference_rate"])/args["inference_rate"]
	f12 = np.ones(12)/12
	f8 = np.ones(8)/8
	f4 = np.ones(4)/4
	f2 = np.ones(2)/2

	ma_kernels = {16: f16, 12: f12, 8: f8, 4: f4, 2: f2}

	while True:
		files = os.listdir(myfolder)

		SNR_substring = 'SNR_batch_{}_segment_{}_'.format(batch, segment)
		preds_substring = 'preds_batch_{}_segment_{}_'.format(batch, segment)
		if any(SNR_substring in file for file in files) \
				and any(preds_substring in file for file in files):
			print("Found files", files) 
			sys.stdout.flush() 

		else:
			print("no files found")
			sys.stdout.flush()
			time.sleep(10)
			continue

		# Load SNR and predictions
		SNR_file = [file for file in files if SNR_substring in file][0]
		preds_file = [file for file in files if preds_substring in file][0]

		print("loading SNR and preds file:", SNR_file,preds_file)

		time.sleep(1)
		sys.stdout.flush()

		while True:
			try:
				SNR = np.load(SNR_file)
				preds = np.load(preds_file)
				break
			except:
				print("failed to load SNR and preds, trying again")
				time.sleep(5)

		#separate out the chop time from the file name

		chop_time = int(SNR_file.split("_")[-1].split(".")[0]) 
		print("chop time is", chop_time)
		chop_idx = chop_time * sample_rate
		#if n_templates != SNR.shape[1]:
		#    print("This batch of SNR has a different number of templates ({} vs {})! This is the correct behaviour it's the last batch...".format(SNR.shape[1], n_templates))
		n_templates = SNR.shape[1]


		deleted_zerolags = []
		delete_times = []

		for gps_time in gps_blacklist:
			if gps_time > valid_times[segment] and gps_time < valid_times[segment] + duration:
				delete_time = int(gps_time - valid_times[segment] - chop_time)

				print("deleted zerolag at time", delete_time)
				print("Actual GPS time of deleted event:", gps_time)
				deleted_zerolags.append(gps_time)
				delete_times.append(delete_time)

		if len(deleted_zerolags) > 0:
			print("deleted zerolags:", deleted_zerolags)

		preds_reshaped = np.array(np.split(preds, 2, axis = -1))


		#TODO: verify that this behaviour is correct with different inference rates
		peak_pos_array = []
		#windows, detectors, templates
		for i in range(0, SNR.shape[2]-2048+sample_rate//args["inference_rate"], sample_rate//args["inference_rate"]):

			peak_pos_array.append(np.argmax(SNR[:, :, i:i+2048], axis = 2))
		peak_pos_array = np.array(peak_pos_array)


		windowed_sample_end_indexes = list(range(sample_rate-1, SNR.shape[-1], sample_rate//args["inference_rate"]))
		windowed_sample_start_indexes = list(np.copy(windowed_sample_end_indexes) - (sample_rate - 1))
		start_end_indexes = list(zip(windowed_sample_start_indexes, windowed_sample_end_indexes))

		zerolags = get_zerolags(
		data = SNR,
		snr_thresh = 4,
		offset = 20,
		buffer_length = 2048,
		overlap = int(0.2*2048),
		num_trigs = 1,
		chop_time = chop_idx,
		)

		zl_array = np.full((len(zerolags), 1, 14), -1, dtype=np.float32)
		stat_array = np.full((len(zerolags), num_time_slides, 11), -1, dtype=np.float32)

		#setting prediction entries to -1000 as predictions can easily be below -1, in which case they wouldn't be replaced.
		zl_array[:,:,6:] = -1000
		stat_array[:,:,3:] = -1000

		#TODO: might not need to copy it in the first place. just roll the same array by 2048 each timeslide.
		SNR_rolled = np.copy(SNR)

		ifo_pred_len = preds.shape[2]//2

		#make a h and l pred array for each unique inference rate
		pred_arrays = {}
		for rate in inference_rates:
			pred_arrays[rate] = {"h": [], "l": [], "delta_t_array": []}

		#h_pred_array = []
		#l_pred_array = []

		zl_id = []
		ts_id = []

		#delta_t_array = []

		#NOTE: the +1 is to include the zerolag as the 0th timeslide. the zl array is saved as a check that the timeslides are correct.
		#i.e. the distribution of timeslide predictions should be the same as the distribution of zerolags.

		for i in range(-1,num_time_slides):
			#for each timeslide
			#roll the Livingston SNR
			
			SNR_rolled[1,:,:] = np.roll(SNR[1,:,:], 2048*(i+1), axis = 1)
			#TODO: also roll pred array! will probably make everything else a lot easier.
			#preds_reshaped[1] = np.roll(preds_reshaped[1], 16*(i+1), axis = 1)
			#peak_pos_array[:,1] = np.roll(peak_pos_array[:,1], 16*(i+1), axis = 0)

			#need to roll by 16 each time.

			#get the zerolags for this timeslide

			timeslides = get_zerolags(
				data = SNR_rolled,
				snr_thresh = 4,
				offset = 20,
				buffer_length = 2048,
				overlap = int(0.2*2048),
				num_trigs = 1,
				chop_time = chop_idx,
			)

			timeslides = np.array(timeslides).squeeze()

			#note: primary and secondary detectors aren't really a thing 

			#use deleted_zerolags to remove invalid timeslides
			#if second p is in deleted_zerolags, we delete second p of timeslide i but also second (p + i +1) %900
			#i.e. if second 300 is deleted, and we're on timeslide 1, we delete seconds 300 and 301
			#if second 300 is deleted, and we're on timeslide 

			#need to subtract sample_rate * i from the livingston indices

			timeslides[:,4] = (timeslides[:,4] - 2048 * (i+1)) % (900 * 2048)
			
			#NOTE: need to be careful here, if we're doing an invalid timeslide we're setting the L pos to something other than -1
			# 2 * 2048 % (900 * 2048)


			for j in range(len(timeslides)):
				#for each second
				#NOTE: len(timeslides) can be different if chop_time is not zero.
				if not is_timeslide_valid(j,i, chop_time):
					#set the timeslide to -1
					timeslides[j] = -1
					continue
				
				if timeslides[j][0] == -1:
					continue

				if j == 0 or j == len(timeslides) -1:
					timeslides[j] = -1
					continue

				#on the ith timeslide, we have moved the data i+1 seconds
				#TODO: factor in chop time, I think it's needed here
				
				#TODO: change how deleted_zerolags works so that you're saving the timeslide and not the gps time to a list

				if j + chop_time in delete_times or (j + chop_time + i + 1) % 900 in delete_times:
					timeslides[j] = -1
					print("seconds {} and {} are invalid due to real event. Zl {}, TS {}".format(j, (j + chop_time + i + 1) % 900, j + chop_time, i))
					continue

				# hanford's stuff is at index j
				# livingston's stuff is at index (j - i + 1) % 900
				
				#primary_detector = np.argmax(timeslides[j, :2])

				#TODO: remove assumption that there's a 16hz inference rate
				H_det_samples = get_windows(start_end_indexes, timeslides[j, 3])
				L_det_samples = get_windows(start_end_indexes, timeslides[j, 4])

				for rate in pred_arrays:
					pred_arrays[rate]["H_det_samples"] = get_windows(start_end_indexes, timeslides[j, 3], stride = sample_rate//rate) * int(16/rate)
					pred_arrays[rate]["L_det_samples"] = get_windows(start_end_indexes, timeslides[j, 4], stride = sample_rate//rate) * int(16/rate)
				
				#TODO: clean up
				#H_det_samples = get_windows(start_end_indexes, timeslides[j, 3+primary_detector])
				#L_det_samples = H_det_samples
				
				template = int(timeslides[j, 5])

				#once we're past this check, we know that all included timslides are valid
				if len(H_det_samples) < args["inference_rate"] + 2 or len(L_det_samples) < args["inference_rate"] + 2 \
					or H_det_samples[0] < 0 or H_det_samples[-1] >= len(start_end_indexes) or L_det_samples[0] < 0 or L_det_samples[-1] >= len(start_end_indexes):
					#set the timeslide to -1
					timeslides[j] = -1
					#print(len(H_det_samples), len(L_det_samples))
					#print("timeslide too short")
					continue

				for rate in pred_arrays:
					h_samples = pred_arrays[rate]["H_det_samples"]
					l_samples = pred_arrays[rate]["L_det_samples"]

					pred_arrays[rate]['h'].append(preds_reshaped[0, template, h_samples])
					pred_arrays[rate]['l'].append(preds_reshaped[1, template, l_samples])
					pred_arrays[rate]['delta_t_array'].append((peak_pos_array[h_samples, 0, template] - peak_pos_array[l_samples, 1, template])/light_travel_time)
					#pred_arrays[rate]['zl_id'].append(j)
					#pred_arrays[rate]['ts_id'].append(i)

				#h_pred_array.append(preds_reshaped[0, template, H_det_samples])
				#l_pred_array.append(preds_reshaped[1, template, L_det_samples])

				#delta_t_array.append((peak_pos_array[H_det_samples, 0, template] - peak_pos_array[L_det_samples, 1, template])/light_travel_time)

				zl_id.append(j)
				ts_id.append(i)

				#pretty much ready to go now! just need to reshape, then pass to combiner.
			
			if i != -1:
				stat_array[:, i, :3] = timeslides[:,:3]


		#h_pred_array = np.array(h_pred_array)  
		#l_pred_array = np.array(l_pred_array)

		#h_pred_array = h_pred_array.reshape(-1, ifo_pred_len)
		#l_pred_array = l_pred_array.reshape(-1, ifo_pred_len)

		for rate in pred_arrays:
			pred_arrays[rate]['h'] = np.array(pred_arrays[rate]['h']).reshape(-1, ifo_pred_len)
			pred_arrays[rate]['l'] = np.array(pred_arrays[rate]['l']).reshape(-1, ifo_pred_len)

			if len(combiner.input) > 2:
				pred_arrays[rate]['delta_t_array'] = np.array(pred_arrays[rate]['delta_t_array']).reshape(-1,1)
				combined_preds = combiner.predict([pred_arrays[rate]['h'], pred_arrays[rate]['l'], 
									   pred_arrays[rate]['delta_t_array']], verbose = 2, batch_size = 1024)
			
			else:
				combined_preds = combiner.predict([pred_arrays[rate]['h'], pred_arrays[rate]['l']], verbose = 2, batch_size = 1024)
			
			combined_preds = combined_preds.reshape(-1, rate + 2)

			pred_arrays[rate]['combined_preds'] = combined_preds

			println("combined preds shape for {}hz: {}".format(rate, combined_preds.shape))

		#if len(combiner.input) > 2:
		#	delta_t_array = np.array(delta_t_array).reshape(-1,1)
		#	combined_preds = combiner.predict([h_pred_array, l_pred_array, delta_t_array], 
		#										verbose = 2, batch_size = 1024)

		#else:
		#	combined_preds = combiner.predict([h_pred_array, l_pred_array], 
		#										verbose = 2, batch_size = 1024)
			
		
		#combined_preds = combined_preds.reshape(-1, 18)

		#println("combined preds shape: {}".format(combined_preds.shape))

		zl_array[:,0,:6] = np.array(zerolags)[:, 0, :6]
		
		for idx, rate in enumerate(inference_rates):
			#print("using conv kernel of size", window_sizes[idx])
			if "pred_array" not in pred_arrays[rate]:
				pred_arrays[rate]["pred_array"] = np.full((len(timeslides), num_time_slides+1, rate+2), -1000, dtype = np.float32)

				for i in range(len(zl_id)):
					#the +1 i
					pred_arrays[rate]["pred_array"][zl_id[i], ts_id[i] + 1] = pred_arrays[rate]['combined_preds'][i]
				
			x = np.apply_along_axis(lambda m: np.convolve(m, ma_kernels[window_sizes[idx]], mode = 'valid'), 2, pred_arrays[rate]["pred_array"])
			#pred_arrays[rate][window_sizes[idx]] 
			ma_prediction = np.max(x, axis = -1)
			
			zl_array[:, 0, 6 + idx] = ma_prediction[:,0]
			stat_array[:, :, 3 + idx] = ma_prediction[:,1:]

		"""
		pred_array = np.full((len(timeslides), num_time_slides+1, 18), -1000, dtype = np.float32)

		for i in range(len(zl_id)):
			#the +1 i
			pred_array[zl_id[i], ts_id[i] + 1] = combined_preds[i]

		x = np.apply_along_axis(lambda m: np.convolve(m, f16, mode = 'valid'), 2, pred_array)
		ma_prediction_16hz_16 = np.max(x, axis = -1)

		x = np.apply_along_axis(lambda m: np.convolve(m, f12, mode = 'valid'), 2, pred_array)
		ma_prediction_16hz_12 = np.max(x, axis = -1)

		x = np.apply_along_axis(lambda m: np.convolve(m, f8, mode = 'valid'), 2, pred_array)
		ma_prediction_16hz_8 = np.max(x, axis = -1)

		x = np.apply_along_axis(lambda m: np.convolve(m, f4, mode = 'valid'), 2, pred_array)
		ma_prediction_16hz_4 = np.max(x, axis = -1)

		max_16hz = np.max(pred_array[:,:, 1:-1], axis = -1)

		#remember, the first timeslide is the zerolag which needs to be saved to zerolag_array
		zl_array[:,0,:6] = np.array(zerolags)[:, 0, :6]
		zl_array[:,0,6] = ma_prediction_16hz_16[:,0]
		zl_array[:,0,7] = ma_prediction_16hz_12[:,0]
		zl_array[:,0,8] = ma_prediction_16hz_8[:,0]
		zl_array[:,0,9] = ma_prediction_16hz_4[:,0]
		zl_array[:,0,10] = max_16hz[:,0]

		stat_array[:, :, 3] = ma_prediction_16hz_16[:,1:]
		stat_array[:, :, 4] = ma_prediction_16hz_12[:,1:]
		stat_array[:, :, 5] = ma_prediction_16hz_8[:,1:]
		stat_array[:, :, 6] = ma_prediction_16hz_4[:,1:]
		stat_array[:, :, 7] = max_16hz[:,1:]
		"""		

		#TODO: need to modify saved stats, as we should save template information since each timeslide is treated independently...
		zl_array[:,0,5] += template_start

		print("saving to zerolag file: zerolags_{}-{}_batch_{}_segment_{}.npy".\
				format(template_start, template_start + n_templates, batch, segment))
		
		np.save(os.path.join(savedir, "zerolags_{}-{}_batch_{}_segment_{}.npy".\
				format(template_start, template_start + n_templates, batch, segment)), zl_array)
		
		if not injs:
			np.save(os.path.join(savedir, "stats_{}-{}_batch_{}_segment_{}.npy".\
							format(template_start, template_start + n_templates, batch, segment)), stat_array)

		os.remove(SNR_file)
		os.remove(preds_file)

		time.sleep(1)
		#delete all the variables to free up memory

		del SNR, zerolags, stat_array, zl_array, combined_preds, timeslides
		del x, pred_arrays

		#del SNR, zerolags, h_pred_array, l_pred_array, stat_array, zl_array, combined_preds, timeslides
		#del x, ma_prediction_16hz_16, ma_prediction_16hz_12, ma_prediction_16hz_8, ma_prediction_16hz_4, max_16hz

		gc.collect()

		segment += n_cleanups
		if batch == args['n_batches'] - 1 and segment >= args['n_noise_segments']:
			print("main job should have finished")
			break
		elif segment >= args['n_noise_segments']:
			template_start += n_templates
			batch += 1
			segment = cleanup_id
			print("batch is now", batch)

		print("segment is now", segment)

	print("finished doing cleanup")

	#write a file to the status folder to indicate that this job is done
	with open(os.path.join(statusfolder, "worker_{}_{}.txt".format(worker_id,cleanup_id)), "w") as f:
		f.write("done")
	time.sleep(1)
	#print the contents of the status folder

	print("status folder contents:")
	print(os.listdir(statusfolder))