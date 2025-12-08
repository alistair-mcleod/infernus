import json
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from infernus.postprocessing import preds_to_far_constrained, lognorm_fit_constrained_print, pdf_to_cdf_arbitrary, get_inj_data, get_dsens
from infernus.real_utils import get_GWTC_events
from pycbc.sensitivity import volume_to_distance_with_errors
from astropy.cosmology import FlatwCDM
cosmo = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)


def apply_func_to_bg_inj(bg,inj, func, rs_index = 8):
	
	return func(bg[:,:,rs_index], axis = 1), func(inj[:,:,rs_index], axis = 1)


def mdc_results(m, inj_file, bg, n_models,mdc_files, noise_dir, max_bg_remove = 5):
	#we need to return the results for ALL models for this MDC file
	mdc_inj_results = {}
	print("Number of models: ", n_models)

	print("Processing MDC file: ", mdc_files[m])
	print("Injfiles are in: ", inj_file[m])
	N_draw, mask, inj_params = get_inj_data(4, noise_dir, mdc_files[m], pipelines = ["pycbc_hyperbank", "mbta", "gstlal"])

	rs_index = 8
	zerolags = np.load(inj_file[m])[0] #the 0 is to get rid of the timeslides axis
	#if any triggers don't have a valid network SNR, set model predictions to -1000.
	#we can't modify the shape as we need to keep the same indexing
	zerolags[np.any(zerolags[:,:,2] < 1, axis = 1), :, 8:] = -1000

	m1 = inj_params["m1"]
	m2 = inj_params["m2"]
	pipeline_fars = inj_params["pipeline_fars"]
	z = inj_params["z"]
	s1x = inj_params["s1x"]
	s1y = inj_params["s1y"]
	s1z = inj_params["s1z"]
	s2x = inj_params["s2x"]
	s2y = inj_params["s2y"]
	s2z = inj_params["s2z"]
	p_draw = inj_params["p_draw"]
	N_draw = inj_params["N_draw"]

	pipelines = ['pycbc_hyperbank', 'mbta', 'gstlal']


	#VT, sigma_VT = get_dsens(z, p_draw, N_draw, pipelines, pipeline_fars)

	upper_thresh = 1e-3
	lower_thresh = 1e-7
	OPA_threshold = 1/(3600*24*30*2)
	#sys.path.append("/fred/oz016/alistair/infernus/infernus")
	#from infernus.postprocessing import preds_to_far_constrained, lognorm_fit_constrained_print,pdf_to_cdf_arbitrary
	mdc_inj_results["inj_params"] = inj_params
	
	for i in range(8, 8+n_models):
		_, inj_func = apply_func_to_bg_inj(zerolags[:2], zerolags, np.median, rs_index = i)
		#
		nn_preds = np.full(mask.sum(), -1000.0)
		nn_preds[inj_params["inj_ids"]] = inj_func[inj_params["zerolags"]]
		#look 1 index either side of inj_params['zerolags'] as the peak might be slightly off
		zl_before = np.array(inj_params['zerolags']) -1
		zl_after = np.array(inj_params['zerolags']) +1
		nn_preds[inj_params["inj_ids"]] = np.maximum(nn_preds[inj_params["inj_ids"]], inj_func[zl_before])
		nn_preds[inj_params["inj_ids"]] = np.maximum(nn_preds[inj_params["inj_ids"]], inj_func[zl_after])

		nn_preds = np.nan_to_num(nn_preds, nan = 0)
		print(nn_preds)
		print(bg[i-8])
		preds = preds_to_far_constrained(bg[i-8], nn_preds, upper = upper_thresh, lower = lower_thresh, verbose = False)
		#preds = np.nan_to_num(preds, nan = 0)
		
		pipeline_fars['NN'+str(i)] = preds
		if 'NN'+str(i) not in pipelines:
			pipelines.append('NN'+str(i))

		pipeline_fars['NN'+str(i)][pipeline_fars['NN'+str(i)] == 1] = np.inf

		bg_max = bg[i-8][-max_bg_remove]
		print("events with prediction higher than background (note: {} highest BG points are removed):".format(max_bg_remove), (nn_preds > bg_max).sum())

		mdc_inj_results[str(i)] = {}
		mdc_inj_results[str(i)]["pipeline_fars"] = pipeline_fars
		mdc_inj_results[str(i)]["nn_preds"] = nn_preds


	mdc_inj_results["pipelines"] = pipelines
	found = {}
	for p in pipelines:
		found[p] = (pipeline_fars[p] < OPA_threshold)
	mdc_inj_results["found"] = found


	return mdc_inj_results

#n_models = bg.shape[2] - 8



def combined_model_results(fps):

	#fps = [fp0, fp1, fp2]
	n_bins = len(fps)
	all_stats = {}
	for bin in range(n_bins):

		print("PROCESSING BIN ", bin)
		submit_args = json.load(open(fps[bin]))
		inj_args = json.load(open(submit_args['injection_args']))
		bg_args = json.load(open(submit_args['background_args']))
		model_args = json.load(open(submit_args['model_args']))
		n_models = model_args["final_model_candidates"]
		noise_dir = inj_args["noise_dir"]
		mdc_file = inj_args["injfile"]
		if isinstance(mdc_file, list):
			print("List of injection files detected,")
		else:
			mdc_file = [mdc_file]
			#mdc_file = mdc_file[int(inj_args['bin'][-1])]
		model_val_dir = os.path.join(inj_args['jobdir'], "results", inj_args['bin'])
		os.makedirs(model_val_dir, exist_ok = True)
		print("Model val dir: ", model_val_dir)
		bg_file = os.path.join(bg_args['save_dir'], "timeslides.npy")
		inj_file = [os.path.join(inj_args['save_dir'], "inj_{}".format(i), "timeslides.npy") for i in range(len(mdc_file))]
		#print("TODO: generalise to multiple injection files!")
		real_dir_root = os.path.join(inj_args['jobdir'], "real_events", inj_args['bin'])
		real_dirs = sorted(os.listdir(real_dir_root))
		

		print("Loading data from ", bg_file, inj_file)

		print("MDC file(s): ", mdc_file)

		bg = np.load(bg_file)
		bg = bg.astype(np.float32)
		bg = bg.reshape(-1, bg.shape[2], bg.shape[3])
		#this check ensures al samples have a network SNR > 0 (i.e. that the sample is valid)
		bg = bg[np.all(bg[:,:,2] > 0, axis = 1)]

		ratio2 = bg[:,:,0] / bg[:,:,1]
		#we need to invert the ratio so it's always >= 1
		ratio2[ratio2 < 1] = 1/ratio2[ratio2 < 1]
		ratio2 = np.any(ratio2 > 10, axis = 1)
		bg = bg[~ratio2]
		print("Background shape after removing high ratio samples: ", bg.shape)

		bg_sort = []
		#now we get the backgrounds for each model
		zerolags = np.load(inj_file[0])[0] #the 0 is to get rid of the timeslides axis
		for i in range(8, bg.shape[2]):
			bg_func, _ = apply_func_to_bg_inj(bg, zerolags, np.median, rs_index = i)
			bg_func = bg_func[np.where(np.isfinite(bg_func))]
			bg_func = np.sort(bg_func)
			bg_sort.append(bg_func)

		bin_stats = {}
		for m in range(len(mdc_file)):
			#print("Processing bin ", bin)
			#print("Processing MDC file: ", mdc_file[m])
			ret = mdc_results(m, inj_file, bg_sort, n_models, mdc_file, noise_dir)
			bin_stats[m] = ret
		bin_stats["bg_sort"] = bg_sort
		all_stats[bin] = bin_stats
	
	OPA_threshold = 1/(3600*24*30*2)

	#for now weights are 1/n_bins
	weights = [1/n_bins for _ in range(n_bins)]
	#weights = [1, 1, 1]
	#now we combine the results from all bins

	all_stats["combined"] = {}
	all_stats["combined"]["best_model"] = []
	for b in range(n_bins):

		bin_idx = b
		#get n_models from the args (mostly legacy from when we used different n_models)
		submit_args = json.load(open(fps[b]))
		model_args = json.load(open(submit_args['model_args']))
		n_models = model_args["final_model_candidates"]
		model_found = np.zeros((n_models), dtype = int)
		#model_found = []
		for m in range(len(mdc_file)):
			all_stats["combined"][m] = {}
			#all_stats["combined"][m]["pipeline_fars"] = {}
			
			inj_params = all_stats[bin_idx][m]['inj_params']
			#the best model is the one with the highest number of detections below the OPA threshold after weighting
			this_found = np.array([(inj_params['pipeline_fars']["NN{}".format(i)]/weights[bin_idx] < OPA_threshold).sum() for i in range(8, n_models+8)])
			print("Model counts for MDC file ", mdc_file[m])
			print("NN model detections below OPA threshold: ", this_found)
			model_found += this_found

		best_model_idx = np.argmax(model_found) + 8
		print()
		print("Best model is NN{} with {} total detections below OPA threshold".format(best_model_idx, model_found[best_model_idx -8]))
		print()
		all_stats["combined"]["best_model"].append(best_model_idx)

	#now for each MDC file, we can get the detection lists for the best model
	#essentially we flatten the bin axis.
	for m in range(len(mdc_file)):
		far_combined = all_stats[0][m]['inj_params']['pipeline_fars']["NN"+str(all_stats["combined"]["best_model"][0])]/weights[0]
		print("FAR contribution from bin 0: ", far_combined)
		#all_stats["combined"][m]['pipeline_fars'] = all_stats[0][m]['inj_params']['pipeline_fars'][all_stats["combined"]["best_model"][0]]
		for b in range(1,3):
			bin_idx = b
			far_bin = all_stats[bin_idx][m]['inj_params']['pipeline_fars']["NN"+str(all_stats["combined"]["best_model"][b])]/weights[b]
			print("FAR contribution from bin {}: ".format(b), far_bin)
			far_combined = np.minimum(far_combined, far_bin)
		
		all_stats["combined"][m]['pipeline_fars'] = all_stats[0][m]['inj_params']['pipeline_fars'].copy()
		#now remove any existing NN models
		#for i in range(8, bg.shape[2]):
		#	if 'NN'+str(i) in all_stats["combined"][m]['pipeline_fars']:
		#		del all_stats["combined"][m]['pipeline_fars']['NN'+str(i)]
		#add in the combined model
		all_stats["combined"][m]['pipeline_fars']['NN_combined'] = far_combined
		all_stats["combined"][m]['inj_params'] = all_stats[0][m]['inj_params']
		all_stats["combined"][m]['found'] = {}
		for p in all_stats["combined"][m]['inj_params']['pipeline_fars'].keys():
			all_stats["combined"][m]['found'][p] = (all_stats["combined"][m]['inj_params']['pipeline_fars'][p] < OPA_threshold)
		all_stats["combined"][m]['found']['NN_combined'] = (far_combined < OPA_threshold)
	#now print far_combined
	return all_stats




def mdc_results_summary(stats, bin, n_models):
	#TODO: figure out bin index for stats lol
	found = stats[0][0]['found']
	print("Model counts:")
	for i in range(8,n_models +8):
		print("NN{}: ".format(i), found['NN'+str(i)].sum())

	print("\npycbc, mbta, gstlal:")
	print(found['pycbc_hyperbank'].sum(), found['mbta'].sum(), found['gstlal'].sum())
	print("")
	print("PyCBC unique events: ", (found['pycbc_hyperbank'] & ~found['mbta'] & ~found['gstlal']).sum())
	print("MBTA unique events: ", (~found['pycbc_hyperbank'] & found['mbta'] & ~found['gstlal']).sum())
	print("GSTLAL unique events: ", (~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal']).sum())
	print("")

	for i in range(8,n_models +8):
		print("unique events in NN{}: ".format(i), (found['NN'+str(i)] & ~found['pycbc_hyperbank'] \
												& ~found['mbta'] & ~found['gstlal']).sum())


	print("")
	for i in range(8,n_models +8):
		print("NN{} duo detection with PyCBC: ".format(i), (found['NN'+str(i)] & found['pycbc_hyperbank'] & ~found['mbta'] & ~found['gstlal']).sum())
		print("NN{} duo detection with MBTA: ".format(i), (found['NN'+str(i)] & ~found['pycbc_hyperbank'] & found['mbta'] & ~found['gstlal']).sum())
		print("NN{} duo detection with GSTLAL: ".format(i), (found['NN'+str(i)] & ~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal']).sum())

	print("")

	for i in range(8,n_models +8):
		print("unique PyCBC events AFTER adding NN{}: ".format(i), (found['pycbc_hyperbank'] & ~found['NN'+str(i)] & ~found['mbta'] & ~found['gstlal']).sum())
		print("unique MBTA events AFTER adding NN{}: ".format(i), (~found['pycbc_hyperbank'] & found['mbta'] & ~found['NN'+str(i)] & ~found['gstlal']).sum())
		print("unique GSTLAL events AFTER adding NN{}: ".format(i), (~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal'] & ~found['NN'+str(i)]).sum())

	#criterion for best pipeline per bin is the most detections below the OPA threshold




def plot_sensitivities(inj_params, pipeline_fars, nn_name, mdc, save_fp, mdc_idx):

	m1 = inj_params["m1"]
	m2 = inj_params["m2"]
	#pipeline_fars = inj_params["pipeline_fars"]
	z = inj_params["z"]
	s1x = inj_params["s1x"]
	s1y = inj_params["s1y"]
	s1z = inj_params["s1z"]
	s2x = inj_params["s2x"]
	s2y = inj_params["s2y"]
	s2z = inj_params["s2z"]
	p_draw = inj_params["p_draw"]
	N_draw = inj_params["N_draw"]


	M1= np.array([40, 30, 20, 10])
	M2 = np.array([1.4, 1.4, 1.4, 1.4])
	OPA_threshold = 1/(3600*24*30*2)

	#M1= np.array([2, 2, 1.4, 1.4])
	#M2 = np.array([2, 1.4, 1.4, 1])

	m2_full_pop = True
	m1_full_pop = True

	if "bns" in mdc:
		m1_full_pop = True
		m2_full_pop = True
		print("BNS, using full population")

	elif "nsbh" in mdc:
		m1_full_pop = False
		m2_full_pop = True
		M1 = np.array([40, 30, 20, 10])
		M2 = np.array([1.4, 1.4, 1.4, 1.4])

	elif "bbh" in mdc:
		print("BBH, using fixed masses")
		m1_full_pop = False
		m2_full_pop = False
		M1 = np.array([40, 30, 20, 10])
		M2 = np.array([40, 30, 20, 10])
		y_axlims = [400,400,150,50]

	fars = np.geomspace(1e-3, 1e-12, 50)

	#for m in range(8,bg.shape[2]):
	fig, axes = plt.subplots(2,2, figsize=(10,8), sharex=True, dpi = 130)

	for i, ax in enumerate(axes.flatten()):
		
		sig_lognorm_m1 = 0.05
		sig_lognorm_m2 = 0.2

		smax_ns = 0.4
		smax_bh = 0.998
		cosmo = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)

		m1_mean = M1[i]
		m2_mean = M2[i]

		pop_params = {
			'm1_mean': m1_mean,
			'm2_mean': m2_mean,
			'sig_lognorm_m1': sig_lognorm_m1,
			'sig_lognorm_m2': sig_lognorm_m2,
			'smax_ns': smax_ns,
			'smax_bh': smax_bh,
			'cosmo': cosmo,
			'm1_full_pop': m1_full_pop,
			'm2_full_pop': m2_full_pop,
			'm1_m2_prior': inj_params['m1_m2_prior'],
			's1_s2_prior': np.log(inj_params['s1_prior']) + np.log(inj_params['s2_prior'])
		}

		pipelines = ['pycbc_hyperbank', 'mbta', 'gstlal']
		pipelines.append(nn_name)

		#print("Pipelines for this plot: ", pipelines)
		VT, sigma_VT = get_dsens(z, p_draw, N_draw, pipelines, pipeline_fars,m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, pop_params = pop_params)
		
		for p in pipelines:
			vt = np.array(VT[p])
			sigma = np.array(sigma_VT[p])
			dist, ehigh, elow = volume_to_distance_with_errors(vt*1e9, sigma*1e9)


			ax.plot(fars, dist, label=p, linewidth=1, alpha = 0.7, zorder = 6)
			ax.fill_between(fars, dist-elow, dist+ehigh, alpha=0.3, ec = 'none', zorder = 6)

		if m2_full_pop and m1_full_pop:
			ax.set_title(f"M1 = any, M2 = any")
		elif m2_full_pop:
			ax.set_title(f"M1 = {m1_mean}, M2 = any")
		else:
			ax.set_title(f"M1 = {m1_mean}, M2 = {m2_mean}")
		ax.set_xscale('log')
		ax.set_xlim(1e-3,1e-11)
		ax.axvline(OPA_threshold, color = 'red', linestyle = '--', alpha = 0.5, linewidth = 1, label = "Detection threshold")
		ax.grid(zorder = -10)
		ax.legend()
		ax.set_ylabel("Sensitive distance (Mpc)")
		ax.set_xlabel("False alarm rate (Hz)")
		if "bbh" in mdc:
			ax.set_ylim(y_axlims[i])
	plt.tight_layout()
	plt.show()
	plt.savefig(os.path.join(save_fp, "combined_sens_vs_far_{}.png".format(mdc_idx)), dpi = 300)
	plt.clf()



def combined_pipeline_mdc_summary(all_stats, mdc_idx):
	found = all_stats['combined'][mdc_idx]['found']

	print("Combined pipeline:", found['NN_combined'].sum())
	print("\npycbc, mbta, gstlal:")
	print(found['pycbc_hyperbank'].sum(), found['mbta'].sum(), found['gstlal'].sum())

	print("")
	print("PyCBC unique events: ", (found['pycbc_hyperbank'] & ~found['mbta'] & ~found['gstlal']).sum())
	print("MBTA unique events: ", (~found['pycbc_hyperbank'] & found['mbta'] & ~found['gstlal']).sum())
	print("GSTLAL unique events: ", (~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal']).sum())
	print("")

	print("Unique combined events: ", (found['NN_combined'] & ~found['pycbc_hyperbank'] \
												& ~found['mbta'] & ~found['gstlal']).sum())

	print("")
	print("NN_combined duo detection with PyCBC: ", (found['NN_combined'] & found['pycbc_hyperbank'] & ~found['mbta'] & ~found['gstlal']).sum())
	print("NN_combined duo detection with MBTA: ", (found['NN_combined'] & ~found['pycbc_hyperbank'] & found['mbta'] & ~found['gstlal']).sum())
	print("NN_combined duo detection with GSTLAL: ", (found['NN_combined'] & ~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal']).sum())

	print("")

	print("unique PyCBC events AFTER adding NN_combined: ", (found['pycbc_hyperbank'] & ~found['NN_combined'] & ~found['mbta'] & ~found['gstlal']).sum())
	print("unique MBTA events AFTER adding NN_combined: ", (~found['pycbc_hyperbank'] & found['mbta'] & ~found['NN_combined'] & ~found['gstlal']).sum())
	print("unique GSTLAL events AFTER adding NN_combined: ", (~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal'] & ~found['NN_combined']).sum())


#get filepaths somehow... 

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Generate combined pipeline summary for multiple bins")
	parser.add_argument('--configfile', type=str, required=True, help='Path to config file used to generate the run')
	args = parser.parse_args()
	submit_args = json.load(open(args.configfile))
	n_bins = len(submit_args['bins'])

	fps = []
	for b in submit_args['bins']:
		fps.append(os.path.join(submit_args['global']['run_dir'], "configs/bin_{}/submit.json".format(b)))
	print("Filepaths: ", fps)
	#get the number of MDC files from the configs
	mdc_file = submit_args['models']['injfile']
	print("MDC files: ", mdc_file)

	#this line does most of the work
	all_stats = combined_model_results(fps)



	upper_thresh = 1e-3
	lower_thresh = 1e-7
	OPA_threshold = 1/(3600*24*30*2)


	#plot the sensitivities for each MDC file
	for mdc_idx in range(len(mdc_file)):
		#mdc_idx = 2
		inj_params = all_stats['combined'][mdc_idx]['inj_params']
		pipeline_fars = all_stats['combined'][mdc_idx]['pipeline_fars']
		mdc = mdc_file[mdc_idx]

		combined_pipeline_mdc_summary(all_stats, mdc_idx)

		plot_sensitivities(inj_params, pipeline_fars, 'NN_combined', mdc, os.path.join(submit_args['global']['run_dir'], "results"), mdc_idx)


	event_stats = {}
	detection_sum = 0
	total_events = 0

	for bin in range(len(fps)):

		submit_args = json.load(open(fps[bin]))
		inj_args = json.load(open(submit_args['injection_args']))


		model_idx = all_stats['combined']['best_model'][bin]
		bg = all_stats[bin]['bg_sort'][model_idx - 8]
		print("Processing bin ", bin, " with model NN", model_idx)

		real_idx = inj_args['duration']//2 - 100
		real_dir_root = os.path.join(inj_args['jobdir'], "real_events", inj_args['bin'])
		real_dirs = sorted(os.listdir(real_dir_root))

		for d in real_dirs:
			if d.startswith("GW"):
				print("Processing event", d)
				real_dir = os.path.join(real_dir_root, d)

				if d not in event_stats:
					event_stats[d] = {}
					event_stats[d]['model_bin'] = bin
					event_stats[d]['model_idx'] = model_idx
					event_stats[d]['FAR'] = 1
					event_stats[d]['SNR'] = -1

					total_events += 1
				
				real_event = np.load(os.path.join(real_dir, "timeslides.npy"))[0]

				_, real_func = apply_func_to_bg_inj(real_event, real_event, np.median, rs_index = model_idx)
				real_preds = preds_to_far_constrained(bg, real_func, upper = upper_thresh, lower = lower_thresh, verbose = False)

				real_far = real_preds[real_idx]
				#if real_event's SNR is -1 (i.e. invalid sample, set FAR to 1)
				real_preds[np.any(real_event[:,:,2] < 0, axis = 1)] = 1.0

				print("Real FAR for NN {} is {} Hz".format(model_idx, real_far))

				if real_preds[real_idx -1] < OPA_threshold or real_preds[real_idx +1] < OPA_threshold:
					print("NOTE: event might have been detected by neighbouring trigger")
					print("FARs for neighbouring triggers: ", real_preds[real_idx -1], real_preds[real_idx +1])
				print("All FARs less than the OPA threshold in this data: ", real_preds[real_preds < OPA_threshold])
				#apply veto condition to real_preds
				if len(real_preds[real_preds < OPA_threshold]) > 0:
					idxs = np.where(real_preds < OPA_threshold)[0]
					ifo_snrs = real_func[idxs]

				if real_far < OPA_threshold or real_preds[real_idx -1] < OPA_threshold or real_preds[real_idx +1] < OPA_threshold:
					if real_preds[real_idx -1] < real_far or real_preds[real_idx +1] < real_far:
						#set the FAR to the lower of the two neighbouring triggers
						#adjust the real_idx accordingly
						real_idx = real_idx -1 if real_preds[real_idx -1] < real_preds[real_idx +1] else real_idx +1
						real_far = real_preds[real_idx]
					#now store the FAR and SNR for this event
					
					if real_far < event_stats[d]['FAR']:
						if event_stats[d]['FAR'] == 1:
							print("First detection for event {}".format(d))
							detection_sum += 1
						else:
							print("Updating event {} FAR from {} to {}".format(d, event_stats[d]['FAR'], real_far))
						event_stats[d]['FAR'] = real_far
						event_stats[d]['SNR'] = real_event[real_idx, 0, 2]
						event_stats[d]['model_bin'] = bin
						event_stats[d]['model_idx'] = model_idx
					# detection_fars[i-8].append(real_far)
					# detection_snrs[i-8].append(real_event[real_idx, 0, 2]) 

				real_idx = inj_args['duration']//2 - 100
		print("Total detections so far: ", detection_sum, " out of ", total_events, "\n")


	#now print the detection summary
	events = get_GWTC_events(m1_lower = 0, m2_lower = 0, m1_upper=1000, m2_upper=1000, exclude_marginal = False)
	for event in event_stats.keys():

		print("Event: ", event)

		gwtc_event = events[event]
		print("Component masses: ", round(gwtc_event['m1'],1), round(gwtc_event['m2'],1))
		print("GPS time: ", gwtc_event['gps'])
		print("LIGO Network SNR: ", gwtc_event['snr'])
		if event_stats[event]['FAR'] == 1:
			print("NOT DETECTED")
			print()
			continue
		print("Pipeline network SNR: ", round(event_stats[event]['SNR'],2))
		print("Model FAR (HZ): ", event_stats[event]['FAR'])
		print("Model FAR (yr-1): ", round(event_stats[event]['FAR']*3600*24*365.25,6))
		print("Model IFAR (yrs): ", round(1/(event_stats[event]['FAR']*3600*24*365.25),2))

		if event_stats[event]['FAR'] > OPA_threshold/6:
			print("NOTE: this event is not detected at the 1/year level")
		if event_stats[event]['FAR'] > OPA_threshold/3:
			print("NOTE: this event is not detected when accounting for trials factor")
		print()


	#finally, save the all_stats to a file for future reference
	results_dir = os.path.join(inj_args['jobdir'], "results")
	np.save(os.path.join(results_dir, "all_stats.npy"), all_stats)