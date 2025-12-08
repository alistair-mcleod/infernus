import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

#sys.path.append("/fred/oz016/alistair/infernus/notebooks/")
#from autoplotter import get_inj_data, get_dsens
from infernus.postprocessing import preds_to_far_constrained, lognorm_fit_constrained_print, pdf_to_cdf_arbitrary, get_inj_data, get_dsens

#sys.path.append("/fred/oz016/alistair/infernus/")

import argparse

#configfile should be a model args json file.
parser = argparse.ArgumentParser()
parser.add_argument('--configfile', type=str, default=None)
args = parser.parse_args()

if args.configfile.split("/")[-1] == "submit.json":
	submit_args = json.load(open(args.configfile))
	inj_args = json.load(open(submit_args['injection_args']))
	bg_args = json.load(open(submit_args['background_args']))
	noise_dir = inj_args["noise_dir"]
	mdc_file = inj_args["injfile"]
	if isinstance(mdc_file, list):
		print("List of injection files detected, only using the first one for results summary for now.")
		inj_file = os.path.join(inj_args['save_dir'], "inj_{}".format(inj_args['bin'][-1]), "timeslides.npy")
		mdc_file = mdc_file[int(inj_args['bin'][-1])]
	else:
		inj_file = os.path.join(inj_args['save_dir'], "timeslides.npy")
	model_val_dir = os.path.join(inj_args['jobdir'], "results", inj_args['bin'])
	os.makedirs(model_val_dir, exist_ok = True)
	print("Model val dir: ", model_val_dir)
	bg_file = os.path.join(bg_args['save_dir'], "timeslides.npy")
	print("TODO: generalise to multiple injection files!")
	real_dir_root = os.path.join(inj_args['jobdir'], "real_events", inj_args['bin'])
	real_dirs = sorted(os.listdir(real_dir_root))
	print("Real dirs: ", real_dirs)


else:

	#handle passing a bg/inj file directly, these files should be in the model_val_dir
	if args.configfile.split("/")[-1] in ["inj.json", "BG.json"]:
		print("Loading BG/INJ file directly, assuming model_val_dir is parent directory")
		model_val_dir = os.path.dirname(args.configfile)
	else:
		model_val_dir = json.load(open(args.configfile))['save_dir']

	print("Model val dir: ", model_val_dir)
	#print("TODO: REVERT LOADING LINE!")
	inj_args = json.load(open(os.path.join(model_val_dir, "inj.json")))
	#inj_args = json.load(open(model_val_dir + "/inj_long.json"))
	noise_dir = inj_args["noise_dir"]
	print("Noise dir is ", noise_dir)
	mdc_file = inj_args["injfile"]

	bg_file = os.path.join(model_val_dir, "BG", "timeslides.npy")
	inj_file = os.path.join(model_val_dir, "inj", "timeslides.npy")
	#inj_file = os.path.join(model_val_dir, "inj_long", "timeslides.npy")
	real_dir_root = os.path.dirname(inj_args['save_dir'])
	real_dirs = sorted(os.listdir(real_dir_root))

print("Loading data from ", bg_file, inj_file)


N_draw, mask, inj_params = get_inj_data(4, noise_dir, mdc_file, pipelines = ["pycbc_hyperbank", "mbta", "gstlal"])

rs_index = 8

bg = np.load(bg_file)
bg = bg.astype(np.float32)
bg = bg.reshape(-1, bg.shape[2], bg.shape[3])
#this check ensures al samples have a network SNR > 0 (i.e. that the sample is valid)
bg = bg[np.all(bg[:,:,2] > 0, axis = 1)]

zerolags = np.load(inj_file)[0] #the 0 is to get rid of the timeslides axis

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

def apply_func_to_bg_inj(bg,inj, func, rs_index = 8):
	return func(bg[:,:,rs_index], axis = 1), func(inj[:,:,rs_index], axis = 1)

bg_sort = []
for i in range(8,bg.shape[2]):
	bg_func, inj_func = apply_func_to_bg_inj(bg, zerolags, np.median, rs_index = i)
	#get rid of nans from the background
	bg_func = bg_func[np.where(np.isfinite(bg_func))]
	#also clip the background at 50
	#bg_func = bg_func[(bg_func < 25)]
	#bg_func = np.nan_to_num(bg_func, nan = -10)

	nn_preds = np.full(mask.sum(), -1000.0)
	nn_preds[inj_params["inj_ids"]] = inj_func[inj_params["zerolags"]]

	#look 1 index either side of inj_params['zerolags'] as the peak might be slightly off
	zl_before = np.array(inj_params['zerolags']) -1
	zl_after = np.array(inj_params['zerolags']) +1
	nn_preds[inj_params["inj_ids"]] = np.maximum(nn_preds[inj_params["inj_ids"]], inj_func[zl_before])
	nn_preds[inj_params["inj_ids"]] = np.maximum(nn_preds[inj_params["inj_ids"]], inj_func[zl_after])

	nn_preds = np.nan_to_num(nn_preds, nan = 0)
	preds = preds_to_far_constrained(bg_func, nn_preds, upper = upper_thresh, lower = lower_thresh)
	#preds = np.nan_to_num(preds, nan = 0)
	
	pipeline_fars['NN'+str(i)] = preds
	if 'NN'+str(i) not in pipelines:
		pipelines.append('NN'+str(i))

	pipeline_fars['NN'+str(i)][pipeline_fars['NN'+str(i)] == 1] = np.inf
	#find the 10th highest background value
	bg_func = np.sort(bg_func)
	bg_sort.append(bg_func)
	bg_max = bg_func[-10]
	print("events with prediction higher than background (note: 10 highest BG points are removed):", (nn_preds > bg_max).sum())
	plt.hist(bg_func, bins = 100, histtype = 'step', label = "BG")
	plt.hist(nn_preds, bins = 100, histtype = 'step', label = "Inj")
	plt.yscale('log')
	plt.xlabel("Ranking statistic")
	plt.ylabel("Counts")

	#save the histogram to a file
	plt.savefig(os.path.join(model_val_dir, "NN{}_hist.png".format(i)),dpi = 300)
	plt.clf()
	print()


	#also plot the cumulative histogram with the extrapolation
	maxval = max(np.max(nn_preds), np.max(bg_func)) + 20
	minval = -100 
	#f = plt.figure(figsize=(3.5,3))
	space = np.linspace(minval, maxval, 10000)

	plt.hist(bg_func[:-10], bins = 1000, alpha = 0.8, density=True, cumulative=-1, histtype='step', label = "Background")
	mean, std = lognorm_fit_constrained_print(bg_func[:-10], upper = upper_thresh, lower = lower_thresh, verbose=False)
	print(mean,std)
	full_constrained = pdf_to_cdf_arbitrary(norm.pdf(space, mean, std))
	lim = np.argmin(np.abs(full_constrained - upper_thresh))

	plt.plot(space[lim:], full_constrained[lim:], label = "Extrapolation", linestyle = "--", linewidth = 1)
	plt.yscale('log')
	plt.ylim(1e-15, 2)
	plt.xlim(-40, )
	plt.xlabel("Ranking statistic")
	plt.ylabel("False alarm rate (Hz)")
	plt.legend(loc='upper right')
	plt.title("NN{}".format(i))
	plt.savefig(os.path.join(model_val_dir, "NN{}_hist_fit.png".format(i)),dpi = 300)
	plt.clf()


print("")

found = {}

for p in pipelines:
	found[p] = (pipeline_fars[p] < OPA_threshold)

print("Model counts:")
for i in range(8,bg.shape[2]):
	print("NN{}: ".format(i), found['NN'+str(i)].sum())

print("\npycbc, mbta, gstlal:")
print(found['pycbc_hyperbank'].sum(), found['mbta'].sum(), found['gstlal'].sum())

print("PyCBC unique events: ", (found['pycbc_hyperbank'] & ~found['mbta'] & ~found['gstlal']).sum())
print("MBTA unique events: ", (~found['pycbc_hyperbank'] & found['mbta'] & ~found['gstlal']).sum())
print("GSTLAL unique events: ", (~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal']).sum())
print("")


for i in range(8,bg.shape[2]):
	print("unique events in NN{}: ".format(i), (found['NN'+str(i)] & ~found['pycbc_hyperbank'] \
											 & ~found['mbta'] & ~found['gstlal']).sum())
	
print("")
for i in range(8,bg.shape[2]):
	print("NN{} duo detection with PyCBC: ".format(i), (found['NN'+str(i)] & found['pycbc_hyperbank'] & ~found['mbta'] & ~found['gstlal']).sum())
	print("NN{} duo detection with MBTA: ".format(i), (found['NN'+str(i)] & ~found['pycbc_hyperbank'] & found['mbta'] & ~found['gstlal']).sum())
	print("NN{} duo detection with GSTLAL: ".format(i), (found['NN'+str(i)] & ~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal']).sum())

print("")

for i in range(8,bg.shape[2]):
	print("unique PyCBC events AFTER adding NN{}: ".format(i), (found['pycbc_hyperbank'] & ~found['NN'+str(i)] & ~found['mbta'] & ~found['gstlal']).sum())
	print("unique MBTA events AFTER adding NN{}: ".format(i), (~found['pycbc_hyperbank'] & found['mbta'] & ~found['NN'+str(i)] & ~found['gstlal']).sum())
	print("unique GSTLAL events AFTER adding NN{}: ".format(i), (~found['pycbc_hyperbank'] & ~found['mbta'] & found['gstlal'] & ~found['NN'+str(i)]).sum())


from pycbc.sensitivity import volume_to_distance_with_errors
from astropy.cosmology import FlatwCDM
cosmo = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)


M1= np.array([40, 30, 20, 10])
M2 = np.array([1.4, 1.4, 1.4, 1.4])

#M1= np.array([2, 2, 1.4, 1.4])
#M2 = np.array([2, 1.4, 1.4, 1])

m2_full_pop = True
m1_full_pop = True

if "bns" in mdc_file:
	m1_full_pop = True
	m2_full_pop = True
	print("BNS, using full population")

elif "nsbh" in mdc_file:
	m1_full_pop = False
	m2_full_pop = True
	M1 = np.array([40, 30, 20, 10])
	M2 = np.array([1.4, 1.4, 1.4, 1.4])

elif "bbh" in mdc_file:
	print("BBH, using fixed masses")
	m1_full_pop = False
	m2_full_pop = False
	M1 = np.array([40, 30, 20, 10])
	M2 = np.array([40, 30, 20, 10])

fars = np.geomspace(1e-3, 1e-12, 50)

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
	best = 8
	#opa_idx = np.argmin(np.abs(fars - OPA_threshold))
	for j in range(8, bg.shape[2]):
		if np.sum(pipeline_fars['NN'+str(j)] < OPA_threshold) > np.sum(pipeline_fars['NN'+str(best)] < OPA_threshold):
			best = j
	print("Best NN is NN{}".format(best))
	#remove all NNs except the best one
	pipelines = [p for p in pipelines if not p.startswith('NN') or p == 'NN'+str(best)]
	print("Pipelines after removing NNs: ", pipelines)	
	VT, sigma_VT = get_dsens(z, p_draw, N_draw, pipelines, pipeline_fars,m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, pop_params = pop_params)
	
	for p in pipelines:
		#if p == "NN8" or p == "NN9":
		#	continue
		vt = np.array(VT[p])
		sigma = np.array(sigma_VT[p])
		dist, ehigh, elow = volume_to_distance_with_errors(vt*1e9, sigma*1e9)
		#if p == "ATLAS":
		#	
		#	ax.plot(fars, dist, label=p, linewidth=1, color = pipeline_colours[p], zorder = 10)
		#	ax.fill_between(fars, dist-elow, dist+ehigh, alpha=0.5, color = pipeline_colours[p], ec = 'none', zorder = 10)

			#sneaky extra plot

		ax.plot(fars, dist, label=p, linewidth=1, alpha = 0.7, zorder = 6)
		ax.fill_between(fars, dist-elow, dist+ehigh, alpha=0.3, ec = 'none', zorder = 6)

		#else:
		#	ax.plot(fars, dist, label=p, linewidth=1, alpha = 0.7, color = pipeline_colours[p], zorder = 6)
		#	ax.fill_between(fars, dist-elow, dist+ehigh, alpha=0.3, color = pipeline_colours[p], ec = 'none', zorder = 6)
		
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

plt.tight_layout()


#plt.xlabel('False alarm rate (Hz)')
#plt.ylabel("Sensitive distance (Mpc)")

plt.show()
#save the plot to the parent directory of bg_file

#import os
#save_path = os.path.dirname(os.path.dirname(bg_file))

#plt.savefig(os.path.join(save_path, "sens_vs_far.png"))

plt.savefig(os.path.join(model_val_dir, "sens_vs_far.png"), dpi = 300)

print("Done! \n")






print("Now looking at real events! Note: real event FARs are computed using the O3 week 3 background, so aren't necessarily accurate")

#we can use the inj args for most stuff
#TODO: add start cutoff (100 s) to args file
real_idx = inj_args['duration']//2 - 100
print("Real event should be {} s into the data".format(real_idx))

detection_sum = np.zeros(shape = bg.shape[2]-8)
detection_list = [[] for _ in range(bg.shape[2]-8)]
detection_fars = [[] for _ in range(bg.shape[2]-8)]
detection_snrs = [[] for _ in range(bg.shape[2]-8)]
total_events = 0

for d in real_dirs:
	#print(d)
	if d.startswith("GW"):
		total_events += 1
		#print(d)
		real_dir = os.path.join(real_dir_root, d)
		
		print("Processing event", d)

		real_event = np.load(os.path.join(real_dir, "timeslides.npy"))[0]

		for i in range(8, real_event.shape[2]):
			#note: the two below lines are super slow. We should have a sorted BG for each model,
			#bg_func, real_func = apply_func_to_bg_inj(bg, real_event, np.median, rs_index = i)
			#real_preds = preds_to_far_constrained(bg_func, real_func, upper = upper_thresh, lower = lower_thresh, verbose = False)
			_, real_func = apply_func_to_bg_inj(bg[:2], real_event, np.median, rs_index = i)
			real_preds = preds_to_far_constrained(bg_sort[i-8], real_func, upper = upper_thresh, lower = lower_thresh, verbose = False)

			real_far = real_preds[real_idx]
			#if real_event's SNR is -1 (i.e. invalid sample, set FAR to 1)
			real_preds[np.any(real_event[:,:,2] < 0, axis = 1)] = 1.0
			#print("Set invalid FARs to 1.0")
			print("Real FAR for NN {} is {} Hz".format(i, real_far))
			if real_preds[real_idx -1] < OPA_threshold or real_preds[real_idx +1] < OPA_threshold:
				print("NOTE: event might have been detected by neighbouring trigger")
				print("FARs for neighbouring triggers: ", real_preds[real_idx -1], real_preds[real_idx +1])
			print("All FARs less than the OPA threshold in this data: ", real_preds[real_preds < OPA_threshold])
			#apply veto condition to real_preds
			if len(real_preds[real_preds < OPA_threshold]) > 0:
				idxs = np.where(real_preds < OPA_threshold)[0]
				ifo_snrs = real_func[idxs]
			if real_far < OPA_threshold or real_preds[real_idx -1] < OPA_threshold or real_preds[real_idx +1] < OPA_threshold:
				detection_sum[i-8] += 1
				detection_list[i-8].append(d)
				if real_preds[real_idx -1] < real_far or real_preds[real_idx +1] < real_far:
					#set the FAR to the lower of the two neighbouring triggers
					real_far = min(real_preds[real_idx -1], real_preds[real_idx +1])
				detection_fars[i-8].append(real_far)
				detection_snrs[i-8].append(real_event[real_idx, 0, 2]) 
		print("Done with event", real_dir)
		print()

print("Number of detections for each model:")
for i in range(8, bg.shape[2]):
	print("NN{}: ".format(i), detection_sum[i-8])

print("Total number of events:", total_events)

from infernus.real_utils import get_GWTC_events
events = get_GWTC_events(m1_lower = 0, m2_lower = 0, m1_upper=1000, m2_upper=1000, exclude_marginal = False)
for i in range(8, bg.shape[2]):
	print("\nDetections for NN{}:".format(i))
	for event in detection_list[i-8]:
		#print event info from event_list
		event_info = events[event]
		print("Summary for event ", event)
		print("Component masses: ", round(event_info['m1'],1), round(event_info['m2'],1))
		print("GPS time: ", event_info['gps'])
		print("Network SNR: ", event_info['snr'])
		print("Pipeline network SNR: ", detection_snrs[i-8][detection_list[i-8].index(event)])
		print("Model FAR (HZ): ", detection_fars[i-8][detection_list[i-8].index(event)])
		print("Model IFAR (yrs): ", round(1 /(detection_fars[i-8][detection_list[i-8].index(event)]*(3600 * 24 * 365.25)),2) )
		if detection_fars[i-8][detection_list[i-8].index(event)] > OPA_threshold/6:
			print("NOTE: this event is not detected at the 1/year level")
		print()



#New sens vs far plotting: one plot per model!!!!

for m in range(8,bg.shape[2]):
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
		#best = 8
		#opa_idx = np.argmin(np.abs(fars - OPA_threshold))
		#for j in range(8, bg.shape[2]):
		#	if np.sum(pipeline_fars['NN'+str(j)] < OPA_threshold) > np.sum(pipeline_fars['NN'+str(best)] < OPA_threshold):
		#		best = j
		pipelines = ['pycbc_hyperbank', 'mbta', 'gstlal']
		pipelines.append("NN{}".format(m))
		#print("Best NN is NN{}".format(best))
		
		#remove all NNs except the best one
		#pipelines = [p for p in pipelines if not p.startswith('NN') or p == 'NN'+str(best)]
		print("Pipelines for this plot: ", pipelines)	
		VT, sigma_VT = get_dsens(z, p_draw, N_draw, pipelines, pipeline_fars,m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, pop_params = pop_params)
		
		for p in pipelines:
			#if p == "NN8" or p == "NN9":
			#	continue
			vt = np.array(VT[p])
			sigma = np.array(sigma_VT[p])
			dist, ehigh, elow = volume_to_distance_with_errors(vt*1e9, sigma*1e9)
			#if p == "ATLAS":
			#	
			#	ax.plot(fars, dist, label=p, linewidth=1, color = pipeline_colours[p], zorder = 10)
			#	ax.fill_between(fars, dist-elow, dist+ehigh, alpha=0.5, color = pipeline_colours[p], ec = 'none', zorder = 10)

				#sneaky extra plot

			ax.plot(fars, dist, label=p, linewidth=1, alpha = 0.7, zorder = 6)
			ax.fill_between(fars, dist-elow, dist+ehigh, alpha=0.3, ec = 'none', zorder = 6)

			#else:
			#	ax.plot(fars, dist, label=p, linewidth=1, alpha = 0.7, color = pipeline_colours[p], zorder = 6)
			#	ax.fill_between(fars, dist-elow, dist+ehigh, alpha=0.3, color = pipeline_colours[p], ec = 'none', zorder = 6)
			
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

	plt.tight_layout()

	plt.savefig(os.path.join(model_val_dir, "sens_vs_far_{}.png".format(m)), dpi = 300)
	plt.clf()