import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import argparse
import gc
import json
import h5py
from GWSamplegen.noise_utils import load_psd, get_valid_noise_times, load_gps_blacklist, get_valid_noise_times_from_segments, get_data_from_OzStar
from GWSamplegen.waveform_utils import load_pycbc_templates, chirp_mass, select_approximant, t_at_f, maximum_f_lower
from GWSamplegen.snr_utils_np import np_get_cutoff_indices, mf_in_place, np_sigmasq, numpy_matched_filter
from GWSamplegen.chisq_utils_np import reduced_chisquared_precomputed_SNR
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import TimeSeries
from pycbc.filter import highpass
from pycbc.detector import Detector
import sys

import multiprocessing as mp
import ctypes


#change dir

#os.chdir('/fred/oz016/alistair/infernus/infernus')

from infernus.noise_utils import noise_generator, noise_fetcher
from infernus.SNR_utils import make_windows_2d
from infernus.real_utils import get_real_events#, get_data_from_OzStar
#from model_utils import onnx_callback
start = time.time()
from infernus.triggering.zerolags import get_zerolags

parser = argparse.ArgumentParser()
parser.add_argument('--jobindex', type=int,default=0)
parser.add_argument('--workerid', type=int, default=0)
parser.add_argument('--totalworkers', type=int, default=1)
parser.add_argument('--totaljobs', type=int, default=1)
parser.add_argument('--argsfile', type=str, default=None)
parser.add_argument('--streamline', type = int, default = None)
parser.add_argument('--injindex', type = int, default = -1)

args = parser.parse_args()

job_id = args.jobindex #job id in job array
#worker_id = args.workerid
n_workers = args.totalworkers
n_jobs = 1
argsfile = args.argsfile
print(argsfile)

if args.streamline == 1:
	print("Streamline mode!")
	streamline = True
else:
	streamline = False

inj_index = None
if args.injindex >= 0:
	inj_index = args.injindex
	print("Using injection file index", inj_index)

print("I'm job number", job_id)



def load_pycbc_templates_from_hdf(hdf_file):
	f = h5py.File(hdf_file, 'r')
	templates = np.zeros((len(f['mass1']),6))
	templates[:,1] = f['mass1'][()]
	templates[:,2] = f['mass2'][()]
	templates[:,3] = f['spin1z'][()]
	templates[:,4] = f['spin2z'][()]
	templates[:,5] = f['f_lower'][()]
	templates[:,0] = chirp_mass(templates[:,1], templates[:,2])
	
	#TODO: generalise this!!!
	templates = templates[templates[:,0].argsort()]

	#nsbh = ((templates[:,1] > 3) & (templates[:,2] < 3.75) & (templates[:,1] < 100))
	#the 3.125 is chosen as max m2 as 2.5 * 1.25 is the maximum m2 detector frame mass in GWTC-3
	#print("number of templates in selected region:", np.sum(nsbh))
	return templates #[nsbh]


args = json.load(open(argsfile, "r"))
noise_dir = args["noise_dir"]
n_cpus = args["ntasks"]
print("number of CPUs:", n_cpus)


maxnoisesegs = args["max_noise_segments"]


try:
	template_bank_dir = args["template_bank_dir"]
	template_bank_name = args["template_bank_name"]

	templates, _, _= load_pycbc_templates(template_bank_name, template_bank_dir)
	print("loading templates from text file")
except:
	template_bank = args["template_bank"]
	templates = load_pycbc_templates_from_hdf(template_bank)
	print("number of templates:", len(templates))
	
	
cut = None
if "template_mass1_min" in args:
	print("cutting templates")
	cut = (templates[:,1] > args["template_mass1_min"]) 
if "template_mass1_max" in args:
	cut = cut & (templates[:,1] < args["template_mass1_max"])
if "template_mass2_min" in args:
	cut = cut & (templates[:,2] > args["template_mass2_min"])
if "template_mass2_max" in args:
	cut = cut & (templates[:,2] < args["template_mass2_max"])
if "template_chirp_mass_min" in args:
	#note there might or might not be a cut on component mass
	print("cutting templates by chirp mass")
	print("note: this overrides component mass cuts if they exist")
	cut = (templates[:,0] > args["template_chirp_mass_min"])
if "template_chirp_mass_max" in args:
	cut = cut & (templates[:,0] < args["template_chirp_mass_max"])
if cut is not None:
	templates = templates[cut]

print("number of templates after cut:", len(templates))


#injfile should have one of the following formats:
#None: perform a background run, with timeslides equal to num_time_slides
#a path to an HDF file: perform an injection run using the injections in the file
#A list of paths to HDF files: perform multiple injection runs, one for each file.
#real: perform a search for a specific event, using real event data
#noninj: perform a search over the specified time period 

#TODO: add either a flag or another injfile format to prevent including times with real events

duration = args["duration"]
sample_rate = args["sample_rate"]
f_lower = args["f_lower"]
fd_approximant = args["fd_approximant"]
td_approximant = args["td_approximant"]
injfile = args["injfile"]
save_dir = args["save_dir"]
num_time_slides = args["n_timeslides"]
num_triggers = args["num_triggers"]
if streamline:
	myfolder = save_dir
else:
	myfolder = os.environ['JOBFS']
	os.makedirs(myfolder, exist_ok=True)
print("saving to", myfolder)

try:
	chisq = args["chisq"]
	n_chisq_bins = args["n_chisq_bins"]
	if chisq:
		print("using reduced chi-squared with {} bins".format(n_chisq_bins))

except:
	chisq = False

try:
	inference_rate = args["inference_rate"]
	print("inference rate is", inference_rate)
except:
	print("no inference rate specified, using default of 16 Hz")

try:
	both_detectors_above_thresh = args["both_detectors_above_thresh"]
except:
	both_detectors_above_thresh = False
print("both detectors above threshold flag set to", both_detectors_above_thresh)

if "bin" in args:
	bin = args["bin"]
	print("using bin:", bin)
else:
	bin = None

try:
	injection_SNR_dir = args["injection_SNR_dir"]
	print("using injection SNR directory:", injection_SNR_dir, ". Make sure the SNR files match this run's configuration.")
except:
	injection_SNR_dir = None

if injection_SNR_dir is not None and injfile not in [None, "noninj", "real"]:
	#we're doing an injection run with precomputed SNRs. Check if the SNRs exist first
	#injection_SNR_dir = os.path.join(injection_SNR_dir, bin)
	if inj_index is not None:
		injection_SNR_dir = os.path.join(injection_SNR_dir, "inj_{}".format(inj_index))
		print("using injection SNR directory:", injection_SNR_dir)
	if not os.path.exists(os.path.join(injection_SNR_dir, "timeslides_{}_0.npy".format(job_id))) or not os.path.exists(os.path.join(injection_SNR_dir, "SNR_array_{}_0.npy".format(job_id))):
		print("injection SNR files not found, we will need to generate them.")

		#generate the directory just in case
		os.makedirs(injection_SNR_dir, exist_ok=True)
		myfolder = injection_SNR_dir

	else:
		print("injection SNR files found, we will use them.")
		sys.exit(0)


if injfile == "None":
	injfile = None
	print("starting background run")
else:
	#we set this to 1 so we can keep the same structure
	num_time_slides = 1
	if injfile == "noninj":
		print("noninj/search")
	else:
		if inj_index is not None:
			injfile = injfile[inj_index]
		print("injecting events from file", injfile)

if isinstance(noise_dir, list) or injfile == "real":
	print("Using new noise loading")
	new_noise = True
else:
	new_noise = False


print(noise_dir)
delta_t = 1/sample_rate
f_final = sample_rate//2
delta_f = 1/duration


N = int(duration/delta_t)
kmin, kmax = np_get_cutoff_indices(f_lower, None, delta_f, N)

ifos = ["H1", "L1"]

#TODO: we could do PSD per-segment with our new BG method

if injfile != "real" and not new_noise:
	psds = load_psd(noise_dir, duration,ifos , f_lower, int(1/delta_t))
	for psd in psds:
		psds[psd] = psds[psd][kmin:kmax]


#TODO: FIX THIS, templates are normally only loaded in frequency domain. Maybe just add start chop time as a parameter?
#hp, _ = get_td_waveform(mass1 = templates[0,1], mass2 = templates[0,2], 
#						delta_t = delta_t, f_lower = f_lower, approximant = td_approximant)

#max_waveform_length = len(hp)/sample_rate
#max_waveform_length = max(32, int(np.ceil(max_waveform_length/10)*10))
max_waveform_length = 100
print("Forcing max_waveform_length to be 100 seconds")


template_start = 0
#templates_per_batch = 30 #TODO: check different sizes.

window_size = 2048
stride = window_size//inference_rate



start_cutoff = max_waveform_length
end_cutoff = duration - 24 #set so total length is a nice number
slice_duration = (end_cutoff - start_cutoff)
#n_windows = (slice_duration*sample_rate - window_size)//stride +1

if new_noise:
	valid_times = get_valid_noise_times_from_segments(noise_dir, duration, slice_duration, blacklisting = False)
else:
	valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, slice_duration, blacklisting = False)

segment = job_id
if segment > 0 and (valid_times[segment] - valid_times[segment-1]) < slice_duration and injfile != "real":
	chop_idx = int((slice_duration - (valid_times[segment] - valid_times[segment-1])) * sample_rate)
	chop_time = chop_idx // sample_rate
	print("short segment, only keeping beyond second:", chop_time)
else:
	chop_time = 0
	chop_idx = 0
#if valid_times[segment] - valid_times[segment-1] < slice_duration:
#	print("This job has a shorter segment...")


if injfile == "real":
	print("Searching for a real event from GWTC!")
	print("fetching GWTC events...")
	#events = get_real_events(ifos = ifos, m1_lower = args["template_mass1_min"], m2_lower = args["template_mass2_min"], \
	#					  				m1_upper = args["template_mass1_max"], m2_upper = args["template_mass2_max"], exclude_marginal = False, padding = duration)
	events = get_real_events(ifos = ifos, m1_lower = 1, m2_lower = 1, \
										m1_upper = 1000, m2_upper = 1000, exclude_marginal = False, padding = duration)
	event = list(events.keys())[segment]
	print("Since I'm job number", job_id, "I'm looking for event", event)
	print("Total number of events: ", len(list(events.keys())))
	psds = {}
	for ifo in ifos:
		psds[ifo] = get_data_from_OzStar(events[event]['gps']-duration/2, duration, ifo)
		psds[ifo] = psds[ifo][:len(psds[ifo])//2 - 10 * sample_rate].psd(4).astype('complex128')
		psds[ifo] = interpolate(psds[ifo], delta_f)
		psds[ifo] = inverse_spectrum_truncation(psds[ifo], int(4*sample_rate), low_frequency_cutoff=f_lower)
		psds[ifo] = psds[ifo][kmin:kmax]

if new_noise and injfile != "real":
	print("fetching PSD for new noise method")
	print("NOTE: we should test the best way to produce a PSD for a segment: using the discarded segment data, or the prior segment, or the current segment?")
	print("For now, using the chopped data")
	psds = {}
	for ifo in ifos:
		psds[ifo] = get_data_from_OzStar(valid_times[segment], duration, ifo)
		psds[ifo] = psds[ifo][:100 * sample_rate].psd(4).astype('complex128')
		psds[ifo] = interpolate(psds[ifo], delta_f)
		psds[ifo] = inverse_spectrum_truncation(psds[ifo], int(4*sample_rate), low_frequency_cutoff=f_lower)
		psds[ifo] = psds[ifo][kmin:kmax]



gps_blacklist = load_gps_blacklist(f_lower)
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

#TODO: USE deleted zerolags to get rid of real events!






all_detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

if injfile is not None and injfile != "noninj" and injfile != "real":
	print("using injection file", injfile)
	f = h5py.File(injfile, 'r')
	mask = (f['injections']['gps_time'][:] > valid_times[segment]) & (f['injections']['gps_time'][:] < valid_times[segment] + duration)
	n_injs = np.sum(mask)
	print("number of injections in this segment:", n_injs)

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
	if "eccentricity" in f['injections']:
		eccentricity = f['injections']['eccentricity'][mask]
	else:
		eccentricity = np.zeros(n_injs)

	#This is necessary for migration to numpy V2, as float32 runs into an overflow
	mass1 = np.array(mass1, dtype = np.float64)
	mass2 = np.array(mass2, dtype = np.float64)
	startgps = []
	for i in range(n_injs):
		startgps.append(np.floor(gps[i] - t_at_f(mass1[i], mass2[i], f_lower)))

	startgps = np.array(startgps)

	hgps = gps + all_detectors['H1'].time_delay_from_earth_center(right_ascension, declination, gps)
	lgps = gps + all_detectors['L1'].time_delay_from_earth_center(right_ascension, declination, gps)
	gps_dict = {'H1': hgps, 'L1': lgps}
	print("GPS times for injections should be correct now (injected from geocentre)")
	#lgps = gps + all_detectors['L1'].time_delay_from_detector(all_detectors['H1'], 
	#											right_ascension, 
	#											declination, 
	#											gps)

	#gps_dict = {'H1': gps, 'L1': lgps}




template_time = mp.Value('d', 0)
mf_time = mp.Value('d', 0)
strain_time = mp.Value('d', 0)
timeslide_time = mp.Value('d', 0)

criterion_time = mp.Value('d', 0)
merge_time = mp.Value('d', 0)







#n_templates = templates_per_batch
#t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)

#n_batches = 


windowed_sample_end_indexes = list(range(sample_rate-1, slice_duration*sample_rate, sample_rate//inference_rate))
windowed_sample_start_indexes = list(np.copy(windowed_sample_end_indexes) - (sample_rate - 1))
start_end_indexes = list(zip(windowed_sample_start_indexes, windowed_sample_end_indexes))

#global template_start_idx, timeslides, batch

def is_timeslide_valid(second,timeslide, chop_time = 0):
	return (timeslide != (second + chop_time)) & (timeslide != ((second + chop_time) - 1))


#global criterion_time, merge_time

#criterion_time = 0
#merge_time = 0


def get_timeslide_new(i, SNR_rolled, template_ids):
	global criterion_time, merge_time
	#Get the ith timeslide. This processes all 900 seconds
	#for each timeslide, roll the SNR array. get 'zerolags', then compare to existing array
	#if i == -1:
	#	return
	#print("timeslide",i)
	
	
	#SNR_rolled = np.copy(nonwindowed_SNR)
	#if injfile is None:
	#	#roll SNR array to perform a time shift
	#	SNR_rolled[1,:,:] = np.roll(nonwindowed_SNR[1,:,:], 2048*(i+1), axis = 1)
	#if we're using an injfile, no time shift is necessary.

	zerolags = get_zerolags(
		data = SNR_rolled,
		snr_thresh = 4,
		offset = 20,
		buffer_length = 2048,
		overlap = int(0.2*2048),
		num_trigs = 1,
		chop_time = chop_idx,
		both_detectors_above_thresh = both_detectors_above_thresh
	)

	zerolags = np.array(zerolags)
	#TODO: we're going to use the template IDs to save the SNRs, need to move this after it's already integrated into timeslides.
	#zerolags[:,:,5] += template_start_idx

	#TODO: this needs timeslides to be -1 to 100.
	if injfile is None:
		zerolags[:,0,4] = (zerolags[:,0,4] - 2048 * (i+1)) % (900 * 2048)
	#print("batch is:", batch)

	#else:
	#print(batch)

	#TODO: overhaul to handle multiple triggers
	h_start = np.floor(np.maximum(zerolags[:,0,3] //stride - sample_rate//stride +1, 0)) * stride
	h_end = np.floor(np.minimum(zerolags[:,0,3] //stride + 1, len(start_end_indexes)-1)) * stride + 2048

	#overwrite_criterion = (zerolags[:,0,2] > timeslides[i,:,0,2]) & \
	#	(h_end.astype('int') - h_start.astype('int') == 4096) & \
	#	is_timeslide_valid(np.arange(timeslides.shape[1]), i, chop_time)
	
	#timeslides[i, overwrite_criterion,:, :6] = zerolags[overwrite_criterion]

	#timeslides[i, overwrite_criterion,0, 6] = h_start[overwrite_criterion]
	#timeslides[i, overwrite_criterion,0, 7] = h_end[overwrite_criterion]
	st = time.time()

	#now we need to aquire the lock on the timeslides array

	#we get the lock of the shared array, rather than the numpy array itself
	with shared_array_base.get_lock():

		if injfile is None:
			overwrite_criterion = (zerolags[:,0,2] > np.min(timeslides[i,:,:,2], axis = 1)) & \
				(h_end.astype('int') - h_start.astype('int') == 4096) & \
				is_timeslide_valid(np.arange(timeslides.shape[1]), i, chop_time)
		
		else:
			overwrite_criterion = (zerolags[:,0,2] > np.min(timeslides[i,:,:,2], axis = 1)) & \
			(h_end.astype('int') - h_start.astype('int') == 4096)
			
		overwrite_targets = np.argmin(timeslides[i,:,:,2][overwrite_criterion],axis = 1)

		timeslides[i, overwrite_criterion, overwrite_targets,:6] = zerolags[overwrite_criterion].squeeze()
		timeslides[i, overwrite_criterion, overwrite_targets, 6] = h_start[overwrite_criterion]
		timeslides[i, overwrite_criterion, overwrite_targets, 7] = h_end[overwrite_criterion]


		
		overwrite_criterion = overwrite_criterion.ravel()
		#next, overwrite the SNR array
		with criterion_time.get_lock():
			criterion_time.value += time.time() - st
		#criterion_time += time.time() - st
		
		#overwrite_criterion is the length of the segment
		tmplt = zerolags[overwrite_criterion,0,5].astype('int')
		start = h_start[overwrite_criterion].astype('int')
		end = h_end[overwrite_criterion].astype('int')
		st = time.time()
		idx = 0
		for j in range(timeslides.shape[1]):
			
			if overwrite_criterion[j]:
				
				SNR_array[:,i,j,overwrite_targets[idx]] = SNR_rolled[:,tmplt[idx], start[idx]:end[idx]]
				idx += 1
		with merge_time.get_lock():
			merge_time.value += time.time() - st
		#merge_time += time.time() - st
		#SNR_array[:,i,overwrite_criterion,overwrite_targets] = SNR_rolled[:,timeslides[i,j,0,5].astype('int'), timeslides[i,j,0,6].astype('int'):timeslides[i,j,0,7].astype('int')]

		"""
		for j in range(timeslides.shape[1]):
			
			if overwrite_criterion[j]:
				
				SNR_array[:,i,j,0] = SNR_rolled[:,timeslides[i,j,0,5].astype('int'), timeslides[i,j,0,6].astype('int'):timeslides[i,j,0,7].astype('int')]

		"""
		#timeslides[i, overwrite_criterion,:, 5] += template_start_idx
		#timeslides[i, overwrite_criterion,overwrite_targets, 5] += template_start_idx
		timeslides[i, overwrite_criterion,overwrite_targets, 5] = np.array(template_ids)[timeslides[i, overwrite_criterion,overwrite_targets, 5].astype('int')]

		#timeslides 0-6 are zerolag info, 6 is the start idx of the SNR time series, 7 is the end idx of the SNR time series. 8 is the response array.
		#really we don't need to store col. 2 (the network SNR), or columns 6 and 7 in the long run.
		#timeslides[i, :,0, 9] = valid_times[segment] + np.arange(timeslides.shape[1])
		#rather than storing the GPS time (which doesn't fit in a float32), we can store the segment ID and the number of seconds into the segment
		timeslides[i, :, overwrite_targets, 3] = segment
		timeslides[i, :, overwrite_targets, 4] = np.arange(timeslides.shape[1]) + chop_time + start_cutoff


def load_templates(params):
	cm, m1, m2, s1z, s2z = params
	temp_fd_approximant = select_approximant(m1, m2, fd_approximant, domain='frequency')
	return get_fd_waveform(mass1 = m1, mass2 = m2,
						   spin1z = s1z, spin2z = s2z,
						   approximant = temp_fd_approximant, f_lower = f_lower,
						   delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]



#template_time = 0
#mf_time = 0
#strain_time = 0
#timeslide_time = 0

#criterion_time = 0
#merge_time = 0

#make the timing tests shared values



# Step 2: Create a shared memory object
shared_array_base = mp.Array(ctypes.c_float, num_time_slides* (900 - int(chop_time)) * num_triggers * 10, lock = True)  # Adjust size as needed

# Step 3: Create a numpy array that points to the shared memory
#timeslides = np.ctypeslib.as_array(shared_array_base.get_obj())
timeslides = np.frombuffer(shared_array_base.get_obj(),dtype=np.float32)
timeslides = timeslides.reshape(num_time_slides, 900 - int(chop_time), num_triggers, 10)  # Adjust shape as needed

#timeslides = np.frombuffer(shared_array_base.get_obj(),dtype=n).reshape(num_time_slides, 900 - int(chop_time), num_triggers, 10)

#timeslides is n_timeslides, n_seconds, n_triggers, 8
#timeslides = np.zeros((num_time_slides, 900 - int(chop_time), num_triggers, 10), dtype=np.float32)

with shared_array_base.get_lock():
	timeslides[:] = -1

SNR_array_base = mp.Array(ctypes.c_float, 2*num_time_slides*(900 - int(chop_time)) * num_triggers *4096, lock = True)
#SNR_array = np.ctypeslib.as_array(SNR_array_base.get_obj())
SNR_array = np.frombuffer(SNR_array_base.get_obj(),dtype=np.float32)
#the 1 is in case we end up saving multiple SNR time series per second
SNR_array = SNR_array.reshape(2, num_time_slides, 900 - int(chop_time), num_triggers, 4096)

#create SNR_array as a memmap

#SNR_array = np.memmap(myfolder + "/SNR_array_{}_{}.npy".format(job_id,worker_id), 
#					  shape=(2, num_time_slides, 900 - int(chop_time), num_triggers, 4096),
#					  dtype=np.float32, mode='w+', offset=128)

#SNR_array = np.zeros((2, 100, 900, 1, 4096), dtype=np.float32)
#SNR_array[:] = -1000

#templates, _, _= load_pycbc_templates(template_bank_name, template_bank_dir)


batch = 0

templates_per_batch = 20 #TODO: check different sizes.
n_templates = templates_per_batch

#if injfile is not None:
#if injfile == 'real':
#	print("temp shortening job.")
#	templates = templates[:templates_per_batch * n_workers * 20]
# templates = templates[:templates_per_batch * n_workers * 10]
# print('temp shortening job')

loop = int(np.ceil(len(templates)/templates_per_batch))
template_banks = []
for i in range(len(templates)):
	if len(template_banks) < i%loop +1:
		template_banks.append([])
	
	template_banks[i%loop].append(i)
#TODO: tweak template shuffling: should make all batches but the last the same number of templates.

#templates_per_job = int(np.ceil(len(templates)/n_workers))
templates_per_job = len(templates)
main_job_templates = templates_per_job
print("templates per job", templates_per_job)
#total_lastjob = len(templates) - templates_per_job * (n_workers-1)
#template_start = worker_id * templates_per_job
template_start = 0
#if worker_id == n_workers - 1:
#	templates_per_job = total_lastjob

print("starting from template", template_start, flush=True)

n_batches = int(np.ceil(templates_per_job/templates_per_batch))
print("n_batches", n_batches)

t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)
#snrs = []
#only need one noise segment

if injfile != "real" and not new_noise:
	noise = noise_fetcher(job_id, valid_times, paths, file_list, duration, sample_rate)
else:
	if injfile == "real":
		print("Fetching real event. Note: make sure GW190814 works (H1 does not have GWOSC data)")
		for i in range(len(ifos)):
			if i == 0:
				noise = get_data_from_OzStar(int(events[event]['gps']-duration/2), duration, ifos[i])
			else:
				noise = np.vstack((noise, get_data_from_OzStar(int(events[event]['gps']-duration/2), duration, ifos[i]))) 
	else:
		print("Fetching noise with new method")
		for i in range(len(ifos)):
			if i == 0:
				noise = get_data_from_OzStar(valid_times[segment], duration, ifos[i])
			else:
				noise = np.vstack((noise, get_data_from_OzStar(valid_times[segment], duration, ifos[i])))

#if necessary, insert injections
if injfile is not None and injfile != "noninj" and injfile != "real":
	for k in range(n_injs):
		if startgps[k] > valid_times[segment] and gps[k] + 1 < valid_times[segment] + end_cutoff:
			print("inserting injection {}".format(k))
			#print the paramters
			print("mass1:", mass1[k], "mass2:", mass2[k], "spin1z:", spin1z[k], "spin2z:", spin2z[k], "distance:", distance[k])
			#insert into the loaded noise
			#temp_approximant = td_approximant
			#temp_f_lower = f_lower
			
			temp_delta_t = delta_t
			temp_delta_t = delta_t/8 #catch-all for now

			temp_td_approximant = select_approximant(mass1[k], mass2[k], td_approximant, domain='time')
			if temp_td_approximant != "SpinTaylorT4" and t_at_f(mass1[k], mass2[k], 10) > 100:
				temp_f_lower = 20
				print("setting f_lower to 20 Hz to avoid generating an expensive waveform")
				print("Approximant:", temp_td_approximant, "mass1:", mass1[k], "mass2:", mass2[k])
			else:
				temp_f_lower = 10
			temp_f_lower = min(temp_f_lower, maximum_f_lower(mass1[k], mass2[k]))

			# if mass1[k] + mass2[k] > 9 and temp_approximant == "SEOBNRv4P":
			# 	temp_approximant = "SEOBNRv4PHM"
			# 	#temp_f_lower = 10
				
			# if mass1[k] + mass2[k] < 9 and temp_approximant == "SEOBNRv4PHM":
			# 	temp_approximant = "SEOBNRv4P"
			
			# #NOTE: changed so that f_lower is always 10 Hz at most, as injections are meant to be generated down to 10 Hz.
			# #temp_f_lower = min(maximum_f_lower(mass1[k], mass2[k]), f_lower)
			# temp_f_lower = min(maximum_f_lower(mass1[k], mass2[k]), 10)
			
			#TODO: generalise to all FD approximants
			if temp_td_approximant == "TaylorF2Ecc":
				temp_f_lower = min(temp_f_lower, 20)
				print("WARNING! Ignoring x and y spins for TaylorF2Ecc approximant")
				#spin1x = spin1x[k], spin1y = spin1y[k],
				#spin2x = spin2x[k], spin2y = spin2y[k],
				hp_f, hc_f = get_fd_waveform(mass1 = mass1[k], mass2 = mass2[k],
										 spin1z = spin1z[k], spin2z = spin2z[k],
										 eccentricity = eccentricity[k],
										 inclination = inclination[k], distance = distance[k], 
										 f_lower = temp_f_lower, delta_f = delta_f, f_ref = 20,
										 approximant = temp_td_approximant, f_final = f_final)
				hp = hp_f.to_timeseries(delta_t=temp_delta_t)
				hc = hc_f.to_timeseries(delta_t=temp_delta_t)
				
			else:
				hp, hc = get_td_waveform(mass1 = mass1[k], mass2 = mass2[k], 
							spin1x = spin1x[k], spin1y = spin1y[k],
							spin2x = spin2x[k], spin2y = spin2y[k],
							spin1z = spin1z[k], spin2z = spin2z[k],
							inclination = inclination[k], distance = distance[k], 
							delta_t = temp_delta_t, f_lower = temp_f_lower, approximant = temp_td_approximant) 
			hp = hp.resample(delta_t)
			hc = hc.resample(delta_t)
			offset = max(int(hp.sample_times[-1]/delta_t),0)
			print("waveform duration:", len(hp)*delta_t, "offset:", offset)

			for ifo in ifos:
				f_plus, f_cross = all_detectors[ifo].antenna_pattern(
					right_ascension=right_ascension[k], declination=declination[k],
					polarization=polarization[k],
					t_gps=gps_dict[ifo][k])
				
				detector_signal = f_plus * hp + f_cross * hc

				end_idx = int((gps_dict[ifo][k]-valid_times[segment]) * 2048)
				#this if statement ensures the waveform doesn't go beyond the start of the segment
				if len(detector_signal) >= end_idx:
					start_idx = 0
					detector_signal = detector_signal[len(detector_signal) - end_idx:]
				else:
					start_idx = end_idx - len(detector_signal)
				print(start_idx, end_idx)
				#print(len(detector_signal), end_idx)
				noise[ifos.index(ifo),end_idx - len(detector_signal)+offset:end_idx+offset] += detector_signal


strain = {}
strain_np = {}

start = time.time()
for ifo in range(len(ifos)):
	strain[ifo] = TimeSeries(noise[ifo], delta_t=delta_t, copy=False)
	strain[ifo] = highpass(strain[ifo],f_lower).to_frequencyseries(delta_f=delta_f).data
	strain[ifo] = np.array([strain[ifo]])[:,kmin:kmax]
	strain_np[ifo] = np.repeat(strain[ifo], n_templates, axis=0)
	with strain_time.get_lock():
		strain_time.value += time.time() - start

#starting the loop

#if __name__ == '__main__':



def calc_batch(i):
	batch_start = time.time()
	#nonwindowed_SNR = np.empty((len(ifos), n_templates, slice_duration*sample_rate), dtype=np.float32)
	print("batch", i, flush=True)
	batch = i
	start = time.time()
	#template_start_idx = i*templates_per_batch + template_start

	#if i == n_batches - 1:
	if len(template_banks[i]) != templates_per_batch:
		#print("loading templates in range",template_start_idx, templates_per_job + template_start)
		#n_templates =  (templates_per_job + template_start) - template_start_idx
		print("loading templates in range", template_banks[i][0], template_banks[i][-1])
		n_templates = len(template_banks[i])
		t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)
		print("last templates for this job, only {}".format(n_templates))
		for ifo in range(len(ifos)):
			#strain = TimeSeries(noise[ifo], delta_t=delta_t, copy=False)
			#strain = highpass(strain,f_lower).to_frequencyseries(delta_f=delta_f).data
			#strain = np.array([strain])[:,kmin:kmax]
			strain_np[ifo] = np.repeat(strain[ifo], n_templates, axis=0)
			with strain_time.get_lock():
				strain_time.value += time.time() - start
			#strain_time += time.time() - start
	else:
		#print("loading templates in range",template_start_idx, (i+1)*templates_per_batch + template_start)
		print("loading templates in range", template_banks[i][0], template_banks[i][-1])
		n_templates = templates_per_batch
		t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)

	#print(n_templates)

	#with mp.Pool(4) as pool:
	#	t_templates = np.array(pool.map(load_templates, templates[template_start_idx:template_start_idx + n_templates]))
	for j in range(n_templates):
		#t_templates[j] = get_fd_waveform(mass1 = templates[template_start_idx + j,1], 
		#						mass2 = templates[template_start_idx + j,2],
		#						spin1z = templates[template_start_idx + j,3], 
		#						spin2z = templates[template_start_idx + j,4],
		#						approximant = fd_approximant, f_lower = f_lower, 
		#						delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]

		temp_fd_approximant = select_approximant(templates[template_banks[i][j],1], templates[template_banks[i][j],2], 
										   fd_approximant, domain='frequency')
		t_templates[j] = get_fd_waveform(mass1 = templates[template_banks[i][j],1], 
								mass2 = templates[template_banks[i][j],2],
								spin1z = templates[template_banks[i][j],3], 
								spin2z = templates[template_banks[i][j],4],
								approximant = temp_fd_approximant, f_lower = f_lower, 
								delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]
	#TODO: uncomment
	with template_time.get_lock():
		template_time.value += time.time() - start
	#template_time += time.time() - start
	#template_conj = np.conjugate(t_templates)

	nonwindowed_SNR = np.empty((len(ifos), n_templates, slice_duration*sample_rate), dtype=np.float32)

	for ifo in range(len(ifos)):
	
		start = time.time()
		a = start_cutoff*sample_rate
		b = end_cutoff*sample_rate
		
		if chisq:
			x = numpy_matched_filter(strain_np[ifo], t_templates, psds[ifos[ifo]], N, kmin, kmax, duration, delta_t, f_lower)
			x *= reduced_chisquared_precomputed_SNR(x, t_templates, strain_np[ifo], psds[ifos[ifo]], kmin, kmax, delta_f, num_bins = n_chisq_bins)
			nonwindowed_SNR[ifo, :, :] = np.abs(x[:,a:b]).astype(np.float32)
		else:
			nonwindowed_SNR[ifo, :, :] = np.abs(numpy_matched_filter(strain_np[ifo], t_templates, psds[ifos[ifo]], N, kmin, kmax, duration, delta_t, f_lower)[:,a:b]).astype(np.float32)
		

		#y = mf_in_place(strain_np, psds[ifos[ifo]], N, kmin, kmax, template_conj, template_norm)
		#nonwindowed_SNR[ifo, :, :] = np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32)

		#print("y shape:", y.shape)
		#mf_time += time.time() - start
		with mf_time.get_lock():
			mf_time.value += time.time() - start



	#snrs.append(nonwindowed_SNR)
	s = time.time()
	if injfile is not None:
		#get_zerolag(nonwindowed_SNR, template_start_idx)
		#get_timeslide_new(0, nonwindowed_SNR, template_start_idx)
		get_timeslide_new(0, nonwindowed_SNR, template_banks[i])
	else:
		
		#print("starting timeslides")
		for j in range(num_time_slides):
			#TODO: since we're passing nonwindowed SNR, we can roll it in-place and save memory.
			#get_timeslide_new(j, nonwindowed_SNR, template_start_idx)
			#roll SNR array to perform a time shift
			nonwindowed_SNR[1,:,:] = np.roll(nonwindowed_SNR[1,:,:], 2048, axis = 1)
			get_timeslide_new(j, nonwindowed_SNR, template_banks[i])
		#with mp.Pool(4) as pool:
		#	pool.map(get_timeslide_new, range(100))
	with timeslide_time.get_lock():
		timeslide_time.value += time.time() - s
	#timeslide_time += time.time() - s

	print("batch time", time.time() - batch_start)
	sys.stdout.flush()
	#do some garbage collection
	gc.collect()

if __name__ == '__main__':
	with mp.Pool(n_workers) as pool:
		#chunk size of 1 is fine as each batch takes several seconds.
		pool.map(calc_batch, range(n_batches), chunksize = 1)

		#now we need to wait for all the processes to finish
		pool.close()
		pool.join()


print("timeslides dtype:", timeslides.dtype)
print("SNR_array dtype:", SNR_array.dtype)

#get the size of myfolder
size_gb = sum(os.path.getsize(f) for f in os.listdir(myfolder) if os.path.isfile(f))/ (1024 ** 3)
print("myfolder size:", size_gb, "GB")
if size_gb > 500:
	print("folder is larger than 500 GB, exiting without saving!")
	exit(1)
else:
	print("folder is smaller than 500 GB, saving timeslides and SNR array")
np.save(os.path.join(myfolder,"timeslides_{}_0.npy".format(job_id)), timeslides)
#save SNR array to file
np.save(os.path.join(myfolder,"SNR_array_{}_0.npy".format(job_id)), SNR_array)

# if streamline:
# 	np.save(os.path.join(save_dir,"timeslides_{}_0.npy".format(job_id)), timeslides)
# 	#save SNR array to file
# 	np.save(os.path.join(save_dir,"SNR_array_{}_0.npy".format(job_id)), SNR_array)
# else:
# 	np.save(os.path.join(myfolder,"timeslides_{}_0.npy".format(job_id)), timeslides)
# 	#save SNR array to file
# 	np.save(os.path.join(myfolder,"SNR_array_{}_0.npy".format(job_id)), SNR_array)

#SNR_array.flush()
#header = np.lib.format.header_data_from_array_1_0(SNR_array)
#with open(myfolder + "/SNR_array_{}_{}.npy".format(job_id, worker_id), 'r+b') as f:
#	np.lib.format.write_array_header_1_0(f, header)

#np.save("/fred/oz016/alistair/infernus/runs/test/BG_mp/SNR_array_{}.npy".format(job_id), SNR_array)

#np.save(myfolder + "/response_array_{}.npy".format(job_id), response_array)

print("template loading time", template_time.value)
print("mf time", mf_time.value)
print("strain time", strain_time.value)
print("timeslide time", timeslide_time.value)
#print("triton time", triton_time)
print("criterion time", criterion_time.value)
print("merge time", merge_time.value)


# if segment == 0 and injfile is None:
# 	#save a copy of the timeslides and SNR array to the save directory
# 	print("saving timeslides and SNR array to", save_dir)
# 	np.save("/fred/oz016/alistair/infernus/SNR_array_{}.npy".format(segment), SNR_array)
# 	np.save("/fred/oz016/alistair/infernus/timeslides_{}.npy".format(segment), timeslides)
# elif segment == 0 and injfile is not None:
# 	print("saving inj timeslides and SNR array to", save_dir)
# 	np.save("/fred/oz016/alistair/infernus/SNR_array_inj_{}.npy".format(segment), SNR_array)
# 	np.save("/fred/oz016/alistair/infernus/timeslides_inj_{}.npy".format(segment), timeslides)

if injfile == "real":
	print("saving correct path to", myfolder + "/real_event.txt")
	#NOTE: changed with new style of workflow
	#save_dir = os.path.join(os.path.dirname(save_dir), event)
	save_dir = os.path.join(save_dir, event)
	with open(myfolder + "/real_event.txt", 'w') as f:
		f.write(save_dir)