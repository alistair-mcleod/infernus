from GWSamplegen.waveform_utils import t_at_f
from pycbc.detector import Detector
import numpy as np
import h5py


all_detectors = {"H1": Detector("H1"), "L1": Detector("L1"), "V1": Detector("V1"), "K1": Detector("K1")}

def load_O3_injections(injfile, start_time= 0, end_time = 0, f_lower = 20, verbose = False):
	"""
	Loads O3a / O3b injections from an hdf file, and returns a dictionary of the events that fall within the specified time range.
	
	input:
		injfile: Either a string path to the hdf file containing the injections, or an already opened h5py file object. 
		start_time: GPS time of the start of the segment to consider
		end_time: GPS time of the end of the segment to consider
	"""
	if isinstance(injfile, str):
		if verbose:
			print("using injection file", injfile)
		f = h5py.File(injfile, 'r')
	elif isinstance(injfile, h5py.File):
		f = injfile
	if "injections" not in f:
		raise ValueError("injections group not found in hdf file, file should be in O3 format")

	if start_time > 0 and end_time > 0:
		mask = (f['injections']['gps_time'][:] > start_time) & (f['injections']['gps_time'][:] < end_time)
	n_injs = np.sum(mask)

	if verbose:
		print("number of injections in this segment:", n_injs)

	T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
	N_draw = f.attrs['total_generated']
	accepted_fraction = f.attrs['n_accepted']/N_draw
	
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
	ret = {"gps": gps, "gps_dict": gps_dict, "mass1": mass1, "mass2": mass2, "spin1x": spin1x, "spin1y": spin1y, "spin1z": spin1z,
			"spin2x": spin2x, "spin2y": spin2y, "spin2z": spin2z, "distance": distance, "inclination": inclination,
			"polarization": polarization, "right_ascension": right_ascension, "declination": declination, 
			"optimal_snr_h": optimal_snr_h, "optimal_snr_l": optimal_snr_l, "eccentricity": eccentricity, "startgps": startgps, "n_injs": n_injs}
	return ret	

def load_O4_injections(injfile, start_time, end_time, f_lower = 20, verbose = False):
	"""
	Loads O4a injections from an hdf file, and returns a dictionary of the events that fall within the specified time range.
	
	input:
		injfile: Either a string path to the hdf file containing the injections, or an already opened h5py file object. 
		start_time: GPS time of the start of the segment to consider
		end_time: GPS time of the end of the segment to consider
	"""

	#check if injfile is an h5py file or a string path to an hdf file
	if isinstance(injfile, str):
		if verbose:
			print("using injection file", injfile)
		f = h5py.File(injfile, 'r')
	elif isinstance(injfile, h5py.File):
		f = injfile
	if "events" not in f:
		raise ValueError("events group not found in hdf file, must be O4a or later")


	events = f['events'][()]
	mask = (events["time_geocenter"] > start_time) & (events["time_geocenter"] < end_time)
	n_injs = np.sum(mask)
	if verbose:
		print(f"Found {n_injs} events in the specified time range.")

	gps = events["time_geocenter"][mask]
	mass1 = events["mass1_detector"][mask]
	mass2 = events["mass2_detector"][mask]
	spin1x = events["spin1x"][mask]
	spin1y = events["spin1y"][mask]
	spin1z = events["spin1z"][mask]
	spin2x = events["spin2x"][mask]
	spin2y = events["spin2y"][mask]
	spin2z = events["spin2z"][mask]
	distance = events["luminosity_distance"][mask]
	inclination = events["inclination"][mask]
	polarization = events["polarization"][mask]
	right_ascension = events["right_ascension"][mask]
	declination = events["declination"][mask]
	optimal_snr_h = events["snr_H"][mask]
	optimal_snr_l = events["snr_L"][mask]
	if "eccentricity" in events.dtype.names:
		eccentricity = events["eccentricity"][mask]
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
	if verbose:
		print("GPS times for injections should be correct now (injected from geocentre)")
	
	ret = {"gps": gps, "gps_dict": gps_dict, "mass1": mass1, "mass2": mass2, "spin1x": spin1x, "spin1y": spin1y, "spin1z": spin1z,
			"spin2x": spin2x, "spin2y": spin2y, "spin2z": spin2z, "distance": distance, "inclination": inclination,
			"polarization": polarization, "right_ascension": right_ascension, "declination": declination, 
			"optimal_snr_h": optimal_snr_h, "optimal_snr_l": optimal_snr_l, "eccentricity": eccentricity, "startgps": startgps, "n_injs": n_injs}
	
	return ret


def load_injections_temp(injfile, start_time, end_time,
						start_cutoff = 100, end_cutoff = 1000, duration = 1024,
						f_lower = 20, pipelines = None, verbose = False):
	#helper function to select the correct loading function
	if isinstance(injfile, str):
		if verbose:
			print("using injection file", injfile)
		f = h5py.File(injfile, 'r')
	elif isinstance(injfile, h5py.File):
		f = injfile
	
	if "injections" in f:
		return load_injections_O3_temp(f, start_time, end_time, start_cutoff, end_cutoff, duration, f_lower, pipelines, verbose)
	elif "events" in f:
		return load_injections_O4_temp(f, start_time, end_time, start_cutoff, end_cutoff, duration, f_lower, pipelines, verbose)
	else:
		raise ValueError("injections or events group not found in hdf file, file should be in O3 or O4 format")



def load_injections_O3_temp(injfile, start_time, end_time,
							start_cutoff = 100, end_cutoff = 1000, duration = 1024,
							 f_lower = 20, pipelines = None, verbose = False):
	"Temporary function to generalise postprocessing.get_inj_data, this function should be combined with load_O3_injections"
	
	#pipelines = ["pycbc_hyperbank", "mbta", "gstlal"]
	
	if isinstance(injfile, str):
		if verbose:
			print("using injection file", injfile)
		f = h5py.File(injfile, 'r')
	elif isinstance(injfile, h5py.File):
		f = injfile
	if "injections" not in f:
		raise ValueError("injections group not found in hdf file, file should be in O3 format")

	T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
	N_draw = f.attrs['total_generated']
	accepted_fraction = f.attrs['n_accepted']/N_draw

	gps_times = f['injections/gps_time'][:]
	network_snr = f['injections/optimal_snr_net'][:]
	h_snr = f['injections/optimal_snr_h'][:]
	l_snr = f['injections/optimal_snr_l'][:]

	m1 = f['injections/mass1_source'][:]
	m2 = f['injections/mass2_source'][:]
	s1x = f['injections/spin1x'][:]
	s1y = f['injections/spin1y'][:]
	s1z = f['injections/spin1z'][:]    
	s2x = f['injections/spin2x'][:]
	s2y = f['injections/spin2y'][:]
	s2z = f['injections/spin2z'][:]
	z = f['injections/redshift'][:]
	distance = f['injections']['distance'][:]
	right_ascension = f['injections']['right_ascension'][:]
	declination = f['injections']['declination'][:]
	inclination = f['injections']['inclination'][:]
	polarization = f['injections']['polarization'][:]

	m1_det = f['injections/mass1'][:]
	m2_det = f['injections/mass2'][:]

	p_draw = f['injections/sampling_pdf'][:]

	#pastro_cwb = f['injections/pastro_cwb'][:]
	#pastro_gstlal = f['injections/pastro_gstlal'][:]    
	#pastro_mbta = f['injections/pastro_mbta'][:]    
	#pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]    
	#pastro_pycbc = f['injections/pastro_pycbc_hyperbank'][:]

	pipeline_fars = {}
	pipeline_pastros = {}
	if pipelines is not None:
		for p in pipelines:
			if p == "pycbc":
				#Key name is "pycbc_hyperbank" in O3
				pipeline_fars["pycbc"] = f[f'injections/far_{p}_hyperbank'][:] / (86400*365.25)
				pipeline_pastros["pycbc"] = f[f'injections/pastro_{p}_hyperbank'][:]
			else:
				pipeline_fars[p] = f[f'injections/far_{p}'][:] / (86400*365.25)
				pipeline_pastros[p] = f[f'injections/pastro_{p}'][:]
		# far_cwb = f['injections/far_cwb'][:]
		# far_gstlal = f['injections/far_gstlal'][:]
		# far_mbta = f['injections/far_mbta'][:]
		# #far_pycbc_bbh = f['injections/far_pycbc_bbh'][:]
		# far_pycbc = f['injections/far_pycbc_hyperbank'][:]

	m1_prior = f['injections/mass1_source_sampling_pdf'][:]
	m1_m2_prior = f['injections/mass1_source_mass2_source_sampling_pdf'][:]

	s1_prior = f['injections/spin1x_spin1y_spin1z_sampling_pdf'][:]
	s2_prior = f['injections/spin2x_spin2y_spin2z_sampling_pdf'][:]


	ret = {
		"gps_times": gps_times, "network_snr": network_snr, "h_snr": h_snr, "l_snr": l_snr,
		"m1": m1, "m2": m2, "s1x": s1x, "s1y": s1y, "s1z": s1z, "s2x": s2x, "s2y": s2y, "s2z": s2z,
		"z": z, "distance": distance, "right_ascension": right_ascension, "declination": declination,
		"inclination": inclination, "polarization": polarization,
		"m1_det": m1_det, "m2_det": m2_det,
		"p_draw": p_draw,
		#"pastro_cwb": pastro_cwb, "pastro_gstlal": pastro_gstlal, "pastro_mbta": pastro_mbta,
		#"pastro_pycbc": pastro_pycbc,
		#"far_cwb": far_cwb, "far_gstlal": far_gstlal, "far_mbta": far_mbta, "far_pycbc": far_pycbc,
		"m1_prior": m1_prior, "m1_m2_prior": m1_m2_prior,
		"s1_prior": s1_prior, "s2_prior": s2_prior, "T_obs": T_obs, "N_draw": N_draw, "accepted_fraction": accepted_fraction,
		"pipeline_fars": pipeline_fars, "pipeline_pastros": pipeline_pastros, "file_format": "O3"
	}

	return ret


def load_injections_O4_temp(injfile, start_time, end_time,
	start_cutoff = 100, end_cutoff = 1000, duration = 1024,
	f_lower = 20, pipelines = None, verbose = False):

	print("Temporary function to generalise postprocessing.get_inj_data, this function should be combined with load_O4_injections")

	if isinstance(injfile, str):
		if verbose:
			print("using injection file", injfile)
		f = h5py.File(injfile, 'r')

	elif isinstance(injfile, h5py.File):
		f = injfile

	if "events" not in f:
		raise ValueError("events group not found in hdf file, must be O4a or later")

	
	T_obs = f.attrs['total_analysis_time']/(365.25*24*3600) # years
	N_draw = f.attrs['total_generated']
	accepted_fraction = f.attrs['num_accepted']/N_draw

	f = f['events']

	gps_times = f["time_geocenter"]
	network_snr = f["snr_net"]
	h_snr = f["snr_H"]
	l_snr = f["snr_L"]

	m1 = f["mass1_source"]
	m2 = f["mass2_source"]
	s1x = f["spin1x"]
	s1y = f["spin1y"]
	s1z = f["spin1z"]
	s2x = f["spin2x"]
	s2y = f["spin2y"]
	s2z = f["spin2z"]
	z = f["z"]
	distance = f["luminosity_distance"]
	right_ascension = f["right_ascension"]
	declination = f["declination"]
	inclination = f["inclination"]
	polarization = f["polarization"]

	m1_det = f["mass1_detector"]
	m2_det = f["mass2_detector"]

	theta1 = f["spin1_polar_angle"]
	phi1 = f["spin1_azimuthal_angle"]
	theta2 = f["spin2_polar_angle"]
	phi2 = f["spin2_azimuthal_angle"]

	#taken from the markdown file here: https://zenodo.org/records/16740117
	lnprob = f["lnpdraw_mass1_source"] \
			+ f["lnpdraw_mass2_source_GIVEN_mass1_source"] \
			+ f["lnpdraw_z"] \
			+ f["lnpdraw_spin1_magnitude"] \
			+ f["lnpdraw_spin1_polar_angle"] \
			+ f["lnpdraw_spin1_azimuthal_angle"] \
			+ f["lnpdraw_spin2_magnitude"] \
			+ f["lnpdraw_spin2_polar_angle"] \
			+ f["lnpdraw_spin2_azimuthal_angle"]
	p_draw = np.e ** lnprob

	pipeline_fars = {}
	pipeline_pastros = {}
	if pipelines is not None:
		for p in pipelines:
			pipeline_fars[p] = f[f'{p}_far'][:] / (86400*365.25)
			if p == "pycbc" or p == "gstlal":
				pipeline_pastros[p] = np.zeros_like(p_draw) #NOTE: not provided in O4a injections, make these keys arbirary
			else:
				pipeline_pastros[p] = f[f'{p}_p_astro'][:]
				
		#far_cwb = f['cwb-bbh_far'] #TODO: generalise, we shouldn't need to handle CWB separately

		# pastro_cwb = f["cwb-bbh_p_astro"]
		# pastro_gstlal = np.zeros_like(pastro_cwb) #NOTE: not provided in O4a injections
		# pastro_mbta = f["mbta_p_astro"]
		# pastro_pycbc = np.zeros_like(pastro_cwb) #NOTE: not provided in O4a injections
		# #pastro_pycbc_bbh = np.zeros_like(pastro_cwb) #TODO: pycbc_BBH is deprecated in O4, make these keys arbirary
		# pipeline_pastros["cwb-bbh"] = pastro_cwb
		# pipeline_pastros["gstlal"] = pastro_gstlal
		# pipeline_pastros["mbta"] = pastro_mbta
		# pipeline_pastros["pycbc"] = pastro_pycbc

	m1_prior = np.e ** f["lnpdraw_mass1_source"]
	#TODO: confirm this is 100% correct
	m1_m2_prior = np.e ** (f["lnpdraw_mass2_source_GIVEN_mass1_source"] + f["lnpdraw_mass1_source"]) 

	#TODO: confirm these are equivalent to sx_sy_sz priors
	s1_prior = np.e ** (f['lnpdraw_spin1_magnitude'] + f['lnpdraw_spin1_polar_angle'] + f['lnpdraw_spin1_azimuthal_angle'])
	s2_prior = np.e ** (f['lnpdraw_spin2_magnitude'] + f['lnpdraw_spin2_polar_angle'] + f['lnpdraw_spin2_azimuthal_angle'])
	weights = f["weights"]
	ret = {
		"gps_times": gps_times, "network_snr": network_snr, "h_snr": h_snr, "l_snr": l_snr,
		"m1": m1, "m2": m2, "s1x": s1x, "s1y": s1y, "s1z": s1z, "s2x": s2x, "s2y": s2y, "s2z": s2z,
		"z": z, "distance": distance, "right_ascension": right_ascension, "declination": declination,
		"inclination": inclination, "polarization": polarization,
		"m1_det": m1_det, "m2_det": m2_det,
		"p_draw": p_draw,
		#"pastro_cwb": pastro_cwb, "pastro_gstlal": pastro_gstlal, "pastro_mbta": pastro_mbta,
		#"pastro_pycbc": pastro_pycbc,
		#"far_cwb": far_cwb, "far_gstlal": pipeline_fars["gstlal"], 
		#"far_mbta": pipeline_fars["mbta"], "far_pycbc": pipeline_fars["pycbc"],
		"m1_prior": m1_prior, "m1_m2_prior": m1_m2_prior,
		"theta1": theta1, "phi1": phi1, "theta2": theta2, "phi2": phi2, "weights": weights,
		"s1_prior": s1_prior, "s2_prior": s2_prior, "T_obs": T_obs, "N_draw": N_draw, "accepted_fraction": accepted_fraction,
		"pipeline_fars": pipeline_fars, "pipeline_pastros": pipeline_pastros, "file_format": "O4"
	}

	return ret