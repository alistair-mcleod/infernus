

import os
import numpy as np
import pycbc.catalog
from gwpy.timeseries import TimeSeries as GWPYTimeSeries
from pycbc.types import TimeSeries
import h5py

from GWSamplegen.noise_utils import get_data_from_OzStar

def attempt_event_load(gps, duration = 1024, start_cutoff = 100, end_cutoff = 1000, verbose = False):
	#By default we want to want to put the event candidate in the middle of a segment. 
	#This may not be possible if the event is too close to the start or end of a segment,
	#but it may still be possible to get the event within the segment using an offset.
	accept = True
	reth = get_data_from_OzStar(int(gps-duration/2), duration, "H1", return_bad_data = True)
	retl = get_data_from_OzStar(int(gps-duration/2), duration, "L1", return_bad_data = True)
	#need to determine if the bad data is before or after the merger.
	#bad_h, bad_l = 0, 0
	bad_h_start, bad_h_end, bad_l_start, bad_l_end, offset = 0, 0, 0, 0, 0
	if len(np.where(reth.data == 0)[0]) > 0:
		bad_h_start = np.where(reth.data == 0)[0][0] / 2048
		bad_h_end = np.where(reth.data == 0)[0][-1] / 2048

	if len(np.where(retl.data == 0)[0]) > 0:
		bad_l_end = np.where(retl.data == 0)[0][-1] / 2048
		bad_l_start = np.where(retl.data == 0)[0][0] / 2048

	if (bad_h_start != 0 or bad_h_end != 0) and verbose:
		print("Bad data starts at", bad_h_start, "seconds in H1 and ends at", bad_h_end, "seconds in H1")
	if (bad_l_start != 0 or bad_l_end != 0) and verbose:
		print("Bad data starts at", bad_l_start, "seconds in L1 and ends at", bad_l_end, "seconds in L1")

	if bad_h_start == 0 and bad_h_end == 0 and bad_l_start == 0 and bad_l_end == 0:
		#if there's no bad data at all we can return True and an offset of 0
		if verbose:
			print("No bad data in segment")
		return accept, 0
	if bad_h_start > duration/2 or bad_l_start > duration/2:
		if verbose:
			print("Bad data is after the merger")
		offset = - duration + int((min(np.array([bad_h_start, bad_l_start])[np.nonzero([bad_h_start, bad_l_start])[0]]) -10))

	if (bad_h_end < duration/2 and bad_h_end !=0) or (bad_l_end < duration/2 and bad_l_end != 0):
		print("Bad data is before the merger")
		offset = int(max(bad_h_end, bad_l_end) + 10)
	if (bad_h_start < duration/2 and bad_h_end > duration/2) or (bad_l_start < duration/2 and bad_l_end > duration/2):
		print("Bad data crosses the merger, rejecting event")
		accept = False
	if duration//2 - offset - start_cutoff < 0 or duration//2 - offset > end_cutoff:
		if verbose:
			print("WARNING: offset is too large, merger will not be visible")
		accept = False
	#then confirm that we can get the data with the offset included
	if accept:
		if verbose:
			print("offset is",offset)
			print("Merger is now", duration/2 - offset, "seconds after the start of the segment")
		reth = get_data_from_OzStar(int(gps-duration/2 + offset), duration, "H1", return_bad_data = True)
		retl = get_data_from_OzStar(int(gps-duration/2 + offset), duration, "L1", return_bad_data = True)
		if len(np.where(reth.data == 0)[0]) > 0 or len(np.where(retl.data == 0)[0]) > 0:
			print("Problem: still bad data in segment")
			accept = False
	return accept, offset

def get_GWTC_events(m1_lower = 0.0, m2_lower = 0.0, m1_upper=1000, m2_upper=1000, exclude_marginal = True, verbose = False):
	check_ifos = ['H1', 'L1', 'V1']
	#Note that the masses should be  in detector frame (i.e. template bank boundaries)
	event_shortlist = {}
	found_events = []
	#From the PyCBC source code.
	catalogs = {'GWTC-1-confident': 'LVC',
				'GWTC-1-marginal': 'LVC',
				'Initial_LIGO_Virgo': 'LVC',
				'O1_O2-Preliminary': 'LVC',
				'O3_Discovery_Papers': 'LVC',
				'GWTC-2': 'LVC',
				'GWTC-2.1-confident': 'LVC',
				'GWTC-2.1-marginal': 'LVC',
				'GWTC-3-confident': 'LVC',
				'GWTC-3-marginal': 'LVC',
				'GWTC-4.0': 'LVC'}
	for catalog in catalogs.keys():
		x = pycbc.catalog.Catalog(source=catalog)
		if exclude_marginal:
			if 'marginal' in catalog:
				continue
		#print(catalog)
		for key in x.data.keys():
			if x.data[key]['mass_2_source']:
				#if key[:8] not in found_events:

				m1 = x.data[key]['mass_1_source'] * (1+x.data[key]['redshift'])
				m2 = x.data[key]['mass_2_source'] * (1+x.data[key]['redshift'])
				if m1 < m1_upper and m1 > m1_lower and m2 < m2_upper and m2 > m2_lower:
						if verbose:
							print(key, "in catalog", catalog)

						add = True
						for other_event in event_shortlist.keys():
							if np.abs(event_shortlist[other_event]['gps'] - x.data[key]['GPS']) < 1:
								if verbose:
									print("Found duplicate event of", key, ":", other_event)
								if key[-1] > other_event[-1]:
									#print("Removing", other_event)
									del event_shortlist[other_event]
									break
								else:
									add = False
						
						if add:

							found_events.append(key[:-3])

							event_shortlist[key] = {}
							event_shortlist[key]['name'] = key
							event_shortlist[key]['m1'] = m1
							event_shortlist[key]['m2'] = m2
							event_shortlist[key]['gps'] = x.data[key]['GPS']
							event_shortlist[key]['catalog'] = catalog
							event_shortlist[key]['snr'] = x.data[key]['network_matched_filter_snr']
							event_shortlist[key]['p_astro'] = x.data[key]['p_astro'] 
							event_shortlist[key]['far'] = x.data[key]['far'] 
							event_ifos = []
							for ifo in check_ifos:
								if ifo in [i['detector'] for i in x.data[key]['strain']]:
									event_ifos.append(ifo)
							event_shortlist[key]['ifos'] = event_ifos
							if verbose:
								print("m1:",m1)
								print("m2:",m2)
								print(" ")

			elif x.data[key]['mass_2_source'] == None and catalog == 'GWTC-4.0' and not exclude_marginal: #Need to add a special case for GWTC-4.0 as some events do not have PE results 
				if verbose:
					print("Special case O4 event, no PE results but still including...")
				event_shortlist[key] = {}
				event_shortlist[key]['name'] = key
				event_shortlist[key]['m1'] = 0
				event_shortlist[key]['m2'] = 0
				event_shortlist[key]['gps'] = x.data[key]['GPS']
				event_shortlist[key]['catalog'] = catalog
				event_shortlist[key]['snr'] = x.data[key]['network_matched_filter_snr']
				event_shortlist[key]['p_astro'] = x.data[key]['p_astro'] 
				event_shortlist[key]['far'] = x.data[key]['far'] 
				event_ifos = []
				for ifo in check_ifos:
					if ifo in [i['detector'] for i in x.data[key]['strain']]:
						event_ifos.append(ifo)
				event_shortlist[key]['ifos'] = event_ifos

				
	return event_shortlist

def get_real_events(ifos = ['H1', 'L1'], m1_lower = 0.0, m2_lower = 0.0, m1_upper=1000, m2_upper=1000, 
					exclude_marginal = True, padding = 1024, require_centred = True, verbose = False):
	#returns a list of GPS times that match the given criteria.
	#padding is the amount of time to fetch around the event (in seconds)
	events = get_GWTC_events(m1_lower = m1_lower, m2_lower = m2_lower, m1_upper=m1_upper, m2_upper=m2_upper, exclude_marginal = exclude_marginal, verbose = verbose)

	good_events = {}
	for event in events.keys():
		good = True
		for ifo in ifos:
			if ifo not in events[event]['ifos']:
				if verbose:
					print("Skipping", event, "as it does not have data in", ifo)
				good = False
				break
		if good:
			good_events[event] = events[event]
	
	#next, try to load the data
	events = good_events
	good_events = {}
	for event in events.keys():
		if verbose:
			print("Loading", event)
		#TODO: add start_cutoff and end_cutoff as parameters
		good, offset = attempt_event_load(int(events[event]['gps']), duration = padding, start_cutoff = 100, end_cutoff = 1000, verbose = verbose)

		# good = True
		# for ifo in ifos:
		# 	#print("Loading", ifo)
		# 	data = get_data_from_OzStar(int(events[event]['gps']-padding/2), padding, ifo)
		# 	if data is None:
		# 		if verbose:
		# 			print("Failed to load", event, "in", ifo)
		# 		good = False
		# 		break
		# 	else:
		# 		if verbose:
		# 			print("Loaded", event, "in", ifo)
		# 		#events[event][ifo] = data
		# 		#append 
		
		if good:
			if require_centred and offset != 0:
				if verbose:
					print("Skipping", event, "as it is not centred in the segment")
				continue
			good_events[event] = events[event]
			good_events[event]['offset'] = offset
	
	return good_events