import numpy as np
import pycbc.catalog
import os
from gwpy.timeseries import TimeSeries as GWPYTimeSeries
from pycbc.types import TimeSeries
import h5py

from GWSamplegen.noise_utils import get_data_from_OzStar

from infernus.real_utils import get_real_events


if __name__ == "__main__":
	import argparse
	import json

	parser = argparse.ArgumentParser(description='Load real events')

	parser.add_argument('--configfile', type=str, default = None)

	cfg = parser.parse_args()
	cfg_file = cfg.configfile

	#load the config file
	with open(cfg_file) as f:
		config = json.load(f)
		print(config)
		#need to extract the template limits

		# m1_lower = config['template_mass1_min']
		# m2_lower = config['template_mass2_min']
		# m1_upper = config['template_mass1_max']
		# m2_upper = config['template_mass2_max']
		m1_lower = 1
		m2_lower = 1
		m1_upper = 1000
		m2_upper = 1000
		print("For now, requiring H1 and L1 data")
		events = get_real_events(ifos = ['H1', 'L1'], m1_lower = m1_lower, m2_lower = m2_lower, m1_upper = m1_upper, m2_upper = m2_upper, \
						   exclude_marginal = False, padding = config['duration'], require_centred= False, verbose= True)
		print(events)

		#create a directory for each event

		for event in events.keys():
			print("Creating directory for", event)
			os.makedirs(os.path.join(config['save_dir'], event), exist_ok = True)
			if events[event]['offset'] != 0:
				print("Event", event, "has an offset of", events[event]['offset'], "seconds. Saving the offset for later")
				#write a text file with the offset
				with open(os.path.join(config['save_dir'], event, "offset.txt"), 'w') as f:
					f.write(str(events[event]['offset']))

			# with open(os.path.join(config['save_dir'], "inj.json"), 'r') as f:
			# 	real = json.load(f)
			# 	real['jobname'] = real['jobname'][:-3] + event
			# 	real['injfile'] = "real"


		print(len(events.keys()))

		#make a copy of the inj config file in the save_dir directory

		#import shutil
		#shutil.copy(os.path.join(config['save_dir'], "inj.json"), os.path.join(config['save_dir'], "real_events.json"))
		#open and modify the real_events.json file (only with the old style of workflow)
		#check if inj.json exists first
		if os.path.exists(os.path.join(config['save_dir'], "inj.json")):
			#print("old style of workflow")
			with open(os.path.join(config['save_dir'], "inj.json"), 'r') as f:
				real = json.load(f)
				real['jobname'] = real['jobname'][:-3] + "real"
				real['injfile'] = "real" 

				#save the modified file
				json.dump(real, open(os.path.join(config['save_dir'], "real_events.json"), "w"), indent = 4)