from GWSamplegen.noise_utils import get_valid_noise_times, get_valid_noise_times_from_segments
import json
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--jsonfile', type=str, default=None)
parser.add_argument('--injindex', type = int, default = -1)
args = parser.parse_args()
argsfile = args.jsonfile
injindex = args.injindex

args = json.load(open(argsfile, "r"))
noise_dir = args["noise_dir"]
maxnoisesegs = args["max_noise_segments"]
duration = args["duration"]
savedir = args["save_dir"]
injfile = args["injfile"]
# if injindex >= 0:
# 	print("Using injection file index", injindex)
# 	print("Need to modify the save directory to include the index")
# 	savedir = os.path.join(savedir, "inj_" + str(injindex))
# 	print("New save directory is ", savedir)

if injfile == "None":
	fail_tolerant = True
	print("Background run, fail tolerant mode is on")
else:
	fail_tolerant = False
	print("Not a background run, fail tolerant mode is off")

print(noise_dir)
delta_f = 1/duration

if isinstance(noise_dir, list):
	print("new noise loading")
	valid_times = get_valid_noise_times_from_segments(noise_dir, duration, 900, blacklisting = False)
else:
	valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, 900, blacklisting = False)

print(len(valid_times), "files to merge")

if isinstance(injfile, list):
	print("Using multiple injfiles, cleaning all segments together")
	loops = len(injfile)
else:
	loops = 1

print("Number of directories to merge:", loops)

#possible scenarios: some injfile runs are fine, one or more isn't.
#no injfile run is fine
#all injfile runs are fine

missing_segments = []
for loop in range(loops):
	for i in range(len(valid_times)):
		#check if ALL files are present
		if isinstance(injfile, list):
			this_savedir = os.path.join(savedir, f"inj_{loop}")
		else:
			this_savedir = savedir
		if os.path.exists(os.path.join(this_savedir, "timeslides_{}.npy".format(i))) == False:
			print("Missing segment", i)
			if fail_tolerant == False:
				missing_segments.append(i)

#sort and remove duplicates
missing_segments = sorted(list(set(missing_segments)))
print("Total missing files: ", len(missing_segments))

max_array = 2048
missing_segments = sorted(set(i % max_array for i in missing_segments))

if len(missing_segments) > 0:
    
    missing_ranges = []
    start = missing_segments[0]
    end = start
    for i in range(1, len(missing_segments)):
        if missing_segments[i] == end + 1:
            end = missing_segments[i]
        else:
            if start == end:
                missing_ranges.append(str(start))
            else:
                missing_ranges.append(f"{start}-{end}")
            start = missing_segments[i]
            end = start
    if start == end:
        missing_ranges.append(str(start))
    else:
        missing_ranges.append(f"{start}-{end+1}")
    missing = '"' + ",".join(missing_ranges) + '"'
    print("bash ${INFERNUS_DIR}/bin/recovery.sh {} {}".format(argsfile, missing))

#if we got here, all files are present, we can merge

for loop in range(loops):
	if isinstance(injfile, list):
		this_savedir = os.path.join(savedir, f"inj_{loop}")
	else:
		this_savedir = savedir
	print("Merging segments in", this_savedir)
	timeslides = np.load(os.path.join(this_savedir, "timeslides_0.npy"))

	for i in range(1, len(valid_times)):
		try:
			ts_load = np.load(os.path.join(this_savedir, "timeslides_{}.npy".format(i)))
			timeslides = np.concatenate((timeslides, ts_load), axis = 1)
			#remove the file
			os.remove(os.path.join(this_savedir, "timeslides_{}.npy".format(i)))
		except:
			print("Failed to merge segment",i)

	np.save(os.path.join(this_savedir, "timeslides.npy"), timeslides)
	print("Merged all segments")
	os.remove(os.path.join(this_savedir, "timeslides_0.npy"))