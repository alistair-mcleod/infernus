#standard imports
import os
import sys
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default="/fred/oz016/alistair/infernus/timeslides/")
parser.add_argument('--injfile', type=str, default=None)
args = parser.parse_args()

timeslides_dir = args.savedir
injfile = args.injfile # only used to check if we're doing injections, not actually used.

if injfile == "None":
	injfile = None
	print("No injfile! must be a BG run")
else:
	print("either a noninj or inj run")

print("timeslides_dir is {}".format(timeslides_dir))

def align_stats(zl, stats):
	true_idx = 0
	stat_buffer = np.zeros((zl.shape[0],stats.shape[1],stats.shape[2])) -1
	print("stat buffer has shape {}".format(stat_buffer.shape))
	for i in range(zl.shape[0]):
		if zl[i][0][0] == -1:
			#print("zl is -1 for zl {}".format(i))
			continue
		stat_buffer[i] = stats[true_idx]
		true_idx += 1
	
	return stat_buffer

def merge_zerolags(zl, new_zl, stats = None, new_stats = None, merge_target = 2):
	#merge_target is what we are overwriting the zerolags based on. By default it is 2, 
	#for merging based on network SNR. if it is 6 instead, it will merge based on highest prediction.
	count = 0
	for i in range(len(zl)):

		if i >= len(new_zl): #TODO: investigate why this is happening
			print("zl is longer than new_zl")
			continue

		if zl[i][0][0] == -1 and new_zl[i][0][0] == -1:
			#print("both zerolags are -1 for zl {}".format(i))
			continue
	
		if zl[i][0][0] == -1:
			zl[i] = np.copy(new_zl[i])
			if stats is not None:
				stats[i] = new_stats[i]
			continue

		if new_zl[i][0][0] == -1:
			continue

		if zl[i][0][merge_target] < new_zl[i][0][merge_target]:
			zl[i] = new_zl[i]
			if stats is not None:
				stats[i] = new_stats[i]
			count += 1

	print("overwrote {} zerolags".format(count))

	if stats is not None:
		return [zl, stats]
	else:
		return [zl]
	
#new method: merging zerolags AND stats

def merge_zerolags_and_stats(zl, new_zl, stats = None, new_stats = None, merge_target = 2):
	#merge_target is what we are overwriting the zerolags based on. By default it is 2, 
	#for merging based on network SNR. if it is 6 instead, it will merge based on highest prediction.
	count = 0
	for i in range(len(zl)):

		if i >= len(new_zl): #TODO: remove, this bug doesn't exist anymore
			print("zl is longer than new_zl")
			continue

		#if zl[i][0][0] == -1 and new_zl[i][0][0] == -1:
		#	#print("both zerolags are -1 for zl {}".format(i))
		#	continue
	
		if zl[i, 0, 0] == -1:
			zl[i] = np.copy(new_zl[i])
			#if stats is not None:
			#	stats[i] = new_stats[i]
			continue

		#if new_zl[i][0][0] == -1:
		#	continue

		#TODO: investigate different criteria for merging zls. 
		#index 2 is the highest SNR, but in the future we might want to use the highest network pred.
		if zl[i, 0, merge_target] < new_zl[i, 0, merge_target]:
			zl[i] = np.copy(new_zl[i])

		if stats is not None:
			#the -3 is due to the zerolag columns not stored in stats (#TODO change to -2 when you include template information)
			stats[i, stats[i, :, merge_target-3] < new_stats[i, :, merge_target-3], : ] = new_stats[i, stats[i, :, merge_target-3] < new_stats[i, :, merge_target-3], :]


	#print("overwrote {} zerolags".format(count))

	if stats is not None:
		return [zl, stats]
	else:
		return [zl]




#timeslides_dir = "/fred/oz016/alistair/infernus/timeslides/"
segment_dict = {}
segment_dict_predrank = {}
segment_dict_predrank2 = {}
segment_dict_predrank3 = {}
segment_dict_predrank4 = {}


breaklimit = 200
breakcount = 0

while len(os.listdir(timeslides_dir)) == 0:
	time.sleep(10)

print("Found a file, starting cleanup loop")

while True:
	
	files = os.listdir(timeslides_dir)
	if len(files) == 0:
		print("waiting a bit")
		time.sleep(1)
		breakcount += 1
		#files = os.listdir(timeslides_dir)
		if breakcount > breaklimit:
			print("breaking, we've hit our waiting limit.")
			break
		continue

	else:
		breakcount = 0

	for file in files:
		zl_file = None
		stats_file = None
		sys.stdout.flush()
		start = time.time()

		if injfile is None:
			if "stats" in file:
				
				#file_components = file.split("_")
				if "zerolags"+file[len("stats"):] in files:
					print("found file: {}".format(file))
					try:
						stats_file = np.load(os.path.join(timeslides_dir,file))
						zl_file = np.load(os.path.join(timeslides_dir,"zerolags"+file[len("stats"):]))
					except:
						print("Error loading files, they're probably not fully saved yet. Waiting...")
						time.sleep(1)
						continue		

		else:
			#no stats files in injection runs
			try:
				#file_components = file.split("_")
				zl_file = np.load(os.path.join(timeslides_dir,file))

			except:
				print("Error loading files, they're probably not fully saved yet. Waiting...")
				time.sleep(1)
				continue	
		

		if (zl_file is not None and stats_file is not None) or (injfile is not None and zl_file is not None):
			print(zl_file.shape)

			if injfile is None:
				print(stats_file.shape)
				#this is where we deal with the files

				#align the stats file with the zerolags file
				#NOTE: commented out in new implementation
				#stats_file = align_stats(zl_file, stats_file)

			#get the segment number
			seg_idx = int(file.split(".")[0].split("_")[-1])
			#check if the segment number is in the dictionary

			if seg_idx in segment_dict.keys():
				print("merging zerolags for segment {}".format(seg_idx))
				if injfile is None:
					segment_dict[seg_idx] = merge_zerolags_and_stats(segment_dict[seg_idx][0], zl_file, segment_dict[seg_idx][1], stats_file)
					segment_dict_predrank[seg_idx] = merge_zerolags_and_stats(segment_dict_predrank[seg_idx][0], zl_file, segment_dict_predrank[seg_idx][1], stats_file, merge_target = 6)
					segment_dict_predrank2[seg_idx] = merge_zerolags_and_stats(segment_dict_predrank2[seg_idx][0], zl_file, segment_dict_predrank2[seg_idx][1], stats_file, merge_target = 7)
					segment_dict_predrank3[seg_idx] = merge_zerolags_and_stats(segment_dict_predrank3[seg_idx][0], zl_file, segment_dict_predrank3[seg_idx][1], stats_file, merge_target = 8)
					segment_dict_predrank4[seg_idx] = merge_zerolags_and_stats(segment_dict_predrank4[seg_idx][0], zl_file, segment_dict_predrank4[seg_idx][1], stats_file, merge_target = 9)
				else:
					segment_dict[seg_idx] = merge_zerolags_and_stats(segment_dict[seg_idx][0], zl_file)
					segment_dict_predrank[seg_idx] = merge_zerolags_and_stats(segment_dict_predrank[seg_idx][0], zl_file, merge_target = 6)
					segment_dict_predrank2[seg_idx] = merge_zerolags_and_stats(segment_dict_predrank2[seg_idx][0], zl_file, merge_target = 7)
					segment_dict_predrank3[seg_idx] = merge_zerolags_and_stats(segment_dict_predrank3[seg_idx][0], zl_file, merge_target = 8)
					segment_dict_predrank4[seg_idx] = merge_zerolags_and_stats(segment_dict_predrank4[seg_idx][0], zl_file, merge_target = 9)


			else:
				print("creating entry {} in segment_dict".format(seg_idx))
				if injfile is None:
					segment_dict[seg_idx] = [np.copy(zl_file), np.copy(stats_file)]
					segment_dict_predrank[seg_idx] = [np.copy(zl_file), np.copy(stats_file)]
					segment_dict_predrank2[seg_idx] = [np.copy(zl_file), np.copy(stats_file)]
					segment_dict_predrank3[seg_idx] = [np.copy(zl_file), np.copy(stats_file)]
					segment_dict_predrank4[seg_idx] = [np.copy(zl_file), np.copy(stats_file)]
				else:
					segment_dict[seg_idx] = [np.copy(zl_file)]
					segment_dict_predrank[seg_idx] = [np.copy(zl_file)]
					segment_dict_predrank2[seg_idx] = [np.copy(zl_file)]
					segment_dict_predrank3[seg_idx] = [np.copy(zl_file)]
					segment_dict_predrank4[seg_idx] = [np.copy(zl_file)]


			#delete the files
			os.remove(os.path.join(timeslides_dir,file))
			if injfile is None:
				os.remove(os.path.join(timeslides_dir,"zerolags"+file[len("stats"):]))

			print("took {} seconds to process file".format(time.time()-start))


print("exited loop, saving files")

master_zerolag = segment_dict[0][0]
master_zerolag_predrank = segment_dict_predrank[0][0]
master_zerolag_predrank2 = segment_dict_predrank2[0][0]
master_zerolag_predrank3 = segment_dict_predrank3[0][0]
master_zerolag_predrank4 = segment_dict_predrank4[0][0]


if injfile is None:
	master_stats = segment_dict[0][1]
	master_stats_predrank = segment_dict_predrank[0][1]
	master_stats_predrank2 = segment_dict_predrank2[0][1]
	master_stats_predrank3 = segment_dict_predrank3[0][1]
	master_stats_predrank4 = segment_dict_predrank4[0][1]

for key in sorted(segment_dict.keys()):
	if key == 0:
		continue
	print("merging zerolags for segment {}".format(key))
	master_zerolag = np.concatenate((master_zerolag, segment_dict[key][0]), axis = 0)
	master_zerolag_predrank = np.concatenate((master_zerolag_predrank, segment_dict_predrank[key][0]), axis = 0)
	master_zerolag_predrank2 = np.concatenate((master_zerolag_predrank2, segment_dict_predrank2[key][0]), axis = 0)
	master_zerolag_predrank3 = np.concatenate((master_zerolag_predrank3, segment_dict_predrank3[key][0]), axis = 0)
	master_zerolag_predrank4 = np.concatenate((master_zerolag_predrank4, segment_dict_predrank4[key][0]), axis = 0)

	if injfile is None:
		master_stats = np.concatenate((master_stats, segment_dict[key][1]), axis = 0)
		master_stats_predrank = np.concatenate((master_stats_predrank, segment_dict_predrank[key][1]), axis = 0)
		master_stats_predrank2 = np.concatenate((master_stats_predrank2, segment_dict_predrank2[key][1]), axis = 0)
		master_stats_predrank3 = np.concatenate((master_stats_predrank3, segment_dict_predrank3[key][1]), axis = 0)
		master_stats_predrank4 = np.concatenate((master_stats_predrank4, segment_dict_predrank4[key][1]), axis = 0)


np.save(os.path.join(timeslides_dir, "zerolags_merged.npy"), master_zerolag)
np.save(os.path.join(timeslides_dir, "zerolags_predrank_merged.npy"), master_zerolag_predrank)
np.save(os.path.join(timeslides_dir, "zerolags_predrank2_merged.npy"), master_zerolag_predrank2)
np.save(os.path.join(timeslides_dir, "zerolags_predrank3_merged.npy"), master_zerolag_predrank3)
np.save(os.path.join(timeslides_dir, "zerolags_predrank4_merged.npy"), master_zerolag_predrank4)



if injfile is None:
	np.save(os.path.join(timeslides_dir, "stats_merged.npy"), master_stats)
	np.save(os.path.join(timeslides_dir, "stats_predrank_merged.npy"), master_stats_predrank)
	np.save(os.path.join(timeslides_dir, "stats_predrank2_merged.npy"), master_stats_predrank2)
	np.save(os.path.join(timeslides_dir, "stats_predrank3_merged.npy"), master_stats_predrank3)
	np.save(os.path.join(timeslides_dir, "stats_predrank4_merged.npy"), master_stats_predrank4)