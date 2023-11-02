#standard imports
import os
import sys
import time
import numpy as np


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

def merge_zerolags(zl, new_zl, stats, new_stats):
	for i in range(len(zl)):

		if i >= len(new_zl): #TODO: investigate why this is happening
			print("zl is longer than new_zl")
			continue

		if zl[i][0][0] == -1 and new_zl[i][0][0] == -1:
			#print("both zerolags are -1 for zl {}".format(i))
			continue
	
		if zl[i][0][0] == -1:
			zl[i] = new_zl[i]
			stats[i] = new_stats[i]
			continue

		if new_zl[i][0][0] == -1:
			continue



		if zl[i][0][2] < new_zl[i][0][2]:
			zl[i] = new_zl[i]
			stats[i] = new_stats[i]

	return [zl, stats]






timeslides_dir = "/fred/oz016/alistair/infernus/timeslides/"
segment_dict = {}






breaklimit = 180
breakcount = 0

while True:
	#flush
	
	files = os.listdir(timeslides_dir)
	if len(files) == 0:
		print("waiting a bit")
		time.sleep(1)
		breakcount += 1
		continue
		#files = os.listdir(timeslides_dir)
		if breakcount > breaklimit:
			print("breaking, we've hit our waiting limit.")
			break
	else:
		breakcount = 0

	for file in files:
		zl_file = None
		stats_file = None
		sys.stdout.flush()
		start = time.time()
		if "stats" in file:
			
			file_components = file.split("_")
			if "zerolags"+file[len("stats"):] in files:
				print("found file: {}".format(file))
				try:
					stats_file = np.load(timeslides_dir+file)
					zl_file = np.load(timeslides_dir+"zerolags"+file[len("stats"):])
				except:
					print("Error loading files, they're probably not fully saved yet. Waiting...")
					time.sleep(1)
					continue		

		if zl_file is not None and stats_file is not None:
			print(zl_file.shape)
			print(stats_file.shape)
			#this is where we deal with the files

			#align the stats file with the zerolags file
			stats_file = align_stats(zl_file, stats_file)

			#get the segment number
			seg_idx = file.split(".")[0].split("_")[-1]
			#check if the segment number is in the dictionary

			if seg_idx in segment_dict.keys():
				print("merging zerolags for segment {}".format(seg_idx))
				segment_dict[seg_idx] = merge_zerolags(segment_dict[seg_idx][0], zl_file, segment_dict[seg_idx][1], stats_file)

			else:
				print("creating entry {} in segment_dict".format(seg_idx))
				segment_dict[seg_idx] = [zl_file, stats_file]

			#delete the files
			os.remove(timeslides_dir+file)
			os.remove(timeslides_dir+"zerolags"+file[len("stats"):])

			print("took {} seconds to process file".format(time.time()-start))


print("exited loop, saving files")

for key in segment_dict.keys():
	np.save("/fred/oz016/alistair/infernus/timeslides_merged/zerolags_{}.npy".format(key), segment_dict[key][0])
	np.save("/fred/oz016/alistair/infernus/timeslides_merged/stats_{}.npy".format(key), segment_dict[key][1])