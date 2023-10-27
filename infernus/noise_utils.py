import numpy as np


def noise_generator(valid_times, paths, file_list, duration, sample_rate):
	file_idx = 0
	
	for i in range(len(valid_times)):
		#print(i)
		if i == 0:
			noise = np.load(file_list[file_idx])
		
		if valid_times[i] + duration > int(paths[file_idx][1]) + int(paths[file_idx][2]):
			file_idx += 1
			print("loading new file")
			noise = np.load(file_list[file_idx])
			
		if int(paths[file_idx][1]) <= valid_times[i]:
			#print("start time good")
			if int(paths[file_idx][1]) + int(paths[file_idx][2]) >= valid_times[i] + duration:
				start_idx = int((valid_times[i] - int(paths[file_idx][1])) * sample_rate)
				end_idx = int(start_idx + duration * sample_rate)
				#print(start_idx, end_idx)
				yield noise[:,start_idx:end_idx]