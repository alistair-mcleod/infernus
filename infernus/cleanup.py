"""Cleanup worker for background run jobs that does time shifts and saves 
zerolags and stats.

################################################################################
This cleanup worker attaches itself to a worker that generates SNR time series 
and gets the respective predictions. It then waits for the worker to finish 
saving the SNR time series and predictions before it continues with processing.

This cleanup worker reads in the SNR time series and predictions. It computes 
zerolags on the SNR time series, then for each zerolag it does 100 time shifts 
to extend the background. The overall process is as follows:
    1. Loops over zerolags produced on the loaded SNR time series.
    2. For each zerolag, it determines the 'primary' detector (highest SNR), 
       and gets that points position in time.
    3. Computes which windowed samples contain this point of SNR time series, 
       and loads the predictions corresponding to those windows.
    4. Samples points in time in the 'secondary' detector for time shifts. 
       Starts by iterating forwards in time from the primary detector maximum, 
       then stops and goes backwards in time from the start point if it can't 
       get enough time slides before the end of the batch.
    5. Loops over each time shift and then loads the predictions of the 
       secondary detector for that time shift.
    6. Pass primary and secondary detector predictions through combining model 
       for all inference rates from `inference_rate` down to 2 in factors of 2.
    7. Compute moving average predictions for all inference rates.
    8. Save tuples of detector SNRs, network SNR and moving average predictions 
       for all time shift triggers in a stats file, grouped by zerolags. Also 
       save the zerolags file.

This code is currently only implemented in the case that there is one aggregated 
zerolag per batch of templates (based on network SNR). If we wanted multiple 
zerolags that are then decided on model prediction, we would need to adapt the 
code to not only read from the first zerolag, and then determine which one will 
be used.
"""

import os
import json
import argparse
import time
import sys

import numpy as np

from triggering.zerolags import get_zerolags

#TODO: change to a better way of splitting models
from model_utils import split_models


def get_windows(start_end_indexes, peak_pos, pad=True):
    windows = []
    for key,val in enumerate(start_end_indexes):
        if val[0] <= peak_pos <= val[1]:
            windows.append(key)
    if pad:
        windows.append(windows[-1] + 1)
        windows.insert(0, windows[0] - 1)
    return windows


if __name__ == "__main__":
    ifo_dict = split_models()

    parser = argparse.ArgumentParser()
    parser.add_argument('--jobindex', type=int)
    parser.add_argument('--workerid', type=int, default=0)
    parser.add_argument('--totalworkers', type=int, default=1)
    parser.add_argument('--totaljobs', type=int, default=1)
    args = parser.parse_args()

    job_id = args.jobindex #job id in job array
    worker_id = args.workerid #worker number of a server
    n_workers = args.totalworkers
    n_jobs = args.totaljobs
    
    
    
    num_time_slides = 100
    time_shift = 2048
    time_shift_var = 256
    light_travel_time = 20  # In units of datapoints, not ms
    sample_rate = 2048
    duration = 800
    
    

    print("JOBFS", os.environ["JOBFS"] )

    myfolder = os.path.join(os.environ["JOBFS"], "job_" +str(job_id), "worker_"+str(worker_id))
    os.chdir(myfolder)
    print("my folder is", myfolder)

    print("starting job {} of {}".format(job_id, n_jobs))
    print("I am cleanup job {} of {} for this server".format(worker_id, n_workers))
    job_id = worker_id + job_id*n_workers
    print("my unique index is {}".format(job_id))
    n_jobs = n_jobs * n_workers
    print("there are {} jobs in total".format(n_jobs))

    while "args.json" not in os.listdir(myfolder):
        print("waiting for args.json")
        time.sleep(5)

    args = json.load(open("args.json", "r"))
    print("args are", args)
    
    template_start = args["template_start"]
    batch = 0  # template batch count
    segment = 0  # noise segment count
    
    windowed_sample_end_indexes = list(range(sample_rate-1, sample_rate*duration, sample_rate//args["inference_rate"]))
    windowed_sample_start_indexes = list(np.copy(list_of_windowed_sample_end_indexes) - (sample_rate - 1))
    start_end_indexes = list(zip(windowed_sample_start_indexes, windowed_sample_end_indexes))

    while True:
        files = os.listdir(myfolder)
        if 'SNR_batch_{}_segment_{}.npy'.format(batch, segment) in files \
              and 'preds_batch_{}_segment_{}.npy'.format(batch, segment) in files:
            print("Found files", files)

        else:
            print("no files found")
            sys.stdout.flush()
            time.sleep(10)
            continue

        SNR = np.load('SNR_batch_{}_segment_{}.npy'.format(batch, segment))
        preds = np.load('preds_batch_{}_segment_{}.npy'.format(batch, segment))
        
        # Calculate zerolags
        zerolags = get_zerolags(
            data = SNR,
            snr_thresh = 4,
            offset = 20,
            buffer_length = 2048,
            overlap = int(0.2*2048),
            num_trigs = 1
        )

        # REMOVE ZEROLAGS THAT CORRESPOND TO TIMES THAT REAL EVENTS OCCUR
        # do this by replacing those zerolags with [-1,-1,-1,-1,-1,-1]
        # zerolag format is (h1_snr, l1_snr, coh_snr, h1_time_idx, l1_time_idx, template_idx) where 
        # `h1_time_idx` and `l1_time_idx` are the data points (0 to 2048000 for 100s) and will need to be converted to GPS times (or convert the GPS times to data points)

        #split the preds along the last axis in half for H and L
        #preds has the shape (n_templates, n_windows, ifo_output*2)
        ifo_pred_len = preds.shape[2]//2
        H_preds = preds[:,:,:ifo_pred_len]
        L_preds = preds[:,:,ifo_pred_len:]
        preds = [H_preds, L_preds]

        n_windows = H_preds.shape[1]

        combopreds = []
        
        stats = []
        
        for key_i, i in enumerate(zerolags):
            # Check this zerolag is valid
            if i[0][0] == -1:
                continue
            
            temp_stats = []
            
            # Determine primary and secondary detectors
            primary_det = np.argmax(i[0][:2])
            secondary_det = 1 if primary_det==0 else 0
            
            primary_det_pos = i[0][3+primary_det]
            
            primary_det_samples = get_windows(start_end_indexes, primary_det_pos)
            
            if len(primary_det_samples) < args["inference_rate"] or primary_det_samples[0] < 0 or primary_det_samples[-1] >= len(start_end_indexes):
                print(f"Not enough space either side to get full moving average predictions for primary detector in zerolag {key_i}:")
                print(i)
                continue
                
            # Load predictions of primary detector
            primary_preds = preds[primary_det, int(i[0][5]), primary_det_samples[0]:primary_det_samples[-1]+1]
            
            # Get central positions for secondary samples
            secondary_centrals = []
            temp = i[0][3+secondary_det]
            while len(secondary_centrals) < num_time_slides:  # Forward pass
                temp = temp + shift + np.random.randint(-shift_random, shift_random+1)
                if temp < start_end_indexes[-1][0]:
                    secondary_centrals.append(int(temp))
                else:
                    print("Stepping in reverse now")
                    break
            temp = i[0][3+secondary_det]
            while len(secondary_centrals) < num_time_slides:  # Backwards pass
                temp = temp - shift + np.random.randint(-shift_random, shift_random+1)
                if temp > start_end_indexes[0][1]:
                    secondary_centrals.append(int(temp))
                else:
                    print("Not enough forward and backward passes to get the desired number of time shifts.")
                    print("Please adjust the number of time shifts, and/or the shift length.")
                    print(f"This error is for zerolag {key_i}")
                    break
            
            # Get max SNR in light travel time around secondary central positions
            # Append necessary stats to stats output list
            for j in sorted(secondary_centrals):
                peak = np.max(SNR[int(i[0][5]), secondary_det, j-offset:j+offset+1])
                peak_pos = j - offset + np.argmax(SNR[int(i[0][5]), secondary_det, j-offset:j+offset+1])

                secondary_det_samples = get_windows(start_end_indexes, peak_pos)

                # Load predictions of secondary detector
                secondary_preds = preds[secondary_det, int(i[0][5]), secondary_det_samples[0]:secondary_det_samples[-1]+1]

                # PASS BOTH DETECTOR PREDICTIONS THROUGH COMBINING MODEL
                # For this dummy test, we just calculate the mean instead of passing it through a combining model
#                 combined_preds = np.mean([primary_preds, secondary_preds], axis=0)
                if primary_det == 0:
                    combined_preds = ifo_dict['combiner'].predict([primary_preds, secondary_preds], verbose = 2)
                else:
                    combined_preds = ifo_dict['combiner'].predict([secondary_preds, primary_preds], verbose = 2)
        #         print(combined_preds)
        #         print(combined_preds[::2])
        #         print(combined_preds[::4])
        #         print(combined_preds[::8])


                # COMPUTE MOVING AVERAGE PREDICTIONS
                ma_prediction_16hz = []
                start = inference_rate
                while start <= len(combined_preds):
                    ma_prediction_16hz.append(np.mean(combined_preds[:start]))
                    start += 1
                ma_prediction_16hz = max(ma_prediction_16hz)
                print(f"MA pred: {ma_prediction_16hz}")

                # Format is (H1 SNR, L1 SNR, Network SNR, Moving Average Prediction)
                # Need individual detector SNRs so they can be combined and filtered properly with new batches of templates on the same noise data
                new_stat = [-1, -1, np.sqrt(np.square(i[0][primary_det]) + np.square(peak)).astype(np.float32), ma_prediction_16hz, "ma_prediction_8hz", "ma_prediction_4hz", "ma_prediction_2hz"]
                new_stat[primary_det] = i[0][primary_det]
                new_stat[secondary_det] = peak
                temp_stats.append(new_stat)

            stats.append(temp_stats)
        

#         save_arr = np.zeros((2, n_windows))

#         start = time.time()
#         L_roll = L_preds.copy()
#         #roll and concatenate L_preds 10 times
#         for i in range(1,100):
#             L_roll = np.concatenate((L_roll, np.roll(L_preds, i * n_windows//100, axis=1)), axis=0)

#         H_preds = np.tile(H_preds, (100,1,1))

#         print("finished rolling, took {} seconds".format(time.time() - start))

#         start = time.time()

#         n_templates = SNR.shape[1]
#         for j in range(n_templates):
#             #L_roll = np.roll(L_preds, i * n_windows//10, axis=1)
#             combopreds = ifo_dict['combiner'].predict([H_preds[j], L_roll[j]], batch_size = 4096, verbose = 2)
#             #for each window, only save the maximum prediction between templates
#             save_arr[0] = np.maximum(save_arr[0], combopreds[0])
#             #if we overwrite the maximum, we need to change the template index
#             save_arr[1] = np.where(save_arr[0] == combopreds[0], j, save_arr[1])


        print("finished timeslides, took {} seconds".format(time.time() - start))

        #combopreds = np.array(combopreds)
#         np.save("/fred/oz016/alistair/infernus/timeslides/combopreds_templates_{}-{}_batch_{}_segment_{}.npy".\
#              format(template_start, template_start + n_templates, batch, segment), save_arr)
        np.save("/fred/oz016/alistair/infernus/timeslides/zerolags_{}-{}_batch_{}_segment_{}.npy".\
                format(template_start, template_start + n_templates, batch, segment), zerolags)
        np.save("/fred/oz016/alistair/infernus/timeslides/stats_{}-{}_batch_{}_segment_{}.npy".\
                format(template_start, template_start + n_templates, batch, segment), zerolags)

        os.remove('SNR_batch_{}_segment_{}.npy'.format(batch, segment))
        os.remove('preds_batch_{}_segment_{}.npy'.format(batch, segment))

        template_start += n_templates

        if batch == args['n_batches'] - 1 and segment == args['n_noise_segments'] - 1:
            print("main job should have finished")
            break
        elif segment == args['n_noise_segments'] - 1:
            batch += 1
            segment = 0
        else:
            segment += 1

        """
        if len(files) > 0:

            print("Found files", files)
            i = files[0].split("_")[2]
            j = files[0].split("_")[4]
            print("i is", i)
            print("j is", j)
            os.remove(os.path.join(myfolder, files[0]))
            if i == 9 and j == 9:
                print("main job should have finished")
                break
        """

    print("finished doing cleanup")
    # for each file, load it, and get the best zerolags

    # save the best zerolags to a file

