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
import gc

import numpy as np

from triggering.zerolags import get_zerolags

#TODO: change to a better way of splitting models
from model_utils import split_models


def get_windows(start_end_indexes, peak_pos, pad=True):
    windows = []
    #print(peak_pos)
    #print(start_end_indexes[max(peak_pos//128-100,0)])
    #print(start_end_indexes[min(peak_pos//128+100, len(start_end_indexes))])
    for key,val in enumerate(start_end_indexes[max(peak_pos//128-100,0):min(peak_pos//128+100, len(start_end_indexes))]):
        if val[0] <= peak_pos <= val[1]:
            windows.append(key+max(peak_pos//128-100,0))

    #TODO: check if this is correct behaviour
    if len(windows) == 0:
        print("this zerolag is invalid due to insufficient windows")
        return windows
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
    sample_rate = 2048
    duration = 900
    light_travel_time = sample_rate//100  #max light travel time between H and L. In units of datapoints, not ms
    
    

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

    #max windowed_start_end_indexes =  1843200
    #windowed_sample_end_indexes = list(range(sample_rate-1, sample_rate*duration, sample_rate//args["inference_rate"]))
    #windowed_sample_start_indexes = list(np.copy(windowed_sample_end_indexes) - (sample_rate - 1))
    #start_end_indexes = list(zip(windowed_sample_start_indexes, windowed_sample_end_indexes))

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

        n_templates = SNR.shape[1]
        
        # Calculate zerolags
        zerolags = get_zerolags(
            data = SNR,
            snr_thresh = 4,
            offset = 20,
            buffer_length = 2048,
            overlap = int(0.2*2048),
            num_trigs = 1
        )

        if len(zerolags) < num_time_slides - 1: #TODO: check if we need the -1
            print("Not enough zerolags for this batch. Skipping.")
            os.remove('SNR_batch_{}_segment_{}.npy'.format(batch, segment))
            os.remove('preds_batch_{}_segment_{}.npy'.format(batch, segment))

            if batch == args['n_batches'] - 1 and segment == args['n_noise_segments'] - 1:
                print("main job should have finished")
                break
            elif segment == args['n_noise_segments'] - 1:
                template_start += n_templates
                batch += 1
                segment = 0
                print("batch is now", batch)
            else:
                segment += 1
                print("segment is now", segment)

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
        
        pstart = time.time()

        pred_array = []
        primary_dets = []
        secondary_dets = []
        zl_lens = []

        prestart = time.time()
        window_time = 0
        centrals = 0 
        secondary = 0
        argwindows = 0
        spreds = 0
        appends = 0

        #save the zerolags to disk
        print("saving to zerolag file: /fred/oz016/alistair/infernus/timeslides/zerolags_{}-{}_batch_{}_segment_{}.npy".\
                format(template_start, template_start + n_templates, batch, segment))
        np.save("/fred/oz016/alistair/infernus/timeslides/zerolags_{}-{}_batch_{}_segment_{}.npy".\
                format(template_start, template_start + n_templates, batch, segment), zerolags)

        print("there are {} zerolags".format(len(zerolags)))
        windowed_sample_end_indexes = list(range(sample_rate-1, SNR.shape[-1], sample_rate//args["inference_rate"]))
        windowed_sample_start_indexes = list(np.copy(windowed_sample_end_indexes) - (sample_rate - 1))
        start_end_indexes = list(zip(windowed_sample_start_indexes, windowed_sample_end_indexes))
        for key_i, i in enumerate(zerolags):
            # Check this zerolag is valid
            if i[0][0] == -1:
                #print(f"Zerolag {key_i} is invalid")
                continue

            #print(f"Processing zerolag {key_i} of {len(zerolags)}")
        
            
            # Determine primary and secondary detectors
            primary_det = np.argmax(i[0][:2])
            secondary_det = 1 if primary_det==0 else 0
            
            s = time.time()
            primary_det_pos = i[0][3+primary_det]

            
            primary_det_samples = get_windows(start_end_indexes, primary_det_pos)
            window_time += time.time() - s
            #add +2 to args[inference rate]
            if len(primary_det_samples) < args["inference_rate"]+2 or primary_det_samples[0] < 0 or primary_det_samples[-1] >= len(start_end_indexes):
                print(f"Not enough space either side to get full moving average predictions for primary detector in zerolag {key_i}:")
                print(i)
                zerolags[key_i][0][0] = -1
                continue
                
            s = time.time()
            # Load predictions of primary detector
            primary_preds = preds[primary_det][int(i[0][5]), primary_det_samples[0]:primary_det_samples[-1]+1]
            if len(primary_preds) != args["inference_rate"]+2:
                print("primary preds has wrong length")
                print(primary_preds)

            # Get central positions for secondary samples
            secondary_centrals = []
            temp = i[0][3+secondary_det]
            while len(secondary_centrals) < num_time_slides:  # Forward pass
                #TODO: investigate if this random shift can make the time shift go out of bounds. 
                #I have had different results when running this on the same data multiple times
                temp = temp + time_shift + np.random.randint(-time_shift_var, time_shift_var+1) 
                if temp + light_travel_time < start_end_indexes[-1][0]:
                    secondary_centrals.append(int(temp))
                else:
                    #print("Stepping in reverse now")
                    break
            temp = i[0][3+secondary_det]
            while len(secondary_centrals) < num_time_slides:  # Backwards pass
                temp = temp - time_shift + np.random.randint(-time_shift_var, time_shift_var+1)
                if temp - light_travel_time > start_end_indexes[0][1]:
                    secondary_centrals.append(int(temp))
                else:
                    print("Not enough forward and backward passes to get the desired number of time shifts.")
                    print("Please adjust the number of time shifts, and/or the time_shift length.")
                    print(f"This error is for zerolag {key_i}")
                    break 
            centrals += time.time() - s
            
            # Get max SNR in light travel time around secondary central positions
            # Append necessary stats to stats output list

                        
            zl_len = 0
            s = time.time()

            for j in sorted(secondary_centrals):
                t = time.time()
                peak_pos = j - light_travel_time + np.argmax(SNR[secondary_det, int(i[0][5]), j-light_travel_time:j+light_travel_time+1])

                secondary_det_samples = get_windows(start_end_indexes, peak_pos)
                if len(secondary_det_samples) == 0:
                    print("Red alert! a secondary detector has no windows. Hopefully setting the zerolag to -1 fixes it.")
                    zerolags[key_i][0][0] = -1
                    continue

                argwindows += time.time() - t
                # Load predictions of secondary detector
                t = time.time()
                secondary_preds = preds[secondary_det][int(i[0][5]), secondary_det_samples[0]:secondary_det_samples[-1]+1]
                if len(secondary_preds) < 18:
                    print("secondary preds has wrong length")
                    print("s. pred array: ",secondary_preds)
                    print("peak_pos: ", peak_pos)
                    print(secondary_det_samples)
                    print('zerolag', key_i)

                spreds += time.time() - t
                # PASS BOTH DETECTOR PREDICTIONS THROUGH COMBINING MODEL
                # For this dummy test, we just calculate the mean instead of passing it through a combining model
#                 combined_preds = np.mean([primary_preds, secondary_preds], axis=0)

                t = time.time()
                if primary_det == 0:
                    pred_array.append([primary_preds, secondary_preds])
                    #combined_preds = ifo_dict['combiner'].predict([primary_preds, secondary_preds], verbose = 2)
                else:
                    pred_array.append([secondary_preds, primary_preds])
                    #combined_preds = ifo_dict['combiner'].predict([secondary_preds, primary_preds], verbose = 2)
                appends += time.time() - t

                zl_len += 1
            zl_lens.append(zl_len)
            secondary += time.time() - s

            #once we get to here, we know the zerolag is valid, so we can append the primary and secondary detectors
            primary_dets.append(primary_det)
            secondary_dets.append(secondary_det)

        print("pre predictions took {} seconds".format(time.time() - prestart))
        print("window time", window_time)
        print("central time", centrals)
        print("secondary time", secondary)
        print("argwindows time", argwindows)
        print("spreds time", spreds)
        print("appends time", appends)
        print("secondary minus timed stuff:" , secondary - argwindows - spreds - appends)

        for p in range(len(pred_array)):
            if len(pred_array[p][0]) != 18:
                print("H pred array has wrong length, for zerolag {}".format(p))
                print(pred_array[p])
            if len(pred_array[p][1]) != 18:
                print("L pred array has wrong length, for zerolag {}".format(p))
                print(pred_array[p])

        #doing all the timeslides for a zerolag at once
        pred_array = np.array(pred_array)#.reshape(-1,2,ifo_pred_len)
        h_pred_array = pred_array[:,0].reshape(-1,ifo_pred_len)
        l_pred_array = pred_array[:,1].reshape(-1,ifo_pred_len)
        print("pred array shape", pred_array.shape)
        combined_preds = ifo_dict['combiner'].predict([h_pred_array, l_pred_array], 
                                                        verbose = 2, batch_size = 4096)
        
        combined_preds = combined_preds.reshape(-1, num_time_slides, 18)
        print("combined preds shape", combined_preds.shape)
        #replace with enumerate
        print("finished predictions, took {} seconds".format(time.time() - pstart))

        poststart = time.time()
        
        jsum = 0

        true_idx = 0

        for key_i, i in enumerate(zerolags):
            if i[0][0] == -1:
                print(f"Zerolag {key_i} is invalid")
                continue

            if true_idx > len(combined_preds):
                print("Ran out of predictions at zerolag {}".format(key_i))
                continue
            

            temp_stats = []

            for idx_j, j in enumerate(sorted(secondary_centrals)):
                peak = np.max(SNR[secondary_dets[true_idx], int(i[0][5]), j-light_travel_time:j+light_travel_time+1])
                # COMPUTE MOVING AVERAGE PREDICTIONS
                ma_prediction_16hz = []
                start = args["inference_rate"]
                while start <= len(combined_preds[true_idx][idx_j]):
                    ma_prediction_16hz.append(np.mean(combined_preds[true_idx][idx_j][:start]))
                    start += 1
                ma_prediction_16hz = max(ma_prediction_16hz)
                #print(f"MA pred: {ma_prediction_16hz}")

                # Format is (H1 SNR, L1 SNR, Network SNR, Moving Average Prediction)
                # Need individual detector SNRs so they can be combined and filtered properly with new batches of templates on the same noise data
                new_stat = [-1, -1, np.sqrt(np.square(i[0][primary_dets[true_idx]]) + np.square(peak)).astype(np.float32), ma_prediction_16hz, "ma_prediction_8hz", "ma_prediction_4hz", "ma_prediction_2hz"]
                new_stat[primary_dets[true_idx]] = i[0][primary_dets[true_idx]]
                new_stat[secondary_dets[true_idx]] = peak
                temp_stats.append(new_stat)
                jsum += 1
                
            true_idx += 1

            stats.append(temp_stats)

        
        print("post predictions took {} seconds".format(time.time() - poststart))

        #combopreds = np.array(combopreds)
#         np.save("/fred/oz016/alistair/infernus/timeslides/combopreds_templates_{}-{}_batch_{}_segment_{}.npy".\
#              format(template_start, template_start + n_templates, batch, segment), save_arr)
        #np.save("/fred/oz016/alistair/infernus/timeslides/zerolags_{}-{}_batch_{}_segment_{}.npy".\
        #        format(template_start, template_start + n_templates, batch, segment), zerolags)
        #np.save("/fred/oz016/alistair/infernus/timeslides/stats_{}-{}_batch_{}_segment_{}.npy".\
        #        format(template_start, template_start + n_templates, batch, segment), stats)

        os.remove('SNR_batch_{}_segment_{}.npy'.format(batch, segment))
        os.remove('preds_batch_{}_segment_{}.npy'.format(batch, segment))

        

        if batch == args['n_batches'] - 1 and segment == args['n_noise_segments'] - 1:
            print("main job should have finished")
            break
        elif segment == args['n_noise_segments'] - 1:
            template_start += n_templates
            batch += 1
            segment = 0
            print("batch is now", batch)
        else:
            segment += 1
            print("segment is now", segment)

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

