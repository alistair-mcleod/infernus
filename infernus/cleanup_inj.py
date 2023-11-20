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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import argparse
import time
import sys
import gc
import numpy as np

from triggering.zerolags import get_zerolags

#TODO: change to a better way of splitting models
from model_utils import split_models, new_split_models

tf_model='/fred/oz016/alistair/BNS_models/real_glitch_metamodel/log_auc11.h5'
double_det, combiner = new_split_models(tf_model)

np.random.seed(1234)



def get_windows(start_end_indexes, peak_pos, pad=True, stride = 128):
    windows = []
    #print(peak_pos)
    #print(start_end_indexes[max(peak_pos//128-100,0)])
    #print(start_end_indexes[min(peak_pos//128+100, len(start_end_indexes))])
    peak_idx = int(peak_pos//stride) #the index of the peak pos
    buf = int(2*sample_rate//stride) #a range of windows to search around
    #print(int(max(peak_idx - buf,0)))
    #print(int(min(peak_idx + buf, len(start_end_indexes))))
    for key,val in enumerate(start_end_indexes[max(peak_idx - buf,0):min(peak_idx + buf, len(start_end_indexes))]):
        if val[0] <= peak_pos <= val[1]:
            windows.append(key+int(max(peak_idx-buf,0)))

    #TODO: check if this is correct behaviour
    if len(windows) == 0:
        print("this zerolag is invalid due to insufficient windows")
        return windows
    if windows[0] < 0 or windows[-1] >= len(start_end_indexes):
        print("this zerolag is invalid due to windows out of range")
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
    parser.add_argument('--savedir', type=str, default="/fred/oz016/alistair/infernus/timeslides/")
    args = parser.parse_args()

    job_id = args.jobindex #job id in job array
    worker_id = args.workerid #worker number of a server
    n_workers = args.totalworkers
    n_jobs = args.totaljobs
    savedir = args.savedir
    
    
    num_time_slides = 100
    time_shift = 2048
    time_shift_var = 256
    sample_rate = 2048
    duration = 900
    light_travel_time = sample_rate//100  #max light travel time between H and L. In units of datapoints, not ms
    
    

    print("JOBFS", os.environ["JOBFS"] )

    statusfolder = os.path.join(os.environ["JOBFS"], "job_" +str(job_id), "completed") #this folder is used to shut down the triton server
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

    injs = True if args["injection_file"] == 1 else False
    print("injs is", injs)
    if injs:
        num_time_slides = 1


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

        #if n_templates != SNR.shape[1]:
        #    print("This batch of SNR has a different number of templates ({} vs {})! This is the correct behaviour it's the last batch...".format(SNR.shape[1], n_templates))
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

        zerolags = np.array(zerolags)
        zerolags = np.concatenate((zerolags, np.zeros((zerolags.shape[0], zerolags.shape[1], 8)) - 1), axis = -1)

        if len(zerolags) < num_time_slides - 1 and not injs: #TODO: check if we need the -1
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
            continue

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
            
        pstart = time.time()

        pred_array = []
        preds_8hz = []
        preds_4hz = []
        preds_2hz = []


        primary_dets = []
        secondary_dets = []
        secondary_centrals_list = []

        prestart = time.time()
        window_time = 0
        centrals = 0 
        secondary = 0
        argwindows = 0
        spreds = 0
        appends = 0



        print("there are {} zerolags".format(len(zerolags)))
        windowed_sample_end_indexes = list(range(sample_rate-1, SNR.shape[-1], sample_rate//args["inference_rate"]))
        windowed_sample_start_indexes = list(np.copy(windowed_sample_end_indexes) - (sample_rate - 1))
        start_end_indexes = list(zip(windowed_sample_start_indexes, windowed_sample_end_indexes))

        #print(start_end_indexes[:10])
        for key_i, i in enumerate(zerolags):
            # Check this zerolag is valid
            if i[0][0] == -1:
                #print(f"Zerolag {key_i} is invalid")

                continue

            #print(f"Processing zerolag {key_i} of {len(zerolags)}")
        
            
            # Determine primary and secondary detectors
            primary_det = np.argmax(i[0][:2])
            
            s = time.time()
            primary_det_pos = i[0][3+primary_det]

            
            primary_det_samples = np.array(get_windows(start_end_indexes, primary_det_pos))
            primary_det_8hz_samples = np.array(get_windows(start_end_indexes[::2], primary_det_pos, stride = 256)) *2
            primary_det_4hz_samples = np.array(get_windows(start_end_indexes[::4], primary_det_pos, stride = 512)) *4
            primary_det_2hz_samples = np.array(get_windows(start_end_indexes[::8], primary_det_pos, stride = 1024)) *8

            window_time += time.time() - s
            #add +2 to args[inference rate]
            if len(primary_det_samples) < args["inference_rate"]+2 or primary_det_samples[0] < 0 or primary_det_samples[-1] >= len(start_end_indexes):
                print(f"Not enough space either side to get full moving average predictions for primary detector in zerolag {key_i}:")
                print(i)
                zerolags[key_i][0][0] = -1
                continue
            if len(primary_det_2hz_samples) < 4 or primary_det_2hz_samples[0] < 0 or primary_det_2hz_samples[-1] >= len(start_end_indexes):
                print("looks like it messed up on the 2 Hz sample")
                print(i)
                zerolags[key_i][0][0] = -1
                continue

            if primary_det_8hz_samples[-1] >= len(start_end_indexes):
                print("8 Hz samples out of range")
            
            if primary_det_4hz_samples[-1] >= len(start_end_indexes):
                print("4 Hz samples out of range")
                
            s = time.time()
            # Load predictions of primary detector
            primary_preds = preds[primary_det][int(i[0][5]), primary_det_samples]
            primary_8hz = preds[primary_det][int(i[0][5]), primary_det_8hz_samples]
            primary_4hz = preds[primary_det][int(i[0][5]), primary_det_4hz_samples]
            primary_2hz = preds[primary_det][int(i[0][5]), primary_det_2hz_samples]

            secondary_det = (1 + primary_det) %2

            secondary_preds = preds[secondary_det][int(i[0][5]), primary_det_samples]
            secondary_8hz = preds[secondary_det][int(i[0][5]), primary_det_8hz_samples]
            secondary_4hz = preds[secondary_det][int(i[0][5]), primary_det_4hz_samples]
            secondary_2hz = preds[secondary_det][int(i[0][5]), primary_det_2hz_samples]


            t = time.time()
            if primary_det == 0:
                pred_array.append([primary_preds, secondary_preds])
                preds_8hz.append([primary_8hz, secondary_8hz])
                preds_4hz.append([primary_4hz, secondary_4hz])
                preds_2hz.append([primary_2hz, secondary_2hz])
            else:
                pred_array.append([secondary_preds, primary_preds])
                preds_8hz.append([secondary_8hz, primary_8hz])
                preds_4hz.append([secondary_4hz, primary_4hz])
                preds_2hz.append([secondary_2hz, primary_2hz])

                appends += time.time() - t

            secondary += time.time() - s

            #once we get to here, we know the zerolag is valid, so we can append the primary and secondary detectors
            primary_dets.append(primary_det)
            secondary_dets.append(secondary_det)
            #we also need to save the secondary centrals
            #print("secondary centrals for this zerolag:", sorted(secondary_centrals))


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

        preds_8hz_h = np.array(preds_8hz)[:,0].reshape(-1,ifo_pred_len)
        preds_8hz_l = np.array(preds_8hz)[:,1].reshape(-1,ifo_pred_len)

        preds_4hz_h = np.array(preds_4hz)[:,0].reshape(-1,ifo_pred_len)
        preds_4hz_l = np.array(preds_4hz)[:,1].reshape(-1,ifo_pred_len)

        preds_2hz_h = np.array(preds_2hz)[:,0].reshape(-1,ifo_pred_len)
        preds_2hz_l = np.array(preds_2hz)[:,1].reshape(-1,ifo_pred_len)

        print("pred array shape", pred_array.shape)
        #combined_preds = ifo_dict['combiner'].predict([h_pred_array, l_pred_array], 
        #                                                verbose = 2, batch_size = 4096)
        combined_preds = combiner.predict([h_pred_array, l_pred_array], 
                                                        verbose = 2, batch_size = 4096)
        
        combined_8hz = combiner.predict([preds_8hz_h, preds_8hz_l],
                                                        verbose = 2, batch_size = 4096)
        
        combined_4hz = combiner.predict([preds_4hz_h, preds_4hz_l],
                                                        verbose = 2, batch_size = 4096)
        
        combined_2hz = combiner.predict([preds_2hz_h, preds_2hz_l],
                                                        verbose = 2, batch_size = 4096)
        
        
        combined_preds = combined_preds.reshape(-1, num_time_slides, 18)
        combined_8hz = combined_8hz.reshape(-1, num_time_slides, 10)
        combined_4hz = combined_4hz.reshape(-1, num_time_slides, 6)
        combined_2hz = combined_2hz.reshape(-1, num_time_slides, 4)

        print("combined preds shape", combined_preds.shape)
        print("combined 8hz shape", combined_8hz.shape)
        print("combined 4hz shape", combined_4hz.shape)
        print("combined 2hz shape", combined_2hz.shape)

        #replace with enumerate
        print("finished predictions, took {} seconds".format(time.time() - pstart))

        poststart = time.time()
        
        true_idx = 0

        for key_i, i in enumerate(zerolags):
            if i[0][0] == -1:
                #print(f"Zerolag {key_i} is invalid")
                #TODO: do more elegantly
                #np.concatenate((zerolags[key_i][0], np.zeros(8,)-1), axis = 0)

                continue

            if true_idx > len(combined_preds):
                print("Ran out of predictions at zerolag {}".format(key_i))
                continue
            

            temp_stats = []

            # COMPUTE MOVING AVERAGE PREDICTIONS
            ma_prediction_2hz = []
            ma_prediction_4hz = []
            ma_prediction_8hz = []
            ma_prediction_16hz = []
            start = args["inference_rate"]
            while start <= len(combined_preds[true_idx][0]):
                ma_prediction_16hz.append(np.mean(combined_preds[true_idx][0][:start]))
                start += 1
            start = args["inference_rate"]//2
            while start <= len(combined_8hz[true_idx][0]):
                ma_prediction_8hz.append(np.mean(combined_8hz[true_idx][0][:start]))
                start += 1
            start = args["inference_rate"]//4
            while start <= len(combined_4hz[true_idx][0]):
                ma_prediction_4hz.append(np.mean(combined_4hz[true_idx][0][:start]))
                start += 1
            start = args["inference_rate"]//8
            while start <= len(combined_2hz[true_idx][0]):
                ma_prediction_2hz.append(np.mean(combined_2hz[true_idx][0][:start]))
                start += 1
            ma_prediction_2hz = max(ma_prediction_2hz)
            ma_prediction_4hz = max(ma_prediction_4hz)
            ma_prediction_8hz = max(ma_prediction_8hz)
            ma_prediction_16hz = max(ma_prediction_16hz)

            pred_stuff = np.array([ma_prediction_16hz, ma_prediction_8hz, ma_prediction_4hz, ma_prediction_2hz,
                                    np.max(combined_preds[true_idx][0][:start]), np.max(combined_8hz[true_idx][0][:start]),
                                    np.max(combined_4hz[true_idx][0][:start]), np.max(combined_2hz[true_idx][0][:start])])

            #print(pred_stuff)
            #print(zerolags[key_i][0])
            
            zerolags[key_i][0][-8:] = pred_stuff
            
            #zerolags[key_i][0].append(ma_prediction_16hz)
            #zerolags[key_i][0].append(ma_prediction_8hz)
            #zerolags[key_i][0].append(ma_prediction_4hz)
            #zerolags[key_i][0].append(ma_prediction_2hz)

            #zerolags[key_i][0].append(combined_preds[true_idx][0][:start])
            #zerolags[key_i][0].append(combined_8hz[true_idx][0][:start])
            #zerolags[key_i][0].append(combined_4hz[true_idx][0][:start])
            #zerolags[key_i][0].append(combined_2hz[true_idx][0][:start])

                
            true_idx += 1

        #adjust template IDs to global template IDs
        zerolags[:,0,5] += template_start
        
        print("post predictions took {} seconds".format(time.time() - poststart))

        #save the zerolags to disk
        print("saving to zerolag file: zerolags_{}-{}_batch_{}_segment_{}.npy".\
                format(template_start, template_start + n_templates, batch, segment))
        
        np.save(os.path.join(savedir, "zerolags_{}-{}_batch_{}_segment_{}.npy".\
                format(template_start, template_start + n_templates, batch, segment)), zerolags)

        os.remove('SNR_batch_{}_segment_{}.npy'.format(batch, segment))
        os.remove('preds_batch_{}_segment_{}.npy'.format(batch, segment))

        time.sleep(1)
        del SNR, zerolags
        gc.collect()

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


    print("finished doing cleanup")
    # for each file, load it, and get the best zerolags

    # save the best zerolags to a file

#write a file to the status folder to indicate that this job is done
with open(os.path.join(statusfolder, "worker_{}.txt".format(worker_id)), "w") as f:
	f.write("done")