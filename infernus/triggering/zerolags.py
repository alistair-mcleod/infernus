from typing import List

import numpy as np


def get_best_zerolags(
    zerolags: List[List[float]],
    num_trigs: int
) -> List[List[float]]:
    zerolags_sorted = sorted(zerolags, key=lambda x: x[2], reverse=True)
    
    # Pad zerolags if required
    # Format is (h1_snr, l1_snr, coh_snr, h1_time_idx, l1_time_idx, template_idx)
    while len(zerolags_sorted) < num_trigs:
        zerolags_sorted.append([-1,-1,-1,-1,-1,-1])
    
    return zerolags_sorted[:num_trigs]


def get_buffer(
    data: np.ndarray,
    buffer_id: int,
    overlap: int,
    buffer_length: int
) -> (np.ndarray, np.ndarray):
    # if first sample, prepend zeros to temp_data
    if buffer_id == 0:
        temp_data = np.concatenate(
            (
                np.zeros((data.shape[0],2, overlap//2)).astype(np.float32),
                data[:,:,buffer_id:(buffer_id + buffer_length + overlap//2)]
            ),
            axis=2)
    # elif last sample, append zeros to temp_data
    elif buffer_id+buffer_length >= data.shape[2]:
        temp_data = np.concatenate(
            (
                data[:,:,(buffer_id - overlap//2):(buffer_id + buffer_length)],
                np.zeros((data.shape[0],2, overlap//2)).astype(np.float32)
            ),
            axis=2)
    # else no prepend/append needed
    else:
        temp_data = data[:,:,(buffer_id - overlap//2):(buffer_id + buffer_length + overlap//2)]
    temp_data_trimmed = temp_data[:,:,overlap//2:-overlap//2]
    
    return temp_data, temp_data_trimmed


def get_secondary(
    temp_data: np.ndarray,
    primary_arg_maxes: np.ndarray,
    offset: int,
    det_idx: int
) -> (np.ndarray, np.ndarray):
    secondary_slices = []
    for j in range(temp_data.shape[0]):
        minimum = primary_arg_maxes[j]-offset if primary_arg_maxes[j]-offset>=0 else 0
        maximum = primary_arg_maxes[j]+offset if primary_arg_maxes[j]+offset<temp_data.shape[2] else temp_data.shape[2]-1
        secondary_slices.append(np.arange(minimum, maximum+1))
    secondary_slices = np.copy(secondary_slices)
            
    if det_idx == 0:
        secondary_maxes = np.max(temp_data[:,1][np.arange(secondary_slices.shape[0])[:,None], secondary_slices], axis=1)
        secondary_arg_maxes = np.argmax(temp_data[:,1][np.arange(secondary_slices.shape[0])[:,None], secondary_slices], axis=1)
    else:
        secondary_maxes = np.max(temp_data[:,0][np.arange(secondary_slices.shape[0])[:,None], secondary_slices], axis=1)
        secondary_arg_maxes = np.argmax(temp_data[:,0][np.arange(secondary_slices.shape[0])[:,None], secondary_slices], axis=1)
    
    return secondary_maxes, secondary_arg_maxes


# This one will allow us to save many zerolags to temp_zerolags, and then we can edit the try statement to
# put multiple of them into `zerolags`. This may be desired for injection runs
# This is slower than just treating the max coherent SNR zerolag, so we don't use this method for background runs
def get_zerolags(
    data: str,
    snr_thresh: int = 4,
    offset: int = 20,
    buffer_length: int = 2048,
    overlap: int = int(0.2*2048),
    num_trigs: int = 1
) -> List[List[float]]:
    zerolags = []
    
    #Refactoring data to be in the form (n_templates, n_detectors, duration) rather than (n_detectors, n_templates, duration)
    data = np.swapaxes(data, 0, 1)
    # Loop across time series in buffers
    for i in np.arange(0, data.shape[2] - buffer_length + 1, step=buffer_length):
        temp_zerolags = []
        
        temp_data, temp_data_trimmed = get_buffer(data, i, overlap, buffer_length)
        
        det = [0, 1]  # Corresponding to [H1, L1]
        for key,val in enumerate(det):
            primary_arg_maxes = np.argmax(temp_data_trimmed, axis=2)[:,val] + overlap//2
            primary_maxes = np.max(temp_data_trimmed, axis=2)[:,val]
            secondary_maxes, secondary_arg_maxes = get_secondary(temp_data, primary_arg_maxes, offset, val)
            
            coh_snrs = np.sqrt(np.square(primary_maxes) + np.square(secondary_maxes))
            
            absolute_primary_arg_maxes = primary_arg_maxes - offset + i
            absolute_secondary_arg_maxes = secondary_arg_maxes + (primary_arg_maxes - offset) - offset + i
            coh_snr_max_arg = np.argmax(coh_snrs)
            if num_trigs > 1:
                coh_snr_max_args = np.argsort(-coh_snrs)[:num_trigs]
            else:
                coh_snr_max_args = [coh_snr_max_arg]
            
            # Format is (h1_snr, l1_snr, coh_snr, h1_time_idx, l1_time_idx, template_idx)
            if val == 0:
                for k in coh_snr_max_args:
                    if primary_maxes[k] >= snr_thresh:
                        temp_zerolags.append([
                            primary_maxes[k], secondary_maxes[k],
                            coh_snrs[k],
                            absolute_primary_arg_maxes[k], absolute_secondary_arg_maxes[k],
                            k
                        ])
#                         break
            else:
                for k in coh_snr_max_args:
                    if primary_maxes[k] >= snr_thresh:
                        temp = [
                            secondary_maxes[k], primary_maxes[k],
                            coh_snrs[k],
                            absolute_secondary_arg_maxes[k], absolute_primary_arg_maxes[k],
                            k
                        ]
                        if temp not in temp_zerolags:
                            temp_zerolags.append(temp)
        
        # Chooses best zerolags by maximum coherent SNR
        try:
            new_zerolag = get_best_zerolags(temp_zerolags, num_trigs)
            if new_zerolag not in zerolags or new_zerolag[0][0] == -1:  # Prevents duplicate zerolags
                zerolags.append(new_zerolag)
        except:
            continue
    
    return zerolags