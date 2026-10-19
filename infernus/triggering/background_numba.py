"""Optional compiled coincidence selection for cached float32 background SNRs.

Keep float32 rounding, first-template ties, H-before-L ties, and exact duplicate
suppression aligned with the NumPy selector.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, error_model="numpy")
def select_zerolags(data, primary_args, primary_maxes, seconds, buffer_length,
                   offset, shift_seconds, array_threshold, scalar_threshold,
                   both_above, max_ratio, ratio_enabled, output):
    """Fill float64 candidate rows from float32 amplitudes, in reference order."""
    length = data.shape[2]
    n_seconds = length // buffer_length
    n_templates = data.shape[1]
    # Fixed homogeneous tuples keep the reference's exact-value duplicate key
    # while avoiding Python lists/sets in the per-second loop.
    seen = set()
    for second_index in range(seconds.size):
        second = seconds[second_index]
        chosen_valid = False
        chosen_h = np.float32(-1)
        chosen_l = np.float32(-1)
        chosen_network = np.float32(-1)
        chosen_h_arg = -1
        chosen_l_arg = -1
        chosen_template = -1
        for detector in range(2):
            native_second = (second - shift_seconds * detector) % n_seconds
            other = 1 - detector
            best_network = np.float32(-1)
            best_primary = np.float32(-1)
            best_secondary = np.float32(-1)
            best_primary_arg = -1
            best_secondary_arg = -1
            best_template = -1
            for template in range(n_templates):
                primary = primary_maxes[detector, native_second, template]
                primary_arg = (primary_args[detector, native_second, template]
                               + second * buffer_length)
                secondary = np.float32(-np.inf)
                secondary_arg = primary_arg - offset
                # Padding belongs to the rolled segment; only in-range L1
                # reads wrap back to their native, unrolled coordinates.
                first_position = primary_arg - offset
                native = (first_position - shift_seconds * buffer_length * other) % length
                for position in range(first_position, primary_arg + offset + 1):
                    if position < 0 or position >= length:
                        value = np.float32(0)
                    else:
                        value = data[other, template, native]
                    # Strict comparison retains NumPy argmax's first tie.
                    if value > secondary:
                        secondary = value
                        secondary_arg = position
                    native += 1
                    if native == length:
                        native = 0
                # Explicit float32 intermediates reproduce NumPy's square,
                # addition and sqrt rounding; fastmath and parallelism are off.
                primary_square = np.float32(primary * primary)
                secondary_square = np.float32(secondary * secondary)
                network = np.float32(np.sqrt(np.float32(primary_square + secondary_square)))
                if ratio_enabled:
                    ratio = np.float32(primary / secondary)
                    if ratio < np.float32(1):
                        ratio = np.float32(np.float32(1) / ratio)
                    if ratio > max_ratio:
                        network = np.float32(0)
                if both_above and (primary < array_threshold or secondary < array_threshold):
                    network = np.float32(0)
                if network > best_network:
                    best_network = network
                    best_primary = primary
                    best_secondary = secondary
                    best_primary_arg = primary_arg
                    best_secondary_arg = secondary_arg
                    best_template = template
            valid = (best_network != np.float32(0)
                     and best_primary >= scalar_threshold
                     and (not both_above or best_secondary >= scalar_threshold))
            # H is evaluated first; only strictly higher L replaces it.
            # Identical H/L candidates therefore need no separate list lookup.
            if valid and (not chosen_valid or best_network > chosen_network):
                chosen_valid = True
                chosen_network = best_network
                chosen_template = best_template
                if detector == 0:
                    chosen_h, chosen_l = best_primary, best_secondary
                    chosen_h_arg, chosen_l_arg = best_primary_arg, best_secondary_arg
                else:
                    chosen_h, chosen_l = best_secondary, best_primary
                    chosen_h_arg, chosen_l_arg = best_secondary_arg, best_primary_arg
        if chosen_valid:
            key = (np.float64(chosen_h), np.float64(chosen_l),
                   np.float64(chosen_network), np.float64(chosen_h_arg),
                   np.float64(chosen_l_arg), np.float64(chosen_template))
            if key not in seen or chosen_h == np.float32(-1):
                for column in range(6):
                    output[second_index, 0, column] = key[column]
                seen.add(key)

