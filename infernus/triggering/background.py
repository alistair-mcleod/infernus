"""Background trigger selection without repeatedly rolling the full SNR array.

The cache preserves the original odd-overlap slice, which drops the last sample
of each primary window, and the original candidate ordering.
"""

import numpy as np

from .zerolags import get_best_zerolags


class BackgroundTimeslideCache:
    """Cache per-second primary peaks for one immutable template batch.

    ``shift_seconds`` counts positive rolls of detector L by ``buffer_length``
    samples.  Only coincidence neighborhoods and selected output windows are
    gathered; the full SNR data is never rolled on the optimized path.
    If Numba is installed, finite float32 batches use a compiled selector.
    Set ``use_numba=False`` to use the NumPy selector instead.

    ``data`` must contain finite amplitudes with shape (2, templates, samples),
    at least one template and an integer number of buffers (at least two).
    ``chop_time`` is a
    nonnegative whole-buffer sample offset. The overlap must satisfy
    ``0 <= offset <= overlap // 2 <= buffer_length`` with a nonzero half-overlap.
    Inputs remain unchanged for the lifetime of this cache.
    """

    def __init__(self, data, buffer_length=2048, overlap=int(0.2 * 2048),
                 offset=20, chop_time=0, *, use_numba=True):
        self.data = data
        self.buffer_length = buffer_length
        self.overlap = overlap
        self.offset = offset
        self.chop_time = chop_time
        self.length = data.shape[-1]
        self._compiled_select = None
        self.n_seconds = self.length // buffer_length
        blocks = data.reshape(2, data.shape[1], self.n_seconds, buffer_length)
        # get_buffer uses [overlap//2:-overlap//2], where the negative
        # floor division removes one extra sample when overlap is odd.
        primary_blocks = blocks[..., :buffer_length - overlap % 2]
        self.primary_args = np.argmax(primary_blocks, axis=-1).transpose(0, 2, 1)
        self.primary_maxes = np.max(primary_blocks, axis=-1).transpose(0, 2, 1)
        self.seconds = np.arange(chop_time // buffer_length, self.n_seconds)
        self.template_indices = np.arange(data.shape[1])
        self.neighborhood = np.arange(-offset, offset + 1)
        if use_numba and data.dtype == np.float32:
            try:
                from .background_numba import select_zerolags
            except ImportError:
                pass
            else:
                self._compiled_select = select_zerolags

    def get_zerolags(self, shift_seconds, snr_thresh=4,
                    both_detectors_above_thresh=False, max_snr_ratio=False):
        """Select one trigger per second for an integer shift and finite cuts."""
        if self._compiled_select is not None and self.seconds.size:
            # NumPy 1 and 2 promote Python and NumPy scalars differently.
            # Array cuts and the final scalar cut also differ in NumPy 1.
            array_threshold = np.float64(np.asarray(
                snr_thresh, dtype=np.result_type(np.dtype(np.float32), snr_thresh)))
            scalar_threshold = np.float64(np.asarray(
                snr_thresh, dtype=np.result_type(np.float32(0), snr_thresh)))
            ratio_limit = np.float64(np.asarray(
                max_snr_ratio, dtype=np.result_type(np.dtype(np.float32), max_snr_ratio)))
            output = np.full((self.seconds.size, 1, 6), -1, dtype=np.float64)
            self._compiled_select(
                self.data, self.primary_args, self.primary_maxes, self.seconds,
                self.buffer_length, self.offset, shift_seconds % self.n_seconds,
                array_threshold, scalar_threshold,
                bool(both_detectors_above_thresh), ratio_limit,
                bool(max_snr_ratio), output,
            )
            return output

        candidates = []
        for detector in (0, 1):
            native_seconds = (self.seconds - (shift_seconds if detector else 0)) % self.n_seconds
            primary_args = self.primary_args[detector, native_seconds]
            primary_maxes = self.primary_maxes[detector, native_seconds]
            absolute_primary = primary_args + self.seconds[:, None] * self.buffer_length
            positions = absolute_primary[..., None] + self.neighborhood
            other = 1 - detector
            native_positions = (positions - (shift_seconds * self.buffer_length if other else 0)) % self.length
            samples = self.data[other, self.template_indices[None, :, None], native_positions]
            # Padding is at the rolled segment's boundaries, not at the
            # unrolled L coordinates, which wrap around independently.
            samples[(positions < 0) | (positions >= self.length)] = 0
            secondary_maxes = np.max(samples, axis=-1)
            secondary_args = np.argmax(samples, axis=-1)
            coh_snrs = np.sqrt(np.square(primary_maxes) + np.square(secondary_maxes))
            if max_snr_ratio:
                ratio = primary_maxes / secondary_maxes
                ratio[ratio < 1] = 1 / ratio[ratio < 1]
                coh_snrs[ratio > max_snr_ratio] = 0
            if both_detectors_above_thresh:
                coh_snrs[(primary_maxes < snr_thresh) | (secondary_maxes < snr_thresh)] = 0
            absolute_secondary = secondary_args + absolute_primary - self.offset
            best_templates = np.argmax(coh_snrs, axis=1)
            candidates.append((primary_maxes, secondary_maxes, coh_snrs,
                               absolute_primary, absolute_secondary, best_templates))

        zerolags = []
        seen = set()
        for second in range(len(self.seconds)):
            temp_zerolags = []
            for detector, candidate in enumerate(candidates):
                primary, secondary, coherent, primary_arg, secondary_arg, best = candidate
                k = best[second]
                if (coherent[second, k] != 0 and primary[second, k] >= snr_thresh
                        and (not both_detectors_above_thresh or secondary[second, k] >= snr_thresh)):
                    trigger = [primary[second, k], secondary[second, k], coherent[second, k],
                               primary_arg[second, k], secondary_arg[second, k], k]
                    if detector:
                        trigger = [trigger[1], trigger[0], trigger[2], trigger[4], trigger[3], k]
                    if not detector or trigger not in temp_zerolags:
                        temp_zerolags.append(trigger)
            # Keep the reference's stable H-before-L ordering on equal SNR.
            new_zerolag = get_best_zerolags(temp_zerolags, 1)
            key = tuple(new_zerolag[0])
            if key not in seen or new_zerolag[0][0] == -1:
                zerolags.append(new_zerolag)
                seen.add(key)
            else:
                zerolags.append([[-1, -1, -1, -1, -1, -1]])
        return zerolags

    def window(self, template_id, start, end, shift_seconds):
        """Copy an in-bounds H/L window with the reference's positive L roll."""
        output = np.empty((2, end - start), dtype=self.data.dtype)
        output[0] = self.data[0, template_id, start:end]
        native_start = (start - shift_seconds * self.buffer_length) % self.length
        first_length = min(end - start, self.length - native_start)
        output[1, :first_length] = self.data[1, template_id, native_start:native_start + first_length]
        output[1, first_length:] = self.data[1, template_id, :end - start - first_length]
        return output
