"""Bounded packing/scattering with the serving script's historical order.

Window changes fastest, followed by trigger slot, timeslide and second.
Keeping request boundaries and padding unchanged matters even when only the
valid rows of the final response are saved.
"""

import numpy as np


class InferenceBatches:
    """Pack only one request at a time, without materializing all SNR windows."""

    def __init__(self, snr, *, inference_rate, stride, light_travel_time,
                 num_time_slides, num_triggers, seconds, window_length=2048):
        self.rate = int(inference_rate)
        self.slides = int(num_time_slides)
        self.triggers = int(num_triggers)
        self.seconds = int(seconds)
        self.stride = int(stride)
        self.light_travel_time = light_travel_time
        self.window_length = int(window_length)
        self.total = self.rate * self.triggers * self.slides * self.seconds
        # This is a read-only strided view, including for memory-mapped input.
        # Advanced indexing below copies only the rows in the current request.
        self.windows = np.lib.stride_tricks.sliding_window_view(
            snr, self.window_length, axis=-1
        )

    def coordinates(self, positions):
        """Return (window, trigger, slide, second) for scalar/vector positions."""
        window = positions % self.rate
        remaining = positions // self.rate
        trigger = remaining % self.triggers
        remaining = remaining // self.triggers
        slide = remaining % self.slides
        second = remaining // self.slides
        return window, trigger, slide, second

    def request_id(self, position):
        return "_".join(str(value) for value in self.coordinates(position))

    def position(self, request_id):
        window, trigger, slide, second = map(int, request_id.split("_"))
        return (((second * self.slides + slide) * self.triggers + trigger)
                * self.rate + window)

    def fill(self, position, send_h, send_l, send_dt):
        """Fill valid rows and return the next absolute request position.

        The legacy loop leaves the final batch's unused suffix at its previous
        values (zeros if this is the first batch). Do not reset that suffix or
        wrap around to the beginning of the segment.
        """
        stop = min(position + len(send_h), self.total)
        rows = stop - position
        window, trigger, slide, second = self.coordinates(
            np.arange(position, stop, dtype=np.int64)
        )
        offset = (window + 1) * self.stride
        h = self.windows[0, slide, second, trigger, offset]
        l = self.windows[1, slide, second, trigger, offset]
        send_h[:rows, :, 0] = h
        send_l[:rows, :, 0] = l
        # Compute argmax before converting to FP32, exactly as the old loop.
        # Integer true-division produces FP64 before assignment to the buffer.
        send_dt[:rows, 0] = (
            np.argmax(h, axis=-1) - np.argmax(l, axis=-1)
        ) / self.light_travel_time
        return stop

    def scatter(self, response_array, response_data, request_id):
        """Scatter one response, discarding the final batch's padding rows."""
        start = self.position(request_id)
        values = response_data.ravel()
        stop = min(start + len(values), self.total)
        window, trigger, slide, second = self.coordinates(
            np.arange(start, stop, dtype=np.int64)
        )
        response_array[slide, second, trigger, window] = values[:stop - start]


def store_prediction_column(timeslides, response_array, response_idx):
    """Keep the original strict SNR > 1 criterion, including NaN handling."""
    destination = timeslides[..., response_idx]
    destination.fill(-1000)
    np.copyto(destination, response_array, where=timeslides[..., 2] > 1,
              casting="unsafe")

