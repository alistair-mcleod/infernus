"""Reusable strict threshold counts for real numeric arrays."""

import numpy as np


class SortedThresholdCounts:
    """Own a score snapshot and reuse each promoted-dtype sort across queries.

    The snapshot protects subsequent results from caller mutation. Sorted
    copies are created lazily, once per dtype chosen by NumPy scalar
    promotion rules. NaNs are excluded from searches, but ``size`` retains all
    input elements so the caller can preserve its original denominator.
    Instances are intended for one statistics worker, not concurrent writers.
    """

    def __init__(self, values):
        values = np.asarray(values)
        self._values = np.array(values, copy=True).ravel()
        self._values.setflags(write=False)
        self._sorted = {}

    @property
    def size(self):
        """Total input count, including NaNs; no statistical normalization."""
        return int(self._values.size)

    def _ordered(self, dtype):
        if dtype not in self._sorted:
            values = self._values.astype(dtype, copy=False)
            ordered = np.sort(values[~np.isnan(values)])
            ordered.setflags(write=False)
            self._sorted[dtype] = ordered
        return self._sorted[dtype]

    def counts(self, thresholds, comparison):
        """Return int64 counts with the exact shape of ``thresholds``."""
        side = {'lt': 'left', 'gt': 'right'}[comparison]
        # A single query needs no sort. Comparing the original scalar also
        # preserves NumPy's handling of Python scalars and mixed integer types.
        if np.ndim(thresholds) == 0:
            compare = np.less if comparison == 'lt' else np.greater
            return np.asarray(np.count_nonzero(compare(self._values, thresholds)), dtype=np.int64)
        thresholds = np.asarray(thresholds)
        flat = thresholds.ravel()
        counts = np.zeros(flat.size, dtype=np.int64)
        if not flat.size or not self._values.size:
            return counts.reshape(thresholds.shape)
        groups = {}
        if self._values.dtype == flat.dtype:
            groups[self._values.dtype.str] = np.arange(flat.size)
        else:
            for index, threshold in enumerate(flat):
                dtype = np.result_type(self._values, threshold)
                groups.setdefault(dtype.str, []).append(index)
        for dtype, indices in groups.items():
            indices = np.asarray(indices, dtype=np.intp)
            ordered = self._ordered(dtype)
            queries = flat[indices].astype(dtype, copy=False)
            result = np.searchsorted(ordered, queries, side=side)
            if comparison == 'gt':
                result = ordered.size - result
            result[np.isnan(queries)] = 0
            counts[indices] = result
        return counts.reshape(thresholds.shape)


def strict_counts(values, thresholds, comparison):
    """Count ``values < threshold`` (lt) or ``values > threshold`` (gt).

    Queries preserve NumPy's array-versus-scalar promotion, strict ties, and
    NaN behavior. Use SortedThresholdCounts to reuse a sort across query sets.
    """
    return SortedThresholdCounts(values).counts(thresholds, comparison)
