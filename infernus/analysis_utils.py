from GWSamplegen.noise_utils import get_data_from_OzStar
#make a spectrogram out of the strain
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import highpass
from pycbc.types import TimeSeries
from astropy.time import Time
import numpy as np 

f_lower = 30



def gps_to_event(gps):
	"""
	Convert a GPS time to an event name in the format of YYMMDD_HHMMSS
	"""
	
	t = Time(gps, format='gps')
	return t.utc.strftime('%y%m%d_%H%M%S')

def GW_event_to_gps(event_name):
	"""
	Convert a GW event name in the format of GWYYMMDD_HHMMSS to a GPS time
	"""
	from astropy.time import Time
	year = int("20" + event_name[2:4])
	month = int(event_name[4:6])
	day = int(event_name[6:8])
	hour = int(event_name[9:11])
	minute = int(event_name[11:13])
	second = int(event_name[13:15])
	t = Time(f"{year}-{month}-{day} {hour}:{minute}:{second}", format='iso', scale='utc')
	return t.gps

def spectrogram_from_strain(data,before = 10, after = 5):
	sample_rate = 2048
	duration = len(data)
	#if data is not a PyCBC timeseries, convert it
	if not hasattr(data, 'sample_times'):
		data = TimeSeries(data, delta_t = 1/sample_rate)
	#print(data)
	zoom = highpass(data,f_lower).whiten(4,4, remove_corrupted = False, low_frequency_cutoff=f_lower)[duration//2 - sample_rate*(before): duration//2 + sample_rate*(after)]
	#print(zoom)
	times, freqs, power = zoom.qtransform(.005, logfsteps=200,qrange=(32,128),frange=(30, 600),)

	return times, freqs, power


def gps_to_segment_second(gps, valid_times, start_cutoff):
	if gps < valid_times[0] + start_cutoff:
		return -1, -1
	segment = np.max(np.where(gps > (valid_times+start_cutoff))[0])
	second = gps - valid_times[segment]
	return segment, second

def zerolag_to_gps(zerolag, valid_times):
	return valid_times[zerolag[0,3].astype(int)] + zerolag[0,4]


def spectrogram_from_zerolag(zerolag, valid_times, duration = 100, before = 10, after = 5, ifo = "H1"):
	gps = zerolag_to_gps(zerolag, valid_times)
	
	data = get_data_from_OzStar(gps - duration//2, duration, ifo)
	times, freqs, vals = spectrogram_from_strain(data, before = before, after = after)
	return times, freqs, vals, gps

def spectrogram_around_gps(gps, duration, before = 10, after = 5, ifo = "H1"):
	data = get_data_from_OzStar(gps - duration//2, duration, ifo)
	times, freqs, vals = spectrogram_from_strain(data, before = before, after = after)
	return times, freqs, vals