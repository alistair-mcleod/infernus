import numpy as np
from pycbc.waveform import get_fd_waveform
from .SNR_utils import make_windows_2d
from .model_utils import split_model_stack

from pycbc.filter import resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from gwpy.timeseries import TimeSeries as GWPYTimeSeries
from GWSamplegen.waveform_utils import load_pycbc_templates, choose_templates_new
from GWSamplegen.snr_utils_np import np_get_cutoff_indices, mf_in_place, np_sigmasq

#import matplotlib.pyplot as plt
import keras


def load_event_data(mergertime, ifo):
	data = GWPYTimeSeries.fetch_open_data(ifo, mergertime-512, mergertime+512, version = 1)
	data = data.to_pycbc()

	return data

def load_event_spectrogram(data, duration, f_lower = 30):
	sample_rate = 4096
	before_after = 10
	zoom = highpass(data,f_lower).whiten(4,4, remove_corrupted = False)[duration*sample_rate//2 - sample_rate*(before_after + 20 ): duration*sample_rate//2 + sample_rate*(before_after)]
	times, freqs, power = zoom.qtransform(.001, logfsteps=1000,qrange=(32,128),frange=(30, 512),)
	return times, freqs, power


def real_event_plot(merger, model_path, ifos=["H1", "L1"], f_lower = 30, template = None):
	mergertime = merger.time
	duration = 1024
	sample_rate = 2048
	delta_t = 1/sample_rate
	n_templates = 5
	delta_f = 1/duration
	f_final = sample_rate//2

	print("Producing spectrogram...")
	if "L1" in ifos:
		data = load_event_data(mergertime, "L1")

	else:
		data = load_event_data(mergertime, "H1")

	times,freqs,power = load_event_spectrogram(data, duration, f_lower)
	print("Spectrogram data loaded successfully.")
	sample_rate = 2048
	
	psds = {}
	for ifo in ifos:
		noise = GWPYTimeSeries.fetch_open_data(ifo, mergertime-512*3, mergertime-512, version = 1)
		noise = noise.to_pycbc()
		#resample at 2048 Hz
		noise = resample_to_delta_t(noise, 1/ 2048)
		psd = noise.psd(4)
		psd = interpolate(psd, delta_f= 1/(duration))
		psd = inverse_spectrum_truncation(psd, int(4 * sample_rate),
											low_frequency_cutoff=f_lower)

		psds[ifo] = psd

	#TODO: turn into function

	template_bank_dir = "/fred/oz016/alistair/GWSamplegen/template_banks/"
	template_bank_name = "PyCBC_98_aligned_spin"

	templates, metricparams, axis= load_pycbc_templates(template_bank_name, template_bank_dir)

	N = int(duration/delta_t)
	kmin, kmax = np_get_cutoff_indices(f_lower, None, delta_f, N)


	t_templates = np.empty((n_templates, kmax-kmin), dtype=np.complex128)

	for psd in psds:
		psds[psd] = psds[psd][kmin:kmax]

	m1_det = merger.mass1 * (1 + merger.z)
	m2_det = merger.mass2 * (1 + merger.z)
	
	if template is None: 
		best = choose_templates_new(templates, metricparams, 10, m1_det, m2_det, aXis=axis)
		print("best templates for this event:",best)

		template_start_idx = best[0]

	else:
		template_start_idx = template
		print("using template",template)
		
	fd_approximant = "TaylorF2"

	for j in range(n_templates):
		t_templates[j] = get_fd_waveform(mass1 = templates[template_start_idx + j,1], 
									mass2 = templates[template_start_idx + j,2],
									spin1z = templates[template_start_idx + j,3], 
									spin2z = templates[template_start_idx + j,4],
									approximant = fd_approximant, f_lower = f_lower, 
									delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]
		
	template_conj = np.conjugate(t_templates)

	#finally, do matched filtering
	start_cutoff = 100
	end_cutoff = 1000
	slice_duration = end_cutoff - start_cutoff
	window_size = 2048
	stride = 128

	print("Loaded noise, PSDs and templates. performaing matched filtering...")

	n_windows = (slice_duration*sample_rate - window_size)//stride +1
	nonwindowed_SNR = np.empty((len(ifos), n_templates, slice_duration*sample_rate), dtype=np.float32)
	windowed_SNR = np.empty((len(ifos), n_templates, n_windows, window_size), dtype=np.float32)

	for ifo in range(len(ifos)):
		
		template_norm = np_sigmasq(t_templates, psds[ifos[ifo]], delta_f)

		data = GWPYTimeSeries.fetch_open_data(ifos[ifo], mergertime-512, mergertime+512, version = 1)
		data = data.to_pycbc()
		strain = resample_to_delta_t(data, 1/2048)

		#strain = TimeSeries(noise[ifo], delta_t=delta_t, copy=False)
		strain = highpass(strain,f_lower).to_frequencyseries(delta_f=delta_f).data
		strain = np.array([strain])[:,kmin:kmax]
		strain_np = np.repeat(strain, n_templates, axis=0)

		y = mf_in_place(strain_np, psds[ifos[ifo]], N, kmin, kmax, template_conj, template_norm)
		nonwindowed_SNR[ifo, :, :] = np.abs(y[:,start_cutoff*sample_rate:end_cutoff*sample_rate]).astype(np.float32)

	for ifo in range(len(ifos)):
		windowed_SNR[ifo] = np.array(make_windows_2d(nonwindowed_SNR[ifo, :, :], window_size, stride))
	
	print("Matched filtering successful. loading model and producing plot...")
	model = keras.models.load_model(model_path, compile = False)

	#recompile the model without the last 2 layers
	#TODO: add some automatic check for:
	#A. if there is a lambda layer
	#B. if there is an activation function to remove
	print(model.layers[-5:])
	model = keras.models.Model(inputs = model.input, outputs = model.layers[-3].output)

	
	peak_pos_array = []
	for i in range(0, nonwindowed_SNR.shape[2]-sample_rate+stride, stride):

		peak_pos_array.append(np.argmax(nonwindowed_SNR[:,: , i:i+sample_rate], axis = 2))
	peak_pos_array = np.array(peak_pos_array)

	light_travel_time = sample_rate//100
	n = 0

	print(windowed_SNR.shape)

	if len(ifos) == 2:
		delta_t_array = (peak_pos_array[:, 0, :] - peak_pos_array[:, 1, :])/light_travel_time
		preds = model.predict([windowed_SNR[0, n, :], windowed_SNR[1, n, :], delta_t_array[:, 0]], workers = 1)

	else:
		double_det, combiner, full_model = split_model_stack(model_path)
		if "H1" in ifos:
			preds0 = double_det.predict([windowed_SNR[0, n, :], np.zeros(shape=windowed_SNR[0,0].shape), np.ones((windowed_SNR.shape[2], 1))])
			preds0[1].fill(1.0)
		else:
			preds0 = double_det.predict([np.zeros(shape=windowed_SNR[0,0].shape), windowed_SNR[0, n, :], np.ones((windowed_SNR.shape[2], 1))])
			preds0[0].fill(1.0)
		
		#print(preds0)
		preds0.append(np.ones((windowed_SNR.shape[2], 1)))
		#print(preds0)
		preds = combiner.predict(preds0)
	
	preds = preds.squeeze()
	return mergertime, preds, times, freqs, power, nonwindowed_SNR, start_cutoff

""" 
def plot_event(ifos, mergertime, preds, times, freqs, power, nonwindowed_SNR, sample_rate = 2048, start_cutoff = 100):


	if len(ifos) == 2:
		fig, axes = plt.subplots(4,1, figsize=(6,8), sharex=True, dpi = 150)
	
	else:
		fig, axes = plt.subplots(3,1, figsize=(6,6), sharex=True, dpi = 150)

	seconds_before = 5
	seconds_after = 5

	plot_times = np.linspace(-seconds_before, seconds_after, sample_rate* (seconds_before + seconds_after))

	ax = 0
	for i in range(len(ifos)):
		axes[ax].plot(plot_times,
					nonwindowed_SNR[i, 0, sample_rate* (512-start_cutoff - seconds_before):sample_rate * (512-start_cutoff + seconds_after)], alpha = 0.8)
		axes[ax].set_ylabel(ifos[i] +" SNR")
		ax+=1

	axes[ax].plot(np.linspace(-411, -411+899, 14385), preds)
	axes[ax].plot(np.linspace(-410.5, -410.5+899, 14385), np.convolve(preds, np.ones(16)/16, mode = 'same'))

	axes[ax].set_ylabel("NN Prediction")

	axes[ax+1].pcolormesh(times- mergertime, freqs, np.log2(power), vmin=1, vmax=6)
	axes[ax+1].axvline(0, c = 'red', alpha = 0.5)
	axes[ax+1].set_yscale('log')
	axes[ax+1].set_ylabel("Frequency (Hz)")
	axes[ax+1].set_xlabel("time from merger (s)")

	plt.xlim(-seconds_before, seconds_after)
	plt.grid(False)
	#plt.subplots_adjust(wspace=0, hspace=0.2)
	#plots should share x axis
	plt.show()

	return fig, axes 

"""