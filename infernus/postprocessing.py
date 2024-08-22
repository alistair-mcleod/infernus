"""
Utilities for processing the data of background and injection runs


"""





import astropy.units as u
#import astropy.cosmology as cosmo
from astropy.cosmology import FlatwCDM
import numpy as np
cosmo = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)
from GWSamplegen.noise_utils import combine_seg_list, get_valid_noise_times
from GWSamplegen.waveform_utils import t_at_f
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm, t
import matplotlib.pyplot as plt
import h5py
from importlib import resources as impresources
from GWSamplegen import segments


#postprocessing for computing sensitive volume

def logdiffexp(x, y):
    ''' Evaluate log(exp(x) - exp(y)) '''
    return x + np.log1p( - np.exp(y - x) )

def log_dVdz(z):
	return np.log(4 * np.pi) + np.log(cosmo.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value)


def log_dNdm1dm2ds1ds2dz_mine(z, logprob_m1m2, logprob_spin, selection, log_dVdz):
    ''' Calculate dN / dm1 dm2 ds1 ds2 dz for selected injections 
    
    Arguments:
     Note: don't need m1, m2, s1 or s2 for full population as we already have the prior
    - s1x, s1y, s1z: primary spin components
    - s2x, s2y, s2z: secondary spin components
    - z: redshift
    - logprob_mass: function that takes in m1, m2 and calculate log p(m1, m2) 
    - logprob_spin: function that takes in spin parameters and calculate log p(s)
    - selection: selection function
    - params: parameters for distribution func
    '''
    
    #log_pm = logprob_mass(m1, m2, params)  # mass distribution p(m1, m2)
    log_pm = logprob_m1m2

    # total spin distribution
    log_ps = logprob_spin
      
    # Calculate the redshift terms, ignoring rate R0 because it will cancel out anyway
    # dN / dz = dV / dz  * 1 / (1 + z) + (1 + z)^kappa
    # where the second term is for time dilation
    # ignoring the rate because it will cancel out anyway
    log_dNdV = 0
    #log_dVdz = np.log(4 * np.pi) + np.log(cosmo.differential_comoving_volume(z).to(
    #    u.Gpc**3 / u.sr).value)
    log_time_dilation = - np.log(1 + z)
    log_dNdz = log_dNdV + log_dVdz + log_time_dilation
    
    return np.where(selection, log_pm + log_ps + log_dNdz, np.NINF)


def get_V(z, logprob_mass, logprob_spin, selection, N_draw, p_draw, log_dVdz):
    ''' Convienient function that returns log_VT, log_err_VT, and N_eff '''
    #TODO: rename stuff to just V, not VT. Also fix log_sig
    
    # Calculate V
    log_dN = log_dNdm1dm2ds1ds2dz_mine(z, logprob_mass, logprob_spin, selection, log_dVdz)
    log_V = - np.log(N_draw) + np.logaddexp.reduce(log_dN - np.log(p_draw))

    # Calculate uncertainty of V and effective number 
    #log_s2 = 2 * np.log(T_obs) - 2 * np.log(N_draw) + np.logaddexp.reduce(
    #    2 * (log_dN - np.log(p_draw)))
    log_s2 = - 2 * np.log(N_draw) + np.logaddexp.reduce(
        2 * (log_dN - np.log(p_draw)))
    log_sig2 = logdiffexp(log_s2, 2.0*log_V - np.log(N_draw))
    log_sig = log_sig2 / 2
    N_eff = np.exp(2 * log_V - log_sig2)
    
    return np.exp(log_V), np.exp(log_sig), N_eff





def get_injection_zerolags( valid_times, start_cutoff, end_cutoff, startgps, endgps):
    #return a list of zerolags that have injections in them.
    #valid_times is the start GPS times of the segments
    #start_cutoff is to account for the part of the SNR segments that are discarded due to edge effects.
    #should be ~100 seconds for 30Hz BNS injections, or ~10-20 seconds for 30 Hz BBH injections.

    #zls_per_segment = end_cutoff - start_cutoff
    timestep = 0

    inj_indexes = []
    inj_IDs = []
    GPS_time = []
    for i in range(len(valid_times)):


        for j in range(len(startgps)):
            if startgps[j] > valid_times[i] and endgps[j] + 1 < valid_times[i] + end_cutoff:

                zl_id = timestep + int(endgps[j] - valid_times[i]) - start_cutoff
                if zl_id not in inj_indexes and zl_id > 0:
                    #print("found injection {} in segment {}".format(j,i))
                    inj_indexes.append(zl_id)
                    inj_IDs.append(j)
                    GPS_time.append(endgps[j])


        if i < len(valid_times) - 1:
            if int(valid_times[i+1] - valid_times[i]) > end_cutoff - start_cutoff:
                timestep += int(end_cutoff - start_cutoff)
            else:
                timestep += int(valid_times[i+1] - valid_times[i])

    return inj_indexes, inj_IDs, GPS_time



#Fitting functions for extrapolation


def pdf_to_cdf_arbitrary(pdf):
	cumulative = np.cumsum(np.flip(pdf))
	cumulative = np.flip(cumulative/cumulative[-1])
	return cumulative



def lognorm_fit(data, method = 'MSE'):
    #p, bins, _ = plt.hist(data, bins = 1000, density=True, cumulative=-1)

    p, bins = np.histogram(data, bins = 1000, density = True)
    p = p[::-1].cumsum()[::-1]
    p/=p[0]

    #plt.clf()

    def lognorm_fit_func(params):
        mean, std = params
        cdf = pdf_to_cdf_arbitrary(norm.pdf(bins[:-1], mean, std))
        return np.sum(np.abs((np.log10(p) - np.log10(cdf))))

    x0 = np.array([np.mean(data), np.std(data)])
    res = minimize(lognorm_fit_func, x0, method = 'Nelder-Mead')

    return res.x


def preds_to_far(bg,preds, extrapolate = True):

	maxval = max(np.max(preds), np.max(bg)) + 10
	minval = min(np.min(preds), np.min(bg))

	bg = np.sort(bg)

	fars = 1 - np.searchsorted(bg, preds) / len(bg)
	fars = np.clip(fars, 1/len(bg), 1)

	if extrapolate:

		space = np.linspace(minval, maxval, 1000)
		mean, std = lognorm_fit(bg)
		cumulative = pdf_to_cdf_arbitrary(norm.pdf(space, mean, std))
		#only extrapolate fars above the max BG value
		#slight change: we go with fars above the 10th highest BG value
		thresh = bg[-10]

		fars[preds > thresh] = np.minimum(fars[preds > thresh], np.interp(preds[preds > thresh], space, cumulative))

		#fars = np.minimum(fars, np.interp(preds, space, cumulative))
		
	return fars



def log_t_fit(data):
    p, bins = np.histogram(data, bins = 1000, density = True)
    p = p[::-1].cumsum()[::-1]
    p/=p[0]

    def log_t_fit_func(params):
        df, mean,std = params
        cdf = pdf_to_cdf_arbitrary(t.pdf(bins[:-1], df, loc = mean, scale = std))
        return np.sum(np.abs((np.log10(p) - np.log10(cdf))))
    
    x0 = np.array([3,np.mean(data), np.std(data)])
    res = minimize(log_t_fit_func, x0, method = 'Nelder-Mead')

    return res.x

def preds_to_far_t(bg, preds, extrapolate = True):
     
    maxval = max(np.max(preds), np.max(bg)) + 10
    minval = min(np.min(preds), np.min(bg))

    bg = np.sort(bg)

    fars = 1 - np.searchsorted(bg, preds) / len(bg)
    fars = np.clip(fars, 1/len(bg), 1)

    if extrapolate:

        space = np.linspace(minval, maxval, 1000)
        nu, mean, std = log_t_fit(bg)
        cumulative = pdf_to_cdf_arbitrary(t.pdf(space, nu, loc = mean, scale = std))
        #only extrapolate fars above the max BG value
        #slight change: we go with fars above the 10th highest BG value
        thresh = bg[-10]

        fars[preds > thresh] = np.minimum(fars[preds > thresh], np.interp(preds[preds > thresh], space, cumulative))

        #fars = np.minimum(fars, np.interp(preds, space, cumulative))
        
    return fars

def lognorm_fit_constrained(data, upper = 1e-4, lower = 1e-6, method = 'MSE'):
   
    p, bins = np.histogram(data, bins = np.linspace(np.min(data),np.max(data), 1000), density = True)
    p = p[::-1].cumsum()[::-1]
    p/=p[0]

    p_upper = np.argmin(np.abs(p - upper))
    p_lower = np.argmin(np.abs(p - lower))
    #p_upper = -10

    #plt.clf()
    #print(p_upper)
    #print(len(p) - p_lower)

    def lognorm_fit_func_c(params):
        mean, std = params
        cdf = pdf_to_cdf_arbitrary(norm.pdf(bins[:-1], mean, std))

        return np.mean(np.abs((np.log10(p[p_upper: p_lower]) - np.log10(cdf[p_upper: p_lower]))))

    x0 = np.array([np.mean(data), np.std(data)])
    
    res = minimize(lognorm_fit_func_c, x0, method = 'Nelder-Mead')

    #plot the fit
    #plt.plot(bins[p_upper:-2], np.log10(p[p_upper:-1]), label = 'data')
    #plt.plot(bins[p_upper:-2], np.log10(pdf_to_cdf_arbitrary(norm.pdf(bins[:-1], res.x[0], res.x[1]))[p_upper:-1]), label = 'fit')

    return res.x

def preds_to_far_constrained(bg,preds, upper = 1e-3, lower = 1e-7, extrapolate = True):

	maxval = max(np.max(preds), np.max(bg)) + 10
	minval = min(np.min(preds), np.min(bg))

	bg = np.sort(bg)

	fars = 1 - np.searchsorted(bg, preds) / len(bg)
	fars = np.clip(fars, 1/len(bg), 1)

	if extrapolate:

		space = np.linspace(minval, maxval, 1000)
		mean, std = lognorm_fit_constrained(bg, upper = upper, lower = lower)
		cumulative = pdf_to_cdf_arbitrary(norm.pdf(space, mean, std))
		#only extrapolate fars above the max BG value
		#slight change: we go with fars above the 10th highest BG value
		thresh = bg[-10]

		fars[preds > thresh] = np.minimum(fars[preds > thresh], np.interp(preds[preds > thresh], space, cumulative))

		#fars = np.minimum(fars, np.interp(preds, space, cumulative))
		
	return fars



 



def get_O3_week(week):
    """Returns the start and end times of the given week of O3."""
    start = 1238166018 + (week-1)*60*60*24*7
    end = start + 60*60*24*7
    return start, end

#duration is duration of noise segments, duration - end_cutoff is the amount of ignored data at the end of each segment





#TODO: make this function work for any observing run, and in any installation.
def get_inj_data(week, noise_dir, background_stats, injection_file, mdc_file, merge_target = 6,
                 duration = 1024, start_cutoff = 100, end_cutoff = 1000, f_lower = 30, 
                 pipelines = ["pycbc_hyperbank", "mbta", "gstlal"],
                 ifo_1 = "H1_O3a.txt",
                 ifo_2 = "L1_O3a.txt",
                 two_detector_restriction = True):

    #TODO: properly divide up this function

    f = h5py.File(mdc_file, 'r')

    T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    N_draw = f.attrs['total_generated']
    accepted_fraction = f.attrs['n_accepted']/N_draw

    gps_times = f['injections/gps_time'][:]
    network_snr = f['injections/optimal_snr_net'][:]
    h_snr = f['injections/optimal_snr_h'][:]
    l_snr = f['injections/optimal_snr_l'][:]

    m1 = f['injections/mass1_source'][:]
    m2 = f['injections/mass2_source'][:]
    s1x = f['injections/spin1x'][:]
    s1y = f['injections/spin1y'][:]
    s1z = f['injections/spin1z'][:]    
    s2x = f['injections/spin2x'][:]
    s2y = f['injections/spin2y'][:]
    s2z = f['injections/spin2z'][:]
    z = f['injections/redshift'][:]
    distance = f['injections']['distance'][:]
    right_ascension = f['injections']['right_ascension'][:]
    declination = f['injections']['declination'][:]
    inclination = f['injections']['inclination'][:]
    polarization = f['injections']['polarization'][:]

    m1_det = f['injections/mass1'][:]
    m2_det = f['injections/mass2'][:]

    p_draw = f['injections/sampling_pdf'][:]

    pastro_cwb = f['injections/pastro_cwb'][:]
    pastro_gstlal = f['injections/pastro_gstlal'][:]    
    pastro_mbta = f['injections/pastro_mbta'][:]    
    pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]    
    pastro_pycbc_broad = f['injections/pastro_pycbc_hyperbank'][:]

    pipeline_fars = {}
    for p in pipelines:
        pipeline_fars[p] = f[f'injections/far_{p}'][:] / (86400*365.25)
    far_cwb = f['injections/far_cwb'][:]
    far_gstlal = f['injections/far_gstlal'][:]
    far_mbta = f['injections/far_mbta'][:]
    far_pycbc_bbh = f['injections/far_pycbc_bbh'][:]
    far_pycbc_broad = f['injections/far_pycbc_hyperbank'][:]

    m1_prior = f['injections/mass1_source_sampling_pdf'][:]
    m1_m2_prior = f['injections/mass1_source_mass2_source_sampling_pdf'][:]

    s1_prior = f['injections/spin1x_spin1y_spin1z_sampling_pdf'][:]
    s2_prior = f['injections/spin2x_spin2y_spin2z_sampling_pdf'][:]


    start, end = get_O3_week(week)
    try:
        ifo_1 = impresources.files(segments).joinpath(ifo_1)
        ifo_2 = impresources.files(segments).joinpath(ifo_2)
        segs, h1, l1 = combine_seg_list(ifo_1,ifo_2,start,end, min_duration=duration)
        print("fetched segment files from GWSamplegen")
    except:
        print("Looking for ifo files elsewhere")
        segs, h1, l1 = combine_seg_list(ifo_1,ifo_2,start,end, min_duration=duration)


    start_times = np.copy([np.floor(gpsi - t_at_f(m1_det[i], m2_det[i], f_lower)) for i, gpsi in enumerate(gps_times)])

    startgps = []
    for i in range(len(gps_times)):
        startgps.append(np.floor(gps_times[i] - t_at_f(m1_det[i], m2_det[i], f_lower)))
    startgps = np.array(startgps)

    mask = np.zeros(len(gps_times), dtype=bool)

    if two_detector_restriction:
        
        for i in range(len(gps_times)):
            for start, end in segs:
                if start_times[i] > start and gps_times[i] + 1 < end - (duration - end_cutoff) and gps_times[i] > start + start_cutoff:
                #if start_times[i] > start and gps_times[i] + 1 < end - (duration - end_cutoff):
                    mask[i] = True
                    break
    else:
        for i in range(len(gps_times)):
            if start_times[i] > start and gps_times[i] + 1 < end - (duration - end_cutoff) and gps_times[i] > start + start_cutoff:
            #if start_times[i] > start and gps_times[i] + 1 < end - (duration - end_cutoff):
                mask[i] = True


    #have to adjust N_draw to account for the fact that we're only using a fraction of the data
    N_draw = int(np.sum(mask)/accepted_fraction)

    valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, end_cutoff-start_cutoff)

    zls, inj_ids, GPS_rec = get_injection_zerolags(valid_times, start_cutoff, end_cutoff, startgps[mask], gps_times[mask])

    #print("some zls:",zls[:10])
    #print(inj_ids[:10])
    
    if type(background_stats) == str:
        stm = np.load(background_stats)
        stm = stm.reshape(-1,11)
        stmnew = stm[stm[:,0] != -1]

    else:
        print("using pre-loaded background")
        stmnew = background_stats

    injs = np.load(injection_file, allow_pickle=True)
    injs = injs.squeeze()
    injs = injs[zls]

    nn_preds = np.full(mask.sum(), -1000.0)
    nn_preds[inj_ids] = injs[:,merge_target]

    m1 = m1[mask]
    m2 = m2[mask]
    s1x = s1x[mask]
    s1y = s1y[mask]
    s1z = s1z[mask]
    s2x = s2x[mask]
    s2y = s2y[mask]
    s2z = s2z[mask]
    z = z[mask]
    distance = distance[mask]
    right_ascension = right_ascension[mask]
    declination = declination[mask]
    inclination = inclination[mask]
    polarization = polarization[mask]
    
    m1_det = m1_det[mask]
    m2_det = m2_det[mask]
    p_draw = p_draw[mask]
    pastro_cwb = pastro_cwb[mask]
    pastro_gstlal = pastro_gstlal[mask]
    pastro_mbta = pastro_mbta[mask]
    pastro_pycbc_bbh = pastro_pycbc_bbh[mask]
    pastro_pycbc_broad = pastro_pycbc_broad[mask]
    far_cwb = far_cwb[mask]
    far_gstlal = far_gstlal[mask]
    far_mbta = far_mbta[mask]
    far_pycbc_bbh = far_pycbc_bbh[mask]
    far_pycbc_broad = far_pycbc_broad[mask]
    m1_prior = m1_prior[mask]
    m1_m2_prior = m1_m2_prior[mask]
    s1_prior = s1_prior[mask]
    s2_prior = s2_prior[mask]
    gps_times = gps_times[mask]
    startgps = startgps[mask]
    network_snr = network_snr[mask]
    h_snr = h_snr[mask]
    l_snr = l_snr[mask]

    for p in pipelines:
        pipeline_fars[p] = pipeline_fars[p][mask]

    #make a dictionary of the variables

    d = {"m1": m1,
        "m2": m2,
        "s1x": s1x,
        "s1y": s1y,
        "s1z": s1z,
        "s2x": s2x,
        "s2y": s2y,
        "s2z": s2z,
        "z": z,
        "distance": distance,
        "right_ascension": right_ascension,
        "declination": declination,
        "inclination": inclination,
        "polarization": polarization,
        "m1_det": m1_det,
        "m2_det": m2_det,
        "p_draw": p_draw,
        "pastro_cwb": pastro_cwb,
        "pastro_gstlal": pastro_gstlal,
        "pastro_mbta": pastro_mbta,
        "pastro_pycbc_bbh": pastro_pycbc_bbh,
        "pastro_pycbc_broad": pastro_pycbc_broad,
        "far_cwb": far_cwb,
        "far_gstlal": far_gstlal,
        "far_mbta": far_mbta,
        "far_pycbc_bbh": far_pycbc_bbh,
        "far_pycbc_broad": far_pycbc_broad,
        "m1_prior": m1_prior,
        "m1_m2_prior": m1_m2_prior,
        "s1_prior": s1_prior,
        "s2_prior": s2_prior,
        "gps_times": gps_times,
        "startgps": startgps,
        "network_snr": network_snr,
        "h_snr": h_snr,
        "l_snr": l_snr,
        "pipeline_fars": pipeline_fars,
        "N_draw": N_draw,
        "mask": mask,
        "pipelines": pipelines,
        "zerolags": zls
    }

    return N_draw, mask, stmnew, nn_preds, injs, d



def load_ifar_data(inj_file, bg_stats, merge_target, has_injections = False, noise_dir = None, week = None, extrapolate = False,
                    mdc_file = "/fred/oz016/alistair/infernus/notebooks/gwtc3/endo3_bnspop-LIGO-T2100113-v12-1238166018-15843600.hdf5"):
    small_injs = np.load(inj_file, allow_pickle=True).squeeze()

    #we can use either injection runs or noninjection runs for this.
    if has_injections:
        N_draw, mask, stat_data, nn_preds, injs, params = get_inj_data(week, noise_dir, bg_stats, inj_file, 
                                            merge_target = merge_target, mdc_file = mdc_file)
        zls = np.array(params['zerolags'])
        not_injs = np.concatenate((zls-6, zls-5, zls-4, zls-3, zls-2, zls-1, zls, zls+1, zls+2, zls+3, zls+4, zls+5, zls+6))
        not_injs = np.unique(np.sort(not_injs))

        noninjm = small_injs[~np.isin(np.arange(len(small_injs)), not_injs)][:,merge_target]

    else:
        print("using noninj run, no zls needed")
        noninjm = small_injs[:,merge_target]
        stat_data = np.load(bg_stats)
        stat_data = stat_data.reshape(-1,11)
        stat_data = stat_data[stat_data[:,0] != -1]


    not_injs_fars = preds_to_far(stat_data[:,merge_target - 3], noninjm, extrapolate = extrapolate)

    far_bins = np.geomspace(1e-7,1,100)

    vals, bins = np.histogram(not_injs_fars, bins = far_bins)

    #len(not_injs_fars) is the length of the foreground

    return bins[:-1], vals, far_bins, len(not_injs_fars)

