"""
Utilities for processing the data of background and injection runs


"""





import astropy.units as u
#import astropy.cosmology as cosmo
from astropy.cosmology import FlatwCDM
import numpy as np
cosmo = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)
from GWSamplegen.noise_utils import combine_seg_list, get_valid_noise_times, get_valid_noise_times_from_segments,gps_to_run
from GWSamplegen.waveform_utils import t_at_f
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm, t
import h5py
from importlib import resources as impresources
from GWSamplegen import segments


#Postprocessing for computing sensitive volume.
#These sensitive volume functions were adapted from the code for the GWTC-3 analysis, which can be found at
#https://zenodo.org/records/7890437 

def logdiffexp(x, y):
    ''' Evaluate log(exp(x) - exp(y)) '''
    return x + np.log1p( - np.exp(y - x) )

def log_dVdz(z):
	return np.log(4 * np.pi) + np.log(cosmo.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value)

#TODO:this is the most up to date version of this function, delete the 'mine' one later
def log_dNdm1dm2ds1ds2dz(
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, z, 
    logprob_mass, logprob_spin, selection, params, log_dVdz):
    ''' Calculate dN / dm1 dm2 ds1 ds2 dz for selected injections 
    
    Arguments:
    - m1, m2: primary and secondary spin components
    - s1x, s1y, s1z: primary spin components
    - s2x, s2y, s2z: secondary spin components
    - z: redshift
    - logprob_mass: function that takes in m1, m2 and calculate log p(m1, m2) 
    - logprob_spin: function that takes in spin parameters and calculate log p(s)
    - selection: selection function
    - params: parameters for distribution func
    '''
    
    log_pm = logprob_mass(m1, m2, params)  # mass distribution p(m1, m2)
    
    #TODO: fix spin distributions: not all injections with m1 or m2 > 2 are BHs!!!
    if params['m1_full_pop'] and params['m2_full_pop']:
        log_ps = params['s1_s2_prior']
    else:
        # primary spin distribution
        s1_max = np.where(m1 < 2, params['smax_ns'], params['smax_bh'])
        spin1_params = params.copy()
        spin1_params['smax'] = s1_max
        log_ps1 = logprob_spin(s1x, s1y, s1z, spin1_params)
        
        # secondary spin distribution
        s2_max = np.where(m2 < 2, params['smax_ns'], params['smax_bh'])
        spin2_params = params.copy()
        spin2_params['smax'] = s2_max
        log_ps2 = logprob_spin(s2x, s2y, s2z, spin2_params)
        
        # total spin distribution
        log_ps = log_ps1 + log_ps2
      
    # Calculate the redshift terms, ignoring rate R0 because it will cancel out anyway
    # dN / dz = dV / dz  * 1 / (1 + z) + (1 + z)^kappa
    # where the second term is for time dilation
    # ignoring the rate because it will cancel out anyway
    cosmo = params['cosmo']
    log_dNdV = 0
    #log_dVdz = np.log(4 * np.pi) + np.log(cosmo.differential_comoving_volume(z).to(
    #    u.Gpc**3 / u.sr).value)
    log_time_dilation = - np.log(1 + z)

    log_dNdz = log_dNdV + log_dVdz + log_time_dilation
    
    return np.where(selection, log_pm + log_ps + log_dNdz, np.NINF)


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


def get_V(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, z, 
          logprob_mass, logprob_spin, selection, N_draw, p_draw, params, log_dVdz):
    ''' Convienient function that returns log_V, log_err_V, and N_eff '''
    
    # Calculate V
    log_dN = log_dNdm1dm2ds1ds2dz(
        m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, z, 
        logprob_mass, logprob_spin, selection, params, log_dVdz)
    log_V = -np.log(N_draw) + np.logaddexp.reduce(log_dN - np.log(p_draw))

    # Calculate uncertainty of V and effective number
    log_s2 = -2 * np.log(N_draw) + np.logaddexp.reduce(
        2 * (log_dN - np.log(p_draw)))
    log_sig2 = logdiffexp(log_s2, 2.0*log_V - np.log(N_draw))
    log_sig = log_sig2 / 2
    N_eff = np.exp(2 * log_V - log_sig2)

    return np.exp(log_V), np.exp(log_sig), N_eff

import scipy.stats as stats

def logprob_mass2_lognorm(m2, m1, params):
    ''' evaluate p(m2 | m1) = c * lognormal(m2 | m, sigma) 
    where 
    - lognormal is log-normal distribution that is truncated at m1
    - c is the normalization correction factor    
    '''
    m2_mean = params['m2_mean']
    sig_lognorm = params['sig_lognorm_m2']
    
    logc = -stats.lognorm.logcdf(m1, sig_lognorm, scale=m2_mean)
    return np.where(
        m2 <  m1, stats.lognorm.logpdf(m2, sig_lognorm, scale=m2_mean) + logc, np.NINF)

def logprob_mass_lognorm(m1, m2, params):
    ''' evaluate p(m1, m2) = p(m1) p(m2 | m1)
    with 
    - p(m1) = log_normal(m1; m, 0.1)
    - p(m2 | m1) = log_normal(m2; m, 0.1) truncated such that m2 < m1    
    '''
    
    sig_lognorm = params['sig_lognorm_m1']
    m1_mean = params['m1_mean']
    
    log_pm1 = stats.lognorm.logpdf(m1, sig_lognorm, scale=m1_mean)
    log_pm2 = logprob_mass2_lognorm(m2, m1, params)
    if params['m1_full_pop'] and params['m2_full_pop']:
        #return 0
        #TODO: Confirm this is correct!!!
        return np.log(params['m1_m2_prior'])
    elif params['m2_full_pop']:
        return log_pm1
    else:
        return log_pm1 + log_pm2

def logprob_spin(sx, sy, sz, params):
    ''' Evaluate p(sx, sy, sz) = (1. / |s|^2) p(|s|, cos theta, phi) = 1. / (4 pi s_max |s|^2)
    where:
    - |s| = sqrt(sx^2 + sy^2 + sz^2)
    
    The mass `m` determines which s_max to use
    '''
    smax = params['smax']    
    s2 = sx**2 + sy**2 + sz**2 
    return np.where(s2 < smax**2, - np.log(4 * np.pi) - np.log(smax) - np.log(s2), np.NINF)


def get_dsens(z, p_draw, N_draw, pipelines, pipeline_fars, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, pop_params=None):
	fars = np.geomspace(1e-3, 1e-12, 50)

	#log_s1_s2 = np.log(s1_prior) + np.log(s2_prior)
	dVdz = log_dVdz(z)
	#print("pop params:", pop_params)
	VT = {}
	sigma_VT = {}
	for pipeline in pipelines:
		VT[pipeline] = []
		sigma_VT[pipeline] = []
		for far in fars:
			if pipeline == "any":
				#make an "or" selection for all pipelines
				selection = np.zeros(len(pipeline_fars['gstlal']), dtype=bool)
				for p in pipeline_fars.keys():
					selection = selection | (pipeline_fars[p] < far)
			elif pipeline == "any (no NN)":
				selection = ((pipeline_fars['gstlal'] < far) |
					(pipeline_fars['pycbc_hyperbank'] < far) | (pipeline_fars['mbta'] < far))
			else:
				selection = pipeline_fars[pipeline] < far
			
			log_vt, log_sigma_vt, N_eff = get_V(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, z, logprob_mass_lognorm, logprob_spin, 
				selection=selection, N_draw=N_draw, p_draw=p_draw, params=pop_params, log_dVdz=dVdz)

			VT[pipeline].append(log_vt)
			sigma_VT[pipeline].append(log_sigma_vt)


	return VT, sigma_VT


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

    p, bins = np.histogram(data, bins = 1000, density = True)
    p = p[::-1].cumsum()[::-1]
    p/=p[0]

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

    def lognorm_fit_func_c(params):
        mean, std = params
        cdf = pdf_to_cdf_arbitrary(norm.pdf(bins[:-1], mean, std))

        return np.mean(np.abs((np.log10(p[p_upper: p_lower]) - np.log10(cdf[p_upper: p_lower]))))

    x0 = np.array([np.mean(data), np.std(data)])
    
    res = minimize(lognorm_fit_func_c, x0, method = 'Nelder-Mead')

    return res.x

def lognorm_fit_constrained_print(data, upper = 1e-4, lower = 1e-7, method = 'MSE', verbose = True):
   
    p, bins = np.histogram(data, bins = np.linspace(np.min(data),np.max(data), 1000), density = True)
    p = p[::-1].cumsum()[::-1]
    p/=p[0]

    p_upper = np.argmin(np.abs(p - upper))
    p_lower = np.argmin(np.abs(p - lower))

    def lognorm_fit_func_c(params):
        mean, std = params
        cdf = pdf_to_cdf_arbitrary(norm.pdf(bins[:-1], mean, std))

        return np.mean(np.abs((np.log10(p[p_upper: p_lower]) - np.log10(cdf[p_upper: p_lower]))))

    x0 = np.array([np.mean(data), np.std(data)])
    
    res = minimize(lognorm_fit_func_c, x0, method = 'Nelder-Mead')
    if verbose:
        print(res)

    mean, std = res.x
    #get the r squared value

    cdf = pdf_to_cdf_arbitrary(norm.pdf(bins[:-1], mean, std))
    cdf = cdf[p_upper:p_lower]
    p = p[p_upper:p_lower]
    
    r2 = 1 - np.sum((np.log10(p) - np.log10(cdf))**2)/np.sum((np.log10(p) - np.mean(np.log10(p)))**2)
    if verbose:
        print("r-squared value: ", r2)
    return res.x

def preds_to_far_constrained(bg,preds, upper = 1e-3, lower = 1e-7, extrapolate = True, verbose = True):

	maxval = max(np.max(preds), np.max(bg)) + 10
	minval = min(np.min(preds), np.min(bg))

	bg = np.sort(bg)

	fars = 1 - np.searchsorted(bg, preds) / len(bg)
	fars = np.clip(fars, 1/len(bg), 1)

	if extrapolate:

		space = np.linspace(minval, maxval, 1000)
		mean, std = lognorm_fit_constrained_print(bg, upper = upper, lower = lower, verbose = verbose)
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


def get_inj_data(week, noise_dir, mdc_file,
                 duration = 1024, start_cutoff = 100, end_cutoff = 1000, f_lower = 30, 
                 pipelines = ["pycbc_hyperbank", "mbta", "gstlal"],
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


    if type(week) == int:
        start, end = get_O3_week(week)
    elif type(week) == tuple:
        start, _ = get_O3_week(week[0])
        _, end = get_O3_week(week[1])
    
    if type(noise_dir) == list: 
        print("Inj run specified as a tuple of GPS times. Make sure they're contiguous.")
        start = noise_dir[0][0]
        end = noise_dir[-1][1]
        #get the obsrun ID from the start and end time. If they agree, use that. If not, raise an error.
        runstart = gps_to_run(start)
        runend = gps_to_run(end)
        if runstart != runend:
            raise ValueError("The GPS times specified for the injection run do not correspond to a single observing run. Please specify GPS times that correspond to a single observing run, e.g. O2, O3a, O3b, O4a, etc.")
        else:
            print("GPS times correspond to run ", runstart)
        ifo_1 = "H1_{}.txt".format(runstart)
        ifo_2 = "L1_{}.txt".format(runstart)
    else:
        ifo_1 = "H1_O3a.txt"
        ifo_2 = "L1_O3a.txt"
        print("defaulting to O3a segment files. Fix this code to be more flexible in the future.")

    try:
        ifo_1 = impresources.files(segments).joinpath(ifo_1)
        ifo_2 = impresources.files(segments).joinpath(ifo_2)
        segs, h1, l1 = combine_seg_list(ifo_1,ifo_2,start,end, min_duration=duration)
        #print("fetched segment files from GWSamplegen")
    except:
        #print("Looking for ifo files elsewhere")
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
    if type(noise_dir) == list:
        print("New noise segment list fetched.")
        valid_times = get_valid_noise_times_from_segments(noise_dir, duration, end_cutoff-start_cutoff)
    else:
        valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, end_cutoff-start_cutoff)

    zls, inj_ids, GPS_rec = get_injection_zerolags(valid_times, start_cutoff, end_cutoff, startgps[mask], gps_times[mask])

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
        "zerolags": zls,
        "inj_ids": inj_ids
    }

    return N_draw, mask, d

def load_ifar_data(inj_file, bg_stats, merge_target, mdc_file, 
        has_injections = False, noise_dir = None, week = None, extrapolate = False):
    #Load a background file, an injection/non-injection file, and compute the FARs for the non-injection data.
    #Getting the non-injection data from an injection run requires a noise directory, an injection file and a week number.

    inj_array = np.load(inj_file, allow_pickle=True).squeeze()

    #we can use either injection runs or noninjection runs for this.
    if has_injections:
        N_draw, mask, stat_data, nn_preds, injs, params = get_inj_data(week, noise_dir, bg_stats, inj_file, 
                                            merge_target = merge_target, mdc_file = mdc_file)
        zls = np.array(params['zerolags'])
        not_injs = np.concatenate((zls-6, zls-5, zls-4, zls-3, zls-2, zls-1, zls, zls+1, zls+2, zls+3, zls+4, zls+5, zls+6))
        not_injs = np.unique(np.sort(not_injs))

        noninjm = inj_array[~np.isin(np.arange(len(inj_array)), not_injs)][:,merge_target]

    else:
        print("using noninj run, no zls needed")
        noninjm = inj_array[:,merge_target]
        stat_data = np.load(bg_stats)
        stat_data = stat_data.reshape(-1,11)
        stat_data = stat_data[stat_data[:,0] != -1]


    not_injs_fars = preds_to_far(stat_data[:,merge_target - 3], noninjm, extrapolate = extrapolate)

    #TODO: make the bin limit an argument
    far_bins = np.geomspace(1e-7,1,100)
    vals, bins = np.histogram(not_injs_fars, bins = far_bins)

    return bins[:-1], vals, far_bins, len(not_injs_fars)

