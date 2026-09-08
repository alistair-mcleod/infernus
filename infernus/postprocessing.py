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
    logprob_mass, logprob_spin, selection, params, log_dVdz = None):
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
    #we can precompute log_dVdz if needed for a slight speedup.
    if log_dVdz is None:
        log_dVdz = np.log(4 * np.pi) + np.log(cosmo.differential_comoving_volume(z).to(
            u.Gpc**3 / u.sr).value)
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
    ''' Convienient function that returns V, err_V, and N_eff '''
    
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


def logprob_spin_polar(s, theta, phi, params):
    """
    Evaluate p(|s|, theta, phi) = sin(theta) / (4 pi smax |s|^2)
    
    The mass "m" determines which s_max to use.

    Used as of O4a for the spin distribution.
    """
      
    smax = params['smax']
    return np.where(
            (s < smax),
            np.log(np.sin(theta) / (4 * np.pi * smax)),
            np.NINF
        )


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
					(pipeline_fars['pycbc'] < far) | (pipeline_fars['mbta'] < far))
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
	minval = np.min(bg)

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

from infernus.injection_utils import load_injections_temp
def get_inj_data(week, noise_dir, mdc_file,
                 duration = 1024, start_cutoff = 100, end_cutoff = 1000, f_lower = 30, 
                 pipelines = ["pycbc", "mbta", "gstlal"],
                 two_detector_restriction = True):

    #TODO: properly divide up this function
    
    # f = h5py.File(mdc_file, 'r')

    # T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    # N_draw = f.attrs['total_generated']
    # accepted_fraction = f.attrs['n_accepted']/N_draw

    # gps_times = f['injections/gps_time'][:]
    # network_snr = f['injections/optimal_snr_net'][:]
    # h_snr = f['injections/optimal_snr_h'][:]
    # l_snr = f['injections/optimal_snr_l'][:]

    # m1 = f['injections/mass1_source'][:]
    # m2 = f['injections/mass2_source'][:]
    # s1x = f['injections/spin1x'][:]
    # s1y = f['injections/spin1y'][:]
    # s1z = f['injections/spin1z'][:]    
    # s2x = f['injections/spin2x'][:]
    # s2y = f['injections/spin2y'][:]
    # s2z = f['injections/spin2z'][:]
    # z = f['injections/redshift'][:]
    # distance = f['injections']['distance'][:]
    # right_ascension = f['injections']['right_ascension'][:]
    # declination = f['injections']['declination'][:]
    # inclination = f['injections']['inclination'][:]
    # polarization = f['injections']['polarization'][:]

    # m1_det = f['injections/mass1'][:]
    # m2_det = f['injections/mass2'][:]

    # p_draw = f['injections/sampling_pdf'][:]

    # pastro_cwb = f['injections/pastro_cwb'][:]
    # pastro_gstlal = f['injections/pastro_gstlal'][:]    
    # pastro_mbta = f['injections/pastro_mbta'][:]    
    # pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]    
    # pastro_pycbc_broad = f['injections/pastro_pycbc_hyperbank'][:]

    # pipeline_fars = {}
    # for p in pipelines:
    #     pipeline_fars[p] = f[f'injections/far_{p}'][:] / (86400*365.25)
    # far_cwb = f['injections/far_cwb'][:]
    # far_gstlal = f['injections/far_gstlal'][:]
    # far_mbta = f['injections/far_mbta'][:]
    # far_pycbc_bbh = f['injections/far_pycbc_bbh'][:]
    # far_pycbc_broad = f['injections/far_pycbc_hyperbank'][:]

    # m1_prior = f['injections/mass1_source_sampling_pdf'][:]
    # m1_m2_prior = f['injections/mass1_source_mass2_source_sampling_pdf'][:]

    # s1_prior = f['injections/spin1x_spin1y_spin1z_sampling_pdf'][:]
    # s2_prior = f['injections/spin2x_spin2y_spin2z_sampling_pdf'][:]

    #TODO: properly handle start and end times
    ret = load_injections_temp(mdc_file, 0, 0, start_cutoff, end_cutoff, 
                               duration, f_lower, pipelines, verbose = True)

    T_obs = ret['T_obs']
    N_draw = ret['N_draw']
    accepted_fraction = ret['accepted_fraction']

    gps_times = ret['gps_times']
    network_snr = ret['network_snr']
    h_snr = ret['h_snr']
    l_snr = ret['l_snr']

    m1 = ret['m1']
    m2 = ret['m2']
    s1x = ret['s1x']
    s1y = ret['s1y']
    s1z = ret['s1z']
    s2x = ret['s2x']
    s2y = ret['s2y']
    s2z = ret['s2z']
    z = ret['z']
    distance = ret['distance']
    right_ascension = ret['right_ascension']
    declination = ret['declination']
    inclination = ret['inclination']
    polarization = ret['polarization']

    m1_det = ret['m1_det']
    m2_det = ret['m2_det']

    p_draw = ret['p_draw']

    # if pipelines is not None:
    #     pastro_cwb = ret['pastro_cwb']
    #     pastro_gstlal = ret['pastro_gstlal']
    #     pastro_mbta = ret['pastro_mbta']
    #     pastro_pycbc = ret['pastro_pycbc']

    pipeline_fars = ret['pipeline_fars']
    pipeline_pastros = ret['pipeline_pastros']

    m1_prior = ret['m1_prior']
    m1_m2_prior = ret['m1_m2_prior']
    s1_prior = ret['s1_prior']
    s2_prior = ret['s2_prior']

    if "weights" in ret:
        weights = ret['weights']
    else:
        weights = np.ones_like(m1_prior)


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
        segs_total = []
        for noise in noise_dir:
            #accounting for non-contiguous noise segments. TODO: ensure that ALL code can handle non-contiguous noise segments, not just this part.
            segs, _, _ = combine_seg_list(ifo_1,ifo_2,noise[0],noise[1], min_duration=duration)
            segs_total.extend(segs)
        segs = segs_total
        #print("fetched segment files from GWSamplegen")
    except:
        #print("Looking for ifo files elsewhere")
        segs, h1, l1 = combine_seg_list(ifo_1,ifo_2,start,end, min_duration=duration)


    startgps = np.copy([np.floor(gpsi - t_at_f(m1_det[i], m2_det[i], f_lower)) for i, gpsi in enumerate(gps_times)])

    mask = np.zeros(len(gps_times), dtype=bool)

    if two_detector_restriction:
        
        for i in range(len(gps_times)):
            for start, end in segs:
                if startgps[i] > start and gps_times[i] + 1 < end - (duration - end_cutoff) and gps_times[i] > start + start_cutoff:
                #if startgps[i] > start and gps_times[i] + 1 < end - (duration - end_cutoff):
                    mask[i] = True
                    break
    else:
        for i in range(len(gps_times)):
            if startgps[i] > start and gps_times[i] + 1 < end - (duration - end_cutoff) and gps_times[i] > start + start_cutoff:
            #if startgps[i] > start and gps_times[i] + 1 < end - (duration - end_cutoff):
                mask[i] = True


    #have to adjust N_draw to account for the fact that we're only using a fraction of the data
    N_draw_old = int(np.sum(mask)/accepted_fraction)
    print("old N_draw: ", N_draw_old)
    seg_sum = 0 
    for seg in segs:
        seg_sum += seg[1] - seg[0]
    print("total segment time: ", seg_sum)
    N_draw = N_draw * seg_sum / (T_obs*365.25*24*3600)
    print("adjusted N_draw: ", N_draw)
    if type(noise_dir) == list:
        print("New noise segment list fetched.")
        valid_times = get_valid_noise_times_from_segments(noise_dir, duration, end_cutoff-start_cutoff, blacklisting = False)
    else:
        valid_times, paths, file_list = get_valid_noise_times(noise_dir,duration, end_cutoff-start_cutoff, blacklisting = False)

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
    #TODO: put p_astros into their own dictionary 
    #pastro_cwb = pastro_cwb[mask]
    #pastro_gstlal = pastro_gstlal[mask]
    #pastro_mbta = pastro_mbta[mask]
    #pastro_pycbc_bbh = pastro_pycbc_bbh[mask]
    #pastro_pycbc = pastro_pycbc[mask]
    #far_cwb = far_cwb[mask]
    #far_gstlal = far_gstlal[mask]
    #far_mbta = far_mbta[mask]
    #far_pycbc_bbh = far_pycbc_bbh[mask]
    #far_pycbc = far_pycbc[mask]
    m1_prior = m1_prior[mask]
    m1_m2_prior = m1_m2_prior[mask]
    s1_prior = s1_prior[mask]
    s2_prior = s2_prior[mask]
    gps_times = gps_times[mask]
    startgps = startgps[mask]
    network_snr = network_snr[mask]
    h_snr = h_snr[mask]
    l_snr = l_snr[mask]

    weights = weights[mask]

    if pipelines is not None:
        for p in pipelines:
            pipeline_fars[p] = pipeline_fars[p][mask]
            pipeline_pastros[p] = pipeline_pastros[p][mask]
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
        #"pastro_cwb": pastro_cwb,
        #"pastro_gstlal": pastro_gstlal,
        #"pastro_mbta": pastro_mbta,
        #"pastro_pycbc_bbh": pastro_pycbc_bbh,
        #"pastro_pycbc": pastro_pycbc,
        #"far_cwb": far_cwb,
        #"far_gstlal": far_gstlal,
        #"far_mbta": far_mbta,
        #"far_pycbc_bbh": far_pycbc_bbh,
        #"far_pycbc": far_pycbc,
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
        "inj_ids": inj_ids,
        "file_format": ret['file_format'],
        "weights": weights
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

def Gpc3_to_Mpc3(V):
    return V * 1e9
def Gpc3_to_Mpc(V):
    #converts a volume in Gpc^3 to a distance in Mpc, assuming a spherical volume
    return (3 * Gpc3_to_Mpc3(V) / (4 * np.pi))**(1/3)

def round_two_figures(x):
    if x >= 10000:
        return '%.1e' % x
    if x >= 1000:
        return '%.0f' % round(x, -2)
    if x >= 100:
        return '%.0f' % round(x, -1)
    if x >= 10:
        return '%.0f' % round(x)
    if x >= 1:
        return '%.1f' % round(x, 1)
    if x >= 0.1:
        return '%.2f' % round(x, 2)
    if x >= 0.01:
        return '%.4f' % round(x, 4)
    if x >= 0.001:
        return '%.5f' % round(x, 5)
    if x == 0:
        return '0'
    return '%.1e' % x

def cartesian_to_spherical(sx, sy, sz):
	s = np.sqrt(sx**2 + sy**2 + sz**2)
	theta = np.arccos(sz / s)  # polar angle
	phi = np.arctan2(sy, sx)  # azimuthal angle
	#for phi we need to convert from [-pi, pi] to [0, 2pi]
	phi = np.where(phi < 0, phi + 2 * np.pi, phi)
	return s, theta, phi


def log_dNdm1dm2ds1ds2dz_O4(
    m1, m2, s1, theta1, phi1, s2, theta2, phi2, z, 
    logprob_mass, logprob_spin, selection, params):
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
    
    # primary spin distribution
    s1_max = np.where(m1 < 2, params['smax_ns'], params['smax_bh'])
    spin1_params = params.copy()
    spin1_params['smax'] = s1_max
    log_ps1 = logprob_spin(s1, theta1, phi1, spin1_params)
    
    # secondary spin distribution
    s2_max = np.where(m2 < 2, params['smax_ns'], params['smax_bh'])
    spin2_params = params.copy()
    spin2_params['smax'] = s2_max
    log_ps2 = logprob_spin(s2, theta2, phi2, spin2_params)
    
    # total spin distribution
    log_ps = log_ps1 + log_ps2
      
    # Calculate the redshift terms, ignoring rate R0 because it will cancel out anyway
    # dN / dz = dV / dz  * 1 / (1 + z) + (1 + z)^kappa
    # where the second term is for time dilation
    # ignoring the rate because it will cancel out anyway
    cosmo = params['cosmo']
    log_dNdV = 0
    log_dVdz = np.log(4 * np.pi) + np.log(cosmo.differential_comoving_volume(z).to(
        u.Gpc**3 / u.sr).value)
    log_time_dilation = - np.log(1 + z)
    log_dNdz = log_dNdV + log_dVdz + log_time_dilation
    #print(np.where(selection,log_pm, np.NINF), np.where(selection,log_ps, np.NINF), np.where(selection,log_dNdz, np.NINF))
    return np.where(selection, log_pm + log_ps + log_dNdz, np.NINF)

def get_logV(m1, m2, s1, theta1, phi1, s2, theta2, phi2, z, 
          logprob_mass, logprob_spin, selection, N_draw, p_draw, params, weights):
    ''' Convienient function that returns log_V, log_sigma_V, and N_eff '''
    
    # Calculate V
    log_dN = log_dNdm1dm2ds1ds2dz(
        m1, m2, s1, theta1, phi1, s2, theta2, phi2, z, 
        logprob_mass, logprob_spin, selection, params)
    #print("log DN:",log_dN)
    log_V = - np.log(N_draw) + np.logaddexp.reduce(log_dN - np.log(p_draw) +np.log(weights))

    # Calculate uncertainty of VT and effective number 
    log_s2 = - 2 * np.log(N_draw) + np.logaddexp.reduce(
        2 * (log_dN - np.log(p_draw)+np.log(weights)))
    log_sig2 = logdiffexp(log_s2, 2.0*log_V - np.log(N_draw))
    log_sig = log_sig2 / 2
    N_eff = np.exp(2 * log_V - log_sig2)
    
    return log_V, log_sig, N_eff

def point_volume(inj_params, m1, m2, pipeline, detection_threshold = 1/(365.25*24*3600)):
	pop_params = {
	'sig_lognorm_m1': 0.1,
	'sig_lognorm_m2': 0.1,
	'smax_ns': 0.4,
	'smax_bh': 0.998,
	'cosmo': FlatwCDM(H0=67.9, Om0=0.3065, w0=-1),
	"m1_mean": m1, "m2_mean": m2,
	"m1_full_pop": False, "m2_full_pop": False
	}
	selection = inj_params["pipeline_fars"][pipeline] < detection_threshold

	if inj_params['file_format'] == "O4":
		s1_mag, s1_theta, s1_phi = cartesian_to_spherical(inj_params['s1x'], inj_params['s1y'], inj_params['s1z'])
		s2_mag, s2_theta, s2_phi = cartesian_to_spherical(inj_params['s2x'], inj_params['s2y'], inj_params['s2z'])
		log_V, log_sigma, n_eff = get_logV(
			inj_params['m1'], inj_params['m2'], s1_mag, s1_theta, s1_phi, s2_mag, s2_theta, s2_phi, inj_params['z'], 
			logprob_mass_lognorm, logprob_spin_polar, selection, inj_params['N_draw'], inj_params['p_draw'], pop_params, inj_params['weights'])
		return np.exp(log_V), np.exp(log_sigma), n_eff
	else:
		print("TODO: handle non O4 file formats")


def apply_func_to_bg_inj(bg,inj, func, rs_index = 8):
	return func(bg[:,:,rs_index], axis = 1), func(inj[:,:,rs_index], axis = 1)

def compute_NN_fars(bg_file, inj_file, inj_params, trigger_selection, return_best_only = False, detection_threshold = 1/(365.25*24*3600), upper_thresh = 1e-3, lower_thresh = 1e-7):
	"""Compute the false alarm rates for injections from an injection set. """
	
	#pipeline_fars = inj_params["pipeline_fars"]
	mask = inj_params["mask"]
	

	#check if bg_file is a numpy array or a string path to a numpy file
	if isinstance(bg_file, str):
		bg = np.load(bg_file)
		bg = bg.astype(np.float32)
		bg = bg.reshape(-1, bg.shape[2], bg.shape[3])
	else:
		#check if it's already in the right shape
		bg = bg_file
		if len(bg.shape) == 4:
			print("reshaping BG file to get rid of timeslides axis")
			bg = bg.reshape(-1, bg.shape[2], bg.shape[3])
	bg = bg[np.all(bg[:,:,2] > 0, axis = 1)]

	if isinstance(inj_file, str):
		zerolags = np.load(inj_file)[0] #the 0 is to get rid of the timeslides axis
		for i in range(zerolags.shape[0]):
			for j in range(zerolags.shape[1]):
				if zerolags[i,j,2] < 0:
					zerolags[i,j,8:] = -100 #catch-all to ensure that triggers with no SNR are rejected

	bg_sort = []
	pipeline_fars = {}
	for i in range(8,bg.shape[2]):
		bg_func, inj_func = apply_func_to_bg_inj(bg, zerolags, trigger_selection, rs_index = i)
		#get rid of nans from the background
		bg_func = bg_func[np.where(np.isfinite(bg_func))]
		#also clip the background at 50
		#bg_func = bg_func[(bg_func < 25)]
		#bg_func = np.nan_to_num(bg_func, nan = -10)

		nn_preds = np.full(mask.sum(), -1000.0)
		nn_preds[inj_params["inj_ids"]] = inj_func[inj_params["zerolags"]]

		#look 1 index either side of inj_params['zerolags'] as the peak might be slightly off
		zl_before = np.array(inj_params['zerolags']) -1
		zl_after = np.array(inj_params['zerolags']) +1
		nn_preds[inj_params["inj_ids"]] = np.maximum(nn_preds[inj_params["inj_ids"]], inj_func[zl_before])
		nn_preds[inj_params["inj_ids"]] = np.maximum(nn_preds[inj_params["inj_ids"]], inj_func[zl_after])

		nn_preds = np.nan_to_num(nn_preds, nan = 0)
		preds = preds_to_far_constrained(bg_func, nn_preds, upper = upper_thresh, lower = lower_thresh)
		#preds = np.nan_to_num(preds, nan = 0)
		

		pipeline_fars['NN'+str(i)] = preds
		#if 'NN'+str(i) not in pipelines:
		#	pipelines.append('NN'+str(i))

		pipeline_fars['NN'+str(i)][pipeline_fars['NN'+str(i)] == 1] = np.inf

	if return_best_only is not False:
		#if return_best_only is an int, we assume that is the best NN to return. If it's True, we compute which NN is best at the detection threshold and return only that one.
		if return_best_only is not True:
			best = return_best_only
		else:
			best = 8
			for j in range(8, bg.shape[2]):
				if np.sum(pipeline_fars['NN'+str(j)] < detection_threshold) > np.sum(pipeline_fars['NN'+str(best)] < detection_threshold):
					best = j
			print("Best NN is NN{}".format(best))
		#remove all NNs except the best one
		for j in range(8, bg.shape[2]):
			if j != best:
				del pipeline_fars['NN'+str(j)]
	return pipeline_fars



import matplotlib.pyplot as plt
from pycbc.sensitivity import volume_to_distance_with_errors
def plot_sensitive_distance(pipelines, sensitivities, sigmas, fars = np.geomspace(1e-3,1e-12,50), detection_threshold = 1/(365.25*24*3600)):
	#plot sensitive distance plot as usual
	
	fig, axes = plt.subplots(2,2, figsize=(10,8), sharex=True)
	for i, ax in enumerate(axes.flatten()):
		for p in pipelines:
			#if p == "NN8" or p == "NN9":
			#	continue
			vt = np.array(sensitivities[p])
			sigma = np.array(sigmas[p])
			dist, ehigh, elow = volume_to_distance_with_errors(vt*1e9, sigma*1e9)


			ax.plot(fars, dist, label=p, linewidth=1, alpha = 0.7, zorder = 6, )
			ax.fill_between(fars, dist-elow, dist+ehigh, alpha=0.3, ec = 'none', zorder = 6)


		ax.set_xscale('log')
		ax.set_xlim(1e-3,1e-11)
		ax.axvline(1/(86400*365.25), color = 'red', linestyle = '--', alpha = 0.5, linewidth = 1, label = "Detection threshold")
		ax.grid(zorder = -10)
		ax.legend()
		ax.set_ylabel("Sensitive distance (Mpc)")
		ax.set_xlabel("False alarm rate (Hz)")
	return fig, axes


import matplotlib.colors as colors
from scipy.optimize import fsolve
def plot_CBC_sensitive_volume(pipelines, sensitivities, sigmas, sigma_threshold = 0.5):

    #TODO: make this an argument
    masses = np.array([1.5,5,10,20,35,60,100])
    M1, M2 = np.meshgrid(masses, masses)
    M1 = M1.flatten()
    M2 = M2.flatten()
    cut = M2 <= M1
    M1 = M1[cut]
    M2 = M2[cut]

    #number of pipelines determines the number of subplots
    n_subplots = len(pipelines)
    fig, axes = plt.subplots(int(np.ceil(n_subplots/2)), 2, figsize=(18, 8*np.ceil(n_subplots/2)), sharex=True)


    ## Chirp Mass
    Mc_contours = [2, 4, 10, 35, 75]
    m1mins = [1, 1, 1.2, 10, 35]  # Minimum m1 values to plot each contour : depends on max m2 plotted
    m1maxs = [7, 40, 250, 250, 250]  # Maximum m1 values to plot each contour

    #fig = plt.figure(figsize=(18, 24))
    #gs = gridspec.GridSpec(3, 2, figure=fig)
    #fig, axes = plt.subplots(int(np.ceil(n_subplots/2)), 2, figsize=(18, 8*np.ceil(n_subplots/2)), sharex=True)


    # Define the chirp mass formula
    def chirp_mass(m2, m1, chirp_mass):
        return ((m1 * m2)**(3/5) / (m1 + m2)**(1/5)) - chirp_mass

    axes = axes.flatten()

    #order should be Any, Any (no NN), our search, pycbc, gstlal, mbta

    # order = list(pipeline_fars.keys())
    # #rearrange order so that "Any (no NN)" is first, then "Any", then the rest
    # if "any (no NN)" in order:
    #     order.remove("any (no NN)")
    #     order.insert(0, "any (no NN)")

    # if "any" in order:
    #     order.remove("any")
    #     order.insert(1, "any")
    order = pipelines

    color_norm = colors.LogNorm(vmin=1e-3, vmax=25)

    for j, ax in enumerate(axes):

        for i in range(len(M1)):

            m1 = M1[i]
            m2 = M2[i]
            point_v, point_sigma = sensitivities[(m1, m2, order[j])], sigmas[(m1, m2, order[j])]
            ## plot the point
            if point_sigma/point_v < sigma_threshold and point_v > 1e-5:
                sc = ax.scatter(
                    m1, m2, c=point_v, s=2000, cmap="summer", norm=color_norm, edgecolor='none',
                    label=order[j])
                
                text = round_two_figures(point_v)
                x,y = (m1, m2)
                ax.text(x, y, text, ha='center', va='center', fontsize=18)
                
                proportion = 1 - point_v/sensitivities[(m1, m2, 'any')]
                start_point = 0.25
                x1 = np.cos(2 * np.pi * np.linspace(start_point, proportion + start_point))
                y1 = np.sin(2 * np.pi * np.linspace(start_point, proportion + start_point))
                xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])
                ax.scatter(m1, m2, c='white', s=2000, marker=xy1, alpha=0.5, edgecolor='none')
            
        ax.set_xlim(1, 200)
        ax.set_ylim(0.9, 200)
        ax.loglog(base=2)

        ticklabels = (1.5, 5, 10, 20, 35, 60, 100)
        ax.set_xticks(ticklabels)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticklabels)
        ax.set_yticklabels(ticklabels)
        
        ax.plot(np.linspace(1, 500), np.linspace(1, 500), ls="--", color='k', zorder=0)
        ax.text(135, 110, r"$q=1$", fontsize=14, rotation=33)
        if order[j] == "cWB-BBH":
            ax.set_title("cWB-BBH", fontsize=22)
        else:
            ax.set_title(order[j], fontsize=22)  

        
        ## Chirp Mass
        # Mc_contours and m1min/max were already defined
        textadjustment_x = [1.05, 2.1, 4.3, 15.0, 38.5]  # Minor tweaks as these plots have different aspect ratio
        textadjustment_y = [2.25, 4.5, 13.2, 47.0, 85.0]
        for i, Mc in enumerate(Mc_contours):
            m_2 = []
            m_1_arr = np.logspace(np.log10(m1mins[i]), np.log10(m1maxs[i]), 100)
            for m_1 in m_1_arr:
                m_2.append(fsolve(chirp_mass, x0=1, args=(m_1, Mc))[0])

            ax.plot(m_1_arr, m_2, ls='--', color='k', zorder=0)
            ax.text(textadjustment_x[i], textadjustment_y[i], r"$\mathcal{M}= %.1f\ M_\odot$" % Mc, fontsize=13, rotation=-40)

    # Global x and y labels
    fig.text(0.5, -0.01, r'Primary mass $m_1$ [$M_{\odot}$]', fontsize=24, ha='center')
    fig.text(0.001, 0.5, r'Secondary mass $m_2$ [$M_{\odot}$]', fontsize=24, va='center', rotation='vertical')

    # Global colorbar
    cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', aspect=30, pad=0.0, anchor= (0.35,-1.7))  
    cbar.set_label(r'Hypervolume $\langle VT \rangle$ [Gpc$^3$ yr]', fontsize=24)


    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.88])

    #fig.show()
    #fig.savefig("VT_all_results_combined.png", dpi=300, bbox_inches='tight')
    return fig, axes



def V_summary_plot(inj_params):
    pipelines = list(inj_params['pipeline_fars'].keys())

    #How to compute any (No NN): just take the minimum FAR across all pipelines excluding the NNs
    inj_params['pipeline_fars']['any'] = np.min(np.array(list(inj_params['pipeline_fars'].values())), axis = 0)
    inj_params['pipeline_fars']['any (no NN)'] = np.min(np.array([inj_params['pipeline_fars'][p] for p in pipelines if ("NN" not in p and "any" not in p)]), axis = 0)

    masses = np.array([1.5,5,10,20,35,60,100])
    M1, M2 = np.meshgrid(masses, masses)
    M1 = M1.flatten()
    M2 = M2.flatten()
    cut = M2 <= M1
    M1 = M1[cut]
    M2 = M2[cut]
    pipelines = inj_params['pipeline_fars'].keys()
    #go by masses rather than by selection function
    results = {}
    results_sigma = {}
    for i in range(len(M1)):
        m1 = M1[i]
        m2 = M2[i]
        #print(f"Calculating VT for masses: m1={m1}, m2={m2}")
        for name in pipelines:
            VT, sigma_VT, Neff = point_volume(inj_params, m1, m2, name, detection_threshold=1/(365.25*24*3600))
            print(f"VT for {name} at m1={m1}, m2={m2}: {VT}, sigma_VT: {sigma_VT}")
            results[(m1, m2, name)] = VT
            results_sigma[(m1, m2, name)] = sigma_VT

    fig, axes = plot_CBC_sensitive_volume(list(inj_params['pipeline_fars'].keys()), results, results_sigma)

    return fig, axes


from pycbc.detector import Detector
from GWSamplegen.waveform_utils import chirp_mass
import os
def plot_missed_vs_found(params, FARs,save_fp, OPA_threshold = 1/(3600*24*365.25), suffix = ""):

	#cosmo = Cosmology()
	d = Detector("H1")
	d2 = Detector("L1")

	distance = params['distance']
	right_ascension = params['right_ascension']
	declination = params['declination']
	polarization = params['polarization']
	gps = params['gps_times']
	inclination = params['inclination']
	chirp_masses = chirp_mass(params['m1'], params['m2'])

	deff_h = d.effective_distance(distance, right_ascension, declination, polarization, gps, inclination)
	deff_l = d2.effective_distance(distance, right_ascension, declination, polarization, gps, inclination)
	fig, axes = plt.subplots(1,2, figsize=(3.6,2.8), sharey=True, width_ratios=[3, 1], dpi = 300)
	fig.tight_layout()

	found = (FARs < OPA_threshold)
	missed = (FARs > OPA_threshold)

	axes[0].scatter(chirp_masses[missed], deff_l[missed], s = 5, c = 'tab:blue', alpha = 1, linewidths=0, label = "Missed", zorder = 3)
	axes[0].scatter(chirp_masses[found], deff_l[found], s = 10, c = 'tab:orange', marker='x', alpha = 1, linewidths=0.6, label = "Found", zorder = 5)

	#make a subplot showing the missed vs found fraction at different distances
	print()
	bins = np.geomspace(1, np.max(deff_l)*1.1, 200)

	rec_fracs = []
	with np.errstate(divide='ignore', invalid='ignore'):
		for bin in bins:
			n_found = np.sum((deff_l < bin) & found)
			n_missed = np.sum((deff_l < bin) & missed)

			if np.isnan(np.divide(n_found, (n_missed+n_found))):
				rec_fracs.append(1)
			else:
				rec_fracs.append(n_found/(n_missed+n_found))

	axes[1].plot(rec_fracs, bins)
	axes[1].set_xlabel("Found \n fraction")
	#draw a line intersecting the 50% recovery fraction
	axes[1].axvline(0.5, color = 'grey', lw = 1)
	axes[1].axhline(bins[np.where(np.array(rec_fracs) < 0.5)[0][0]], color = 'grey', lw = 1)
	axes[1].set_xlim(0,1.1)
	axes[1].set_xticks([0,0.5,1])
	#plt.plot(bins,rec_fracs)

	print("half recovery distance:", bins[np.where(np.array(rec_fracs) < 0.5)[0][0]])

	#plt.hist(deff_l[missed], bins = bins, alpha = 0.5, color = 'grey', density = True, label = "missed")

	axes[0].set_xlabel("Chirp mass (M$_\odot$)")
	axes[0].set_ylabel("L1 effective distance (Mpc)")
	axes[0].legend(loc = 'lower right')
	axes[0].grid(zorder = -1)
	
	#axes[0].set_xscale("log")
	#print("chirp mass scale:", np.max(chirp_masses)/np.min(chirp_masses))
	if np.max(chirp_masses)/np.min(chirp_masses) > 20:
		axes[0].set_xscale("log")
	axes[0].set_xlim(np.min(chirp_masses)*0.9,np.max(chirp_masses)*1.1)

	plt.yscale("log")
	plt.ylim(10,np.max(deff_l)*1.1)

	plt.subplots_adjust(wspace = 0.15)

	plt.tight_layout()
	#plt.show()
	plt.savefig(os.path.join(save_fp, f"deff_{suffix}.pdf"))
	#plt.savefig("figures/deff.pdf", bbox_inches = 'tight')#,dpi = 400)