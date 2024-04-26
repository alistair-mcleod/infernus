"""
Utilities for processing the data of background and injection runs


"""





import astropy.units as u
#import astropy.cosmology as cosmo
from astropy.cosmology import FlatwCDM
import numpy as np
cosmo = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)


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
                if zl_id not in inj_indexes:
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



#Fitting functions for 

from scipy.optimize import minimize
from scipy.stats import norm, skewnorm
import matplotlib.pyplot as plt

def pdf_to_cdf_arbitrary(pdf):
	cumulative = np.cumsum(np.flip(pdf))
	cumulative = np.flip(cumulative/cumulative[-1])
	return cumulative



def lognorm_fit(data, method = 'MSE'):
    #p, bins, _ = plt.hist(data, bins = 1000, density=True, cumulative=-1)

    p, bins = np.histogram(data, bins = 1000, density = True)
    p = p[::-1].cumsum()[::-1]
    p/=p[0]

    plt.clf()

    def lognorm_fit_func(params):
        mean, std = params
        cdf = pdf_to_cdf_arbitrary(norm.pdf(bins[:-1], mean, std))
        return np.sum(np.abs((np.log10(p) - np.log10(cdf))))

    x0 = np.array([np.mean(data), np.std(data)])
    res = minimize(lognorm_fit_func, x0, method = 'Nelder-Mead')

    return res.x


def preds_to_far(bg,preds, extrapolate = True):

	maxval = max(np.max(preds), np.max(bg))
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

from scipy.stats import t

def log_t_fit(data):
    p, bins, _ = plt.hist(data, bins = 1000, density=True, cumulative=-1)
    plt.clf()

    def log_t_fit_func(params):
        df, mean,std = params
        cdf = pdf_to_cdf_arbitrary(t.pdf(bins[:-1], df, loc = mean, scale = std))
        return np.sum(np.abs((np.log10(p) - np.log10(cdf))))
    
    x0 = np.array([3,np.mean(data), np.std(data)])
    res = minimize(log_t_fit_func, x0, method = 'Nelder-Mead')

    return res.x

def preds_to_far_t(bg, preds, extrapolate = True):
     
    maxval = max(np.max(preds), np.max(bg))
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