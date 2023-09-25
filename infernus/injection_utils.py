#  Copyright (C) 2017-2019 Jolien Creighton, Sarah Caudill, Thomas Dent
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with with program; see the file COPYING. If not, write to the
#  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#  MA  02111-1307  USA

## @file
# The python module for utilities used in constructing rates injection sets.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import numpy
import numpy.random

from glue.ligolw import lsctables

import lal
import lalsimulation

#
# =============================================================================
#
#                          Distance Distributions
#
# =============================================================================
#

def draw_redshift(zmax, omega, sfrpower=0.):
    '''
    Yields a random redshift from a cosmologically-correct distribution.
    Uses Metropolis algorithm to draw from the desired pdf.

    zmax : maximum redshift
    omega : cosmology params
    sfrpower : power of (1+z) to multiply the constant-comoving-rate
        distribution by. Ex. 2.7 -> * (1+z)**2.7 (Madau SFR factor)
    '''

    def pdf(z):
        '''
        This redshift pdf yields a uniform distribution
        in comoving volume divided by (1+z).
        '''
        # FIXME: XLALUniformComovingVolumeDensity() currently includes
        # a factor of 1/(1+z) that converts to source-frame time.
        # If this changes, modify the code below.
        return lal.UniformComovingVolumeDensity(z, omega) * (1. + z)**sfrpower
        #return lal.UniformComovingVolumeDensity(z, omega) * (1. + z)**(sfrpower - 1.)

    z0 = numpy.random.uniform(0.0, zmax)
    p0 = pdf(z0)
    while True:
        # acceptance rate is 50% so take every 10th
        # draw from distribution to avoid repeating
        # the same value too often
        for _ in range(10):
            z = numpy.random.uniform(0.0, zmax)
            p = pdf(z)
            if p > p0 or numpy.random.random() < p / p0:
                z0 = z
                p0 = p
        yield z0

#
# =============================================================================
#
#                             Mass Distributions
#
# =============================================================================
#

def powerlaw_setup(minv, maxv, alpha):
    a = (maxv / minv) ** (alpha + 1.) - 1.
    b = 1. / (alpha + 1.)
    return a, b


def powerlaw_sample(x_rand, minv, a, b):
    return minv * (1. + a * x_rand) ** b


def draw_mass_pair_imf(mass_dict, alpha_salpeter=-2.35):
    '''
    Yields random masses, with the first component drawn from
    the Salpeter initial mass function distribution and the
    second mass drawn uniformly between min_mass and the mass of
    the first component.  Note: max_mtotal has very little
    effect on the resulting distribution.

    Requires max_mass, min_mass, max_mtotal keys
    '''
    # Input checking
    min_mass = mass_dict['min_mass']
    max_mass = mass_dict['max_mass']
    assert min_mass > 0
    assert max_mass >= min_mass
    assert mass_dict['max_mtotal'] > 2. * min_mass

    a, b = powerlaw_setup(min_mass, max_mass, alpha_salpeter)

    while True:
        x = numpy.random.random()
        m1 = powerlaw_sample(x, min_mass, a, b)
        m2 = numpy.random.uniform(min_mass, m1)

        if m1 + m2 < mass_dict['max_mtotal']:
            yield tuple(numpy.random.permutation((m1, m2)))


def draw_mass_pair_power(mass_dict, alpha=-2.35, beta=2.):
    '''
    Yields mass pairs with the primary mass drawn from the Salpeter power
    law distribution and the secondary drawn from a distribution proportional
    to the square of mass between min_mass and the primary mass.
    (Or any other power laws, if the default kwargs are overridden.)

    Requires max_mass, min_mass keys
    '''
    try:
        m1pow = mass_dict['mass1_power']
        if m1pow == None: m1pow = alpha
    except:
        m1pow = alpha  # default to alpha if not specified
    try:
        m2pow = mass_dict['mass2_power']
        if m2pow == None: m2pow = beta
    except:
        m2pow = beta  # default to beta
    print('power laws', m1pow, m2pow)
    min_mass = mass_dict['min_mass']
    max_mass = mass_dict['max_mass']
    assert min_mass > 0
    assert max_mass >= min_mass

    a1, b1 = powerlaw_setup(min_mass, max_mass, m1pow)
    while True:
        x1 = numpy.random.random()
        x2 = numpy.random.random()
        m1 = powerlaw_sample(x1, min_mass, a1, b1)
        a2, b2 = powerlaw_setup(min_mass, m1, m2pow)
        m2 = powerlaw_sample(x2, min_mass, a2, b2)
        yield m1, m2


def draw_mass_pair_uniform(mass_dict):
    '''
    Yields a random mass pair (m1, m2) where each mass is drawn from a uniform
    distribution so that min_mass <= m1,m2 < max_mass.

    Requires max_mass, min_mass keys
    '''
    min_mass = mass_dict['min_mass']
    max_mass = mass_dict['max_mass']
    assert min_mass is not None
    assert max_mass >= min_mass

    while True:
        yield numpy.random.uniform(min_mass, max_mass, size=2)


def draw_mass_distinct_uniform(mass_dict):
    '''
    Yields a random mass pair (m1, m2) where each mass is drawn from a separate
    uniform distribution so that min_mass1 <= m1 < max_mass1 and
    min_mass2 <= m2 <= max_mass2 and m1+m2 <= max_mtotal.

    Requires max_mass{1,2}, min_mass{1,2}, max_mtotal keys
    '''
    min_mass1 = mass_dict['min_mass1']
    max_mass1 = mass_dict['max_mass1']
    min_mass2 = mass_dict['min_mass2']
    max_mass2 = mass_dict['max_mass2']
    assert min_mass1 is not None
    assert min_mass2 is not None
    assert max_mass1 >= min_mass1
    assert max_mass2 >= min_mass2
    assert mass_dict['max_mtotal'] > min_mass1 + min_mass2

    while True:
        m1 = numpy.random.uniform(min_mass1, max_mass1)
        m2 = numpy.random.uniform(min_mass2, max_mass2)

        if m1 + m2 < mass_dict['max_mtotal']:
            yield m1, m2


def draw_mass_pair_uniform_in_log_mass(mass_dict):
    '''
    Yields random masses drawn uniformly-in-log between min_mass
    and max_mass, discarding those with a total mass exceeding
    max_mtotal.

    Requires max_mass, min_mass, max_mtotal keys
    '''
    lnmmin = numpy.log(mass_dict['min_mass']) # already checks for Nones
    lnmmax = numpy.log(mass_dict['max_mass'])
    assert lnmmax >= lnmmin
    assert mass_dict['max_mtotal'] > 2. * mass_dict['min_mass']

    while True:
        m1 = numpy.exp(numpy.random.uniform(lnmmin, lnmmax))
        m2 = numpy.exp(numpy.random.uniform(lnmmin, lnmmax))

        if m1 + m2 < mass_dict['max_mtotal']:
            yield m1, m2


def draw_mass_distinct_uniform_in_log_mass(mass_dict):
    '''
    Yields a random mass pair (m1, m2) where each mass is drawn from a separate
    uniform-in-log  distribution so that min_mass1 <= m1 < max_mass1 and
    min_mass2 <= m2 <= max_mass2 and m1+m2 <= max_mass.

    Requires max_mass{1,2}, min_mass{1,2}, max_mtotal keys
    '''
    lnmmin1 = numpy.log(mass_dict['min_mass1'])
    lnmmax1 = numpy.log(mass_dict['max_mass1'])
    lnmmin2 = numpy.log(mass_dict['min_mass2'])
    lnmmax2 = numpy.log(mass_dict['max_mass2'])
    assert lnmmax1 >= lnmmin1
    assert lnmmax2 >= lnmmin2
    assert mass_dict['max_mtotal'] > mass_dict['min_mass1'] + mass_dict['min_mass2']

    while True:
        m1 = numpy.exp(numpy.random.uniform(lnmmin1, lnmmax1))
        m2 = numpy.exp(numpy.random.uniform(lnmmin2, lnmmax2))

        if m1 + m2 < mass_dict['max_mtotal']:
            yield m1, m2


def draw_mass_pair_normal(mass_dict):
    '''
    Yields a random mass pair (m1, m2) where each mass is drawn from a normal
    distribution of mean mean_mass and standard deviation sigma_mass, clipped
    so that min_mass <= m1,m2 < max_mass.

    Requires mean_mass, sigma_mass, max_mass, min_mass keys
    '''
    mean_mass = mass_dict['mean_mass']
    sigma_mass = mass_dict['sigma_mass']
    assert mean_mass is not None
    assert sigma_mass is not None
    assert mass_dict['min_mass'] is not None
    assert mass_dict['max_mass'] > mass_dict['min_mass']

    while True:
        m = numpy.random.normal(mean_mass, sigma_mass, size=2)

        if max(m) < mass_dict['max_mass'] and min(m) >= mass_dict['min_mass']:
            yield m


def draw_mass_distinct_uniform_uniforminlog(mass_dict):
    '''
    Yields a random mass pair (m1, m2) where mass1 is drawn from a
    uniform distribution so that min_mass1 <= m1 < max_mass1 and
    mass2 is drawn from a uniform_in_log_mass distribution so that
    min_mass2 <= m2 < max_mass2

    Requires max_mass{1,2}, min_mass{1,2} keys
    '''
    min_mass1 = mass_dict['min_mass1']
    max_mass1 = mass_dict['max_mass1']
    lnm2min = numpy.log(mass_dict['min_mass2'])
    lnm2max = numpy.log(mass_dict['max_mass2'])
    assert min_mass1 is not None
    assert max_mass1 >= min_mass1
    assert lnm2max >= lnm2min

    while True:
         m1 = numpy.random.uniform(min_mass1, max_mass1)
         m2 = numpy.exp(numpy.random.uniform(lnm2min, lnm2max))
         yield m1, m2


def draw_mass_distinct_normal_imf(mass_dict, alpha_salpeter=-2.35):
    '''
    Yields a random mass pair (m1, m2) where mass1 is drawn from a normal
    distribution of mean mean_mass and standard deviation sigma_mass and
    mass2 is drawn from a Salpeter initial mass function distribution.

    Requires mean_mass, sigma_mass, min_mass{1,2}, max_mass{1,2}, max_mtotal keys
    '''
    mean_mass = mass_dict['mean_mass']
    sigma_mass = mass_dict['sigma_mass']
    min_mass1 = mass_dict['min_mass1']
    max_mass1 = mass_dict['max_mass1']
    min_mass2 = mass_dict['min_mass2']
    max_mass2 = mass_dict['max_mass2']
    assert mean_mass is not None
    assert sigma_mass is not None
    assert min_mass1 is not None
    assert max_mass1 >= min_mass1
    assert min_mass2 is not None
    assert max_mass2 >= min_mass2
    assert mass_dict['max_mtotal'] > min_mass1 + min_mass2

    a, b = powerlaw_setup(min_mass2, max_mass2, alpha_salpeter)

    while True:
        # FIXME Is this the right way to sample these
        m1 = numpy.random.normal(mean_mass, sigma_mass)
        x = numpy.random.random()
        m2 = powerlaw_sample(x, min_mass2, a, b)

        if m1 < max_mass1 and m1 >= min_mass1 and (m1 + m2) < mass_dict['max_mtotal']:
            yield m1, m2

def draw_mass_distinct_uniform_imf(mass_dict, alpha_salpeter=-2.35):
    '''
    Yields a random mass pair (m1, m2) where mass1 is drawn from a uniform
    distribution so that min_mass1 <= m1 < max_mass1 and
    mass2 is drawn from a Salpeter initial mass function distribution.

    Requires min_mass{1,2}, max_mass{1,2} keys
    '''
    min_mass1 = mass_dict['min_mass1']
    max_mass1 = mass_dict['max_mass1']
    min_mass2 = mass_dict['min_mass2']
    max_mass2 = mass_dict['max_mass2']
    assert min_mass1 is not None
    assert max_mass1 >= min_mass1
    assert min_mass2 is not None
    assert max_mass2 >= min_mass2

    a, b = powerlaw_setup(min_mass2, max_mass2, alpha_salpeter)

    while True:
        m1 = numpy.random.uniform(min_mass1, max_mass1)
        x = numpy.random.random()
        m2 = powerlaw_sample(x, min_mass2, a, b)

        yield m1, m2

#
# =============================================================================
#
#                             Spin Distributions
#
# =============================================================================
#

def draw_spin_aligned(spin_dict):
    '''
    Yields a random spin tuple (s_x, s_y, s_z) where s_x = s_y = 0
    and s_z is uniformly random in (-max_spin, +max_spin).
    '''
    max_spin = spin_dict['max_spin']

    while True:
        sgn = 2. * (numpy.random.random_integers(0, 1) - 0.5)
        s = sgn * numpy.random.uniform(0., max_spin)
        yield numpy.array([0., 0., s])


def draw_spin_isotropic(spin_dict):
    '''
    Yields a random spin tuple (s_x, s_y, s_z) isotropically
    distributed with uniform magnitude distribution.
    '''
    max_spin = spin_dict['max_spin']

    while True:
        s = numpy.random.uniform(-1., 1., size=3)
        ssq = sum(s ** 2.)

        # s is a vector uniformly distributed inside the unit sphere
        # p(|s|) ~ |s|^2
        if ssq < 1.:
            # s * |s|^2 has magnitude |chi| = |s|^3
            # p(|chi|) d|chi| = p(|s|) d|s| ~ |s|^2 d|s| ~ const. d|chi|
            yield s * ssq * max_spin


def draw_spin_aligned_aligned(spin_dict):
    '''
    Yields random spin tuples (s1_x, s1_y, s1_z) and (s2_x, s2_y, s2_z)
    where s1_x = s1_y = s2_x = s2_y = 0, s1_z is uniformly random in
    (-max_spin1, +max_spin1) and s2_z is uniformly random in
    (-max_spin2, +max_spin2).
    '''
    max_spin1 = spin_dict['max_spin1']
    max_spin2 = spin_dict['max_spin2']

    while True:
        sgn1 = 2. * (numpy.random.random_integers(0, 1) - 0.5)
        s1 = sgn1 * numpy.random.uniform(0., max_spin1)
        s1_tuple = numpy.array([0., 0., s1])

        sgn2 = 2. * (numpy.random.random_integers(0, 1) - 0.5)
        s2 = sgn2 * numpy.random.uniform(0., max_spin2)
        s2_tuple = numpy.array([0., 0., s2])

        yield numpy.concatenate((s1_tuple, s2_tuple))


def draw_spin_isotropic_aligned(spin_dict):
    '''
    Yields a random spin tuple (s2_x, s2_y, s2_z) where s2_x = s2_y = 0
    and s2_z is uniformly random in (-max_spin2, +max_spin2). Yields a random
    spin tuple (s1_x, s1_y, s1_z) isotropically distributed with uniform
    magnitude distribution.
    '''
    max_spin1 = spin_dict['max_spin1']
    max_spin2 = spin_dict['max_spin2']

    while True:
        sgn = 2. * (numpy.random.random_integers(0, 1) - 0.5)
        s2 = sgn * numpy.random.uniform(0., max_spin2)
        s2_tuple = numpy.array([0., 0., s2])

        s1 = numpy.random.uniform(-1., 1., size=3)
        ssq = sum(s1 ** 2.)

        if ssq < 1.:
            s1_tuple = s1 * ssq * max_spin1
            yield numpy.concatenate((s1_tuple, s2_tuple))


def draw_spin_isotropic_isotropic(spin_dict):
    '''
    Yields distinct random spin tuples (s1_x, s1_y, s1_z)
    and (s2_x, s2_y, s2_z) isotropically distributed with uniform
    magnitude distribution.
    '''
    max_spin1 = spin_dict['max_spin1']
    max_spin2 = spin_dict['max_spin2']

    while True:
        s1 = numpy.random.uniform(-1., 1., size=3)
        ssq1 = sum(s1 ** 2.)

        s2 = numpy.random.uniform(-1., 1., size=3)
        ssq2 = sum(s2 ** 2.)

        if ssq1 < 1. and ssq2 < 1.:
            s1_tuple = s1 * ssq1 * max_spin1
            s2_tuple = s2 * ssq2 * max_spin2
            yield numpy.concatenate((s1_tuple, s2_tuple))


def zpdf_interpolate(omega, zmax, zpower):
    from scipy.interpolate import interp1d
    zs = numpy.expm1(numpy.linspace(numpy.log(1.), numpy.log(1. + zmax), 1024))
    pzs = []
    for z in zs:
        pzs.append(lal.UniformComovingVolumeDensity(z, omega) * (1. + z)**zpower)
    z_norm = numpy.trapz(numpy.array(pzs), zs)
    return interp1d(zs, pzs/z_norm)


def imf_m2squared_pdf(row, mass_dict, omega, zmax, alpha=-2.35, beta=2., zpower=0, zinterp=None):
    """Returns the normalized density `p(m1_source, m2_source, z, s1z, s2z)` for the `power_pair` distribution.

    :param row: injection XML row filled with injected values incl. redshift in `alpha3`
    :param mass_dict: dict storing min and max mass and mass power law indices
    :param omega: cosmology parameters
    :param zmax: maximum redshift
    :param alpha: (default -2.35) m1 marginal powerlaw
    :param beta: (default 2) m2 conditional powerlaw p(m2 | m1)
    :param zpower: merger rate is `(1+z)**zpower` in the comoving frame.
    :param zinterp: scipy.interpolate object giving redshift PDF

    :return: Normalized density in `m1_source, m2_source, s1z, s2z, z` space.
    """
    try:
        m1pow = mass_dict['mass1_power']
        if m1pow is None: m1pow = alpha
    except:
        m1pow = alpha  # default to alpha if not specified
    try:
        m2pow = mass_dict['mass2_power']
        if m2pow is None: m2pow = beta
    except:
        m2pow = beta  # default to beta
    z = row.alpha3
    m1 = row.mass1 / (1. + z)  # XML stores the redshifted (detector frame) mass, we want the source frame mass
    m2 = row.mass2 / (1. + z)

    m1_norm = (1. + alpha) /\
              (mass_dict['max_mass']**(1. + m1pow) - mass_dict['min_mass']**(1. + m1pow))
    m2_norm = (1. + beta) / (m1**(1. + m2pow) - mass_dict['min_mass']**(1. + m2pow))
    sz_norm = 0.5  # two copies of norm of uniform s1,2z distribution

    if zinterp == None:
        zinterp = zpdf_interpolate(omega, zmax, zpower)

    return m1**alpha * m2**beta * zinterp(z) * m1_norm * m2_norm * sz_norm * sz_norm


#
# =============================================================================
#
#                           Utilities
#
# =============================================================================
#

def is_hopeless_hp(row, paramdict, h1_psd, l1_psd, fph, fch, fpl, fcl):
    '''
    Determines if an injection cannot possibly be found.  This is done in a
    very crude manner using the H/L effective distances.  It uses a static
    PSD and computes the SNR at 1 Mpc.  Inclination is set to zero.
    '''
    # generate optimally-oriented waveform at 1 Mpc with the intrinsic
    # parameters specified in the row

    approximant = lalsimulation.SimInspiralGetApproximantFromString(paramdict['approximant'])

    parameters = {
        'm1' : row.mass1 * lal.MSUN_SI,
        'm2' : row.mass2 * lal.MSUN_SI,
        'S1x' : row.spin1x,
        'S1y' : row.spin1y,
        'S1z' : row.spin1z,
        'S2x' : row.spin2x,
        'S2y' : row.spin2y,
        'S2z' : row.spin2z,
        'distance' : 1e6 * lal.PC_SI,
        'inclination' : 0.0,
        'phiRef' : row.coa_phase,
        'longAscNodes' : 0.0,
        'eccentricity' : 0.0,
        'meanPerAno' : 0.0,
        'deltaF' : paramdict['delta_frequency'],
        'f_min' : row.f_lower,
        'f_max' : paramdict['max_frequency'],
        'f_ref' : 0.0,
        'LALpars' : None,
        'approximant' : approximant
    }

    htilde, _ = lalsimulation.SimInspiralChooseFDWaveform(**parameters)

    # compute SNR @ 1 Mpc
    rho1_h = lalsimulation.MeasureSNRFD(htilde, h1_psd, row.f_lower, -1.)
    rho1_l = lalsimulation.MeasureSNRFD(htilde, l1_psd, row.f_lower, -1.)

    rhomin_h = rho1_h / row.eff_dist_h
    rhomin_l = rho1_l / row.eff_dist_l

    if rhomin_h ** 2. + rhomin_l ** 2. < paramdict['snr_threshold'] ** 2.:
        return True, rhomin_h, rhomin_l
    return False, rhomin_h, rhomin_l


def is_hopeless_hp_hc(row, paramdict, h1_psd, l1_psd, fph, fch, fpl, fcl):
    '''
    Determines if an injection cannot possibly be found.  This is done in a
    crude manner but uses more information than is_hopeless_hp().  It uses a
    static PSD as well as distance and inclination information.
    '''
    approximant = lalsimulation.SimInspiralGetApproximantFromString(paramdict['approximant'])

    parameters = {
        'm1' : row.mass1 * lal.MSUN_SI,
        'm2' : row.mass2 * lal.MSUN_SI,
        'S1x' : row.spin1x,
        'S1y' : row.spin1y,
        'S1z' : row.spin1z,
        'S2x' : row.spin2x,
        'S2y' : row.spin2y,
        'S2z' : row.spin2z,
        'distance' : row.distance * 1e6 * lal.PC_SI,
        'inclination' : row.inclination,
        'phiRef' : row.coa_phase,
        'longAscNodes' : 0.0,
        'eccentricity' : 0.0,
        'meanPerAno' : 0.0,
        'deltaF' : paramdict['delta_frequency'],
        'f_min' : row.f_lower,
        'f_max' : paramdict['max_frequency'],
        'f_ref' : 0.0,
        'LALpars' : None,
        'approximant' : approximant
    }

    hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(**parameters)

    htilde_h_data = fph * hptilde.data.data + fch * hctilde.data.data
    htilde_l_data = fpl * hptilde.data.data + fcl * hctilde.data.data
    hptilde.data.data = htilde_h_data
    hctilde.data.data = htilde_l_data
    hptilde.sampleUnits = lal.StrainUnit * lal.SecondUnit  # hack since not all wf give correct freq series units
    hctilde.sampleUnits = lal.StrainUnit * lal.SecondUnit  # hack since not all wf give correct freq series units

    rho_h = lalsimulation.MeasureSNRFD(hptilde, h1_psd, row.f_lower, -1.)
    rho_l = lalsimulation.MeasureSNRFD(hctilde, l1_psd, row.f_lower, -1.)

    if rho_h ** 2. + rho_l ** 2. < paramdict['snr_threshold'] ** 2.:
        return True, rho_h, rho_l
    return False, rho_h, rho_l


def is_hopeless_generic(row, paramdict, h1_psd, l1_psd, fph, fch, fpl, fcl):
    '''
    Determines if an injection cannot possibly be found.
    '''
    snr = dict.fromkeys(("H1", "L1"), 0.0)
    approximant = lalsimulation.SimInspiralGetApproximantFromString(paramdict['approximant'])

    parameters = {
        'm1' : row.mass1 * lal.MSUN_SI,
        'm2' : row.mass2 * lal.MSUN_SI,
        'S1x' : row.spin1x,
        'S1y' : row.spin1y,
        'S1z' : row.spin1z,
        'S2x' : row.spin2x,
        'S2y' : row.spin2y,
        'S2z' : row.spin2z,
        'distance' : row.distance * 1e6 * lal.PC_SI,
        'inclination' : row.inclination,
        'phiRef' : row.coa_phase,
        'longAscNodes' : 0.0,
        'eccentricity' : 0.0,
        'meanPerAno' : 0.0,
        'deltaT' : 1.0 / 16384.,
        'f_min' : row.f_lower,
        'f_ref' : 0.0,
        'LALparams' : None,
        'approximant' : approximant
    }

    injtime = row.time_geocent
    h_plus, h_cross = lalsimulation.SimInspiralTD(**parameters)

    h_plus.epoch += injtime
    h_cross.epoch += injtime

    for instrument in snr:
        if instrument == 'H1':
            psd = h1_psd
        if instrument == 'L1':
            psd = l1_psd
        h = lalsimulation.SimDetectorStrainREAL8TimeSeries(h_plus, h_cross, row.longitude, row.latitude, row.polarization, lalsimulation.DetectorPrefixToLALDetector(instrument))
        snr[instrument] = lalsimulation.MeasureSNR(h, psd, row.f_lower, -1)

    if snr['H1'] ** 2. + snr['L1'] ** 2. < paramdict['snr_threshold'] ** 2.:
        return True, snr['H1'], snr['L1']
    return False, snr['H1'], snr['L1']


def is_hopeless_skip(row, paramdict, h1_psd, l1_psd, fph, fch, fpl, fcl):
    '''
    Every injection is hopeless and has zero SNR.
    '''
    return True, 0., 0.


def is_hopeless_accept(row, paramdict, h1_psd, l1_psd, fph, fch, fpl, fcl):
    '''
    Every injection is a success and has network SNR exactly equal to threshold.
    '''
    return False, paramdict['snr_threshold'] / (2. ** 0.5), paramdict['snr_threshold'] / (2. ** 0.5)


def draw_sim_inspiral_row(paramdict, h1_psd, l1_psd, omega):
    '''
    Yields sim_inspiral rows drawn using the distributions described above,
    and with increasing random GPS times, until gps_end_time is reached.
    '''
    accept = 0
    reject = 0

    extra_params = {
        'process_id':         "process:process_id:0",
        'waveform':           paramdict['waveform'],
        'source':             "",
        'psi0':               0,
        'psi3':               0,
        'alpha':              0,
        'alpha1':             0,
        'alpha2':             0,
        'alpha3':             0,
        'alpha4':             0,
        'alpha5':             0,
        'alpha6':             0,
        'beta':               0,
        'theta0':             0,
        'phi0':               0,
        'f_lower':            paramdict['min_frequency'],
        'f_final':            0,
        'numrel_mode_min':    0,
        'numrel_mode_max':    0,
        'numrel_data':        "",
        'amp_order':          -1,
        'taper':              "TAPER_START",
        'bandpass':           0
    }

    detectors = {
        'g' : lal.CachedDetectors[lal.GEO_600_DETECTOR],
        'h' : lal.CachedDetectors[lal.LHO_4K_DETECTOR],
        'l' : lal.CachedDetectors[lal.LLO_4K_DETECTOR],
        't' : lal.CachedDetectors[lal.TAMA_300_DETECTOR],
        'v' : lal.CachedDetectors[lal.VIRGO_DETECTOR]
    }

    mass_distr = {
        'IMF_PAIR' : draw_mass_pair_imf,
        'IMF_M2SQ' : draw_mass_pair_power,  # keep for backward compatibilty
        'POWER_PAIR' : draw_mass_pair_power,
        'UNIFORM_PAIR' : draw_mass_pair_uniform,
        'UNIFORMLNM_PAIR' : draw_mass_pair_uniform_in_log_mass,
        'NORMAL_PAIR' : draw_mass_pair_normal,
        'UNIFORM_DISTINCT' : draw_mass_distinct_uniform,
        'UNIFORMLNM_DISTINCT' : draw_mass_distinct_uniform_in_log_mass,
        'UNIFORM_UNIFORMLNM_DISTINCT' : draw_mass_distinct_uniform_uniforminlog,
        'NORMAL_IMF_DISTINCT' : draw_mass_distinct_normal_imf,
	'UNIFORM_IMF_DISTINCT' : draw_mass_distinct_uniform_imf
    }

    spin_distr = {
        'ALIGNED' : draw_spin_aligned,
        'ALIGNED_EQUAL' : draw_spin_aligned,
        'ISOTROPIC' : draw_spin_isotropic,
        'ALIGNED_ALIGNED' : draw_spin_aligned_aligned,
        'ISOTROPIC_ALIGNED' : draw_spin_isotropic_aligned,
        'ISOTROPIC_ISOTROPIC' : draw_spin_isotropic_isotropic
    }

    snr_calc = {
        'OPTIMALLY_ORIENTED_1MPC' : is_hopeless_hp,
        'INJ_PARAMS' : is_hopeless_hp_hc,
        'GENERIC' : is_hopeless_generic,
        'SKIP' : is_hopeless_skip,
        'ACCEPT' : is_hopeless_accept
    }

    draw_mass_pair = mass_distr[paramdict['mass_distribution']]
    draw_spin = spin_distr[paramdict['spin_distribution']]

    if paramdict['redshift_power'] is not None:
        print('Using constant comoving rate density * (1+z)**%.1f' % paramdict['redshift_power'])
        random_redshift = iter(draw_redshift(
            paramdict['max_redshift'], omega, paramdict['redshift_power']))
    else:
        random_redshift = iter(draw_redshift(paramdict['max_redshift'], omega))
    random_mass_pair = iter(draw_mass_pair(paramdict))
    random_spin = iter(draw_spin(paramdict))

    approx_snr = snr_calc[paramdict['snr_calculation']]

    if not paramdict['randomize_start_time']:
        t0 = lal.LIGOTimeGPS(paramdict['gps_start_time'])
    else:
        t0 = lal.LIGOTimeGPS(paramdict['gps_start_time']) + numpy.random.uniform(0, 20)
    simid = 0

    while True:

        # determine the next arrival time
        t = t0 + simid * paramdict['time_step']
        # Jitter the time
        tj = lal.LIGOTimeGPS(numpy.random.uniform(t - paramdict['time_interval'], t + paramdict['time_interval']))
        if tj >= paramdict['gps_end_time']:
            break

        # create a new sim_inspiral row
        row = lsctables.New(lsctables.SimInspiralTable).RowType()

        # set the garbage columns
        for k, v in extra_params.items():
            setattr(row, k, v)

        row.geocent_end_time = tj.gpsSeconds
        row.geocent_end_time_ns = tj.gpsNanoSeconds
        row.end_time_gmst = lal.GreenwichMeanSiderealTime(tj)

        # rejection method to cut out hopeless injections
        while True:

            # draw extrinsic params
            z = next(random_redshift)
            row.inclination = numpy.arccos(numpy.random.uniform(-1.0, 1.0))
            row.coa_phase = numpy.random.uniform(0.0, 2.0 * numpy.pi)
            row.polarization = numpy.random.uniform(0.0, 2.0 * numpy.pi)
            row.longitude = numpy.random.uniform(0.0, 2.0 * numpy.pi)
            row.latitude = numpy.arcsin(numpy.random.uniform(-1.0, 1.0))
            row.distance = lal.LuminosityDistance(omega, z)
            row.alpha3 = z  # hack to record redshift in unused column

            # draw intrinsic params
            row.mass1, row.mass2 = next(random_mass_pair)
            row.mass1 *= 1.0 + z
            row.mass2 *= 1.0 + z
            if paramdict['spin_distribution'] in ['ALIGNED_ALIGNED', 'ISOTROPIC_ALIGNED', 'ISOTROPIC_ISOTROPIC']:
                row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z = next(random_spin)
            elif paramdict['spin_distribution'] == 'ALIGNED_EQUAL':
                row.spin1x, row.spin1y, row.spin1z = next(random_spin)
                row.spin2x, row.spin2y, row.spin2z = row.spin1x, row.spin1y, row.spin1z
            else:
                row.spin1x, row.spin1y, row.spin1z = next(random_spin)
                row.spin2x, row.spin2y, row.spin2z = next(random_spin)

            # calculate values of derived columns
            row.eta = row.mass1 * row.mass2 / (row.mass1 + row.mass2)**2
            row.mchirp = row.eta**0.6 * (row.mass1 + row.mass2)

            # calculate and set detector-specific columns
            for site, det in detectors.items():
                tend = tj + lal.TimeDelayFromEarthCenter(det.location, row.longitude, row.latitude, tj)
                fp, fc = lal.ComputeDetAMResponse(det.response, row.longitude, row.latitude, row.polarization, row.end_time_gmst)
                if site == 'h':
                    fph = fp
                    fch = fc
                if site == 'l':
                    fpl = fp
                    fcl = fc
                cosi = numpy.cos(row.inclination)
                deff = row.distance * ((0.5 * (1. + cosi**2.) * fp) ** 2. + (cosi * fc)**2.) ** -0.5
                setattr(row, site + "_end_time", tend.gpsSeconds)
                setattr(row, site + "_end_time_ns", tend.gpsNanoSeconds)
                setattr(row, "eff_dist_" + site, deff)

            hopeless, h_snr, l_snr = approx_snr(row, paramdict, h1_psd, l1_psd, fph, fch, fpl, fcl)
            row.alpha4 = h_snr
            row.alpha5 = l_snr

            if not hopeless:
                accept += 1
                break

            reject += 1

        # set the simulation_id
        row.simulation_id = "sim_inspiral:simulation_id:%d" % simid
        simid += 1

        yield accept, reject, row

