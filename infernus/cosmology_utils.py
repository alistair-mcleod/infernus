#  Copyright (C) 2017 Jolien Creighton
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
# The python module for utilities needed for cosmological calculations.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import numpy
import scipy.integrate

import lal


#
# =============================================================================
#
#                      Cosmological and Other Constants
#
# =============================================================================
#

# FIXME: go from astropy instead
def get_cosmo_params():
    # From Planck2015, Table IV
    omega = lal.CreateCosmologicalParametersAndRate().omega
    lal.SetCosmologicalParametersDefaultValue(omega)
    omega.h = 0.679
    omega.om = 0.3065
    omega.ol = 0.6935
    omega.ok = 1.0 - omega.om - omega.ol
    omega.w0 = -1.0
    omega.w1 = 0.0
    omega.w2 = 0.0

    return omega

#
# =============================================================================
#
#                               Cosmology Utilities
#
# =============================================================================
#


def surveyed_spacetime_volume(gps_start_time, gps_end_time, max_redshift, omega):
    '''
    Returns the total spacetime volume surveyed:

        <VT> = T \int dz \frac{dV_c}{dz} \frac{1}{1+z}

    Results are given in units Gpc^3 yr (Julian).
    '''
    # Note: LAL's cosmology routines returns distances in Mpc

    def integrand(z, omega):
        '''
        Returns the integrand

            (1 + z) D_A^2(z) / E(z)

        in units of Mpc^2.  Multiply the integral by 4 * pi * D_H
        to get the desired integral.
        '''
        return (1.0 + z) * lal.AngularDistance(omega, z)**2 * lal.HubbleParameter(z, omega)

    I, _ = scipy.integrate.quad(integrand, 0.0, max_redshift, args=omega)

    # multiply by remaining factors and scale to Gpc^3
    V = 4.0 * numpy.pi * lal.HubbleDistance(omega) * I / (1e3)**3
    
    # surveyed time in Julian years
    T = (gps_end_time - gps_start_time) / lal.YRJUL_SI

    return V * T



#modified by Alistair

def surveyed_distance(max_redshift,omega):
    '''
    Returns the total distance surveyed:
    '''
        
    return lal.LuminosityDistance(omega, max_redshift)
    

def surveyed_volume(max_redshift, omega):
    '''
    Returns the total spacetime volume surveyed:

        <VT> = T \int dz \frac{dV_c}{dz} \frac{1}{1+z}

    Results are given in units Gpc^3 yr (Julian).
    '''
    # Note: LAL's cosmology routines returns distances in Mpc

    def integrand(z, omega):
        '''
        Returns the integrand

            (1 + z) D_A^2(z) / E(z)

        in units of Mpc^2.  Multiply the integral by 4 * pi * D_H
        to get the desired integral.
        '''
        return (1.0 + z) * lal.AngularDistance(omega, z)**2 * lal.HubbleParameter(z, omega)

    I, _ = scipy.integrate.quad(integrand, 0.0, max_redshift, args=omega)

    # multiply by remaining factors and scale to Gpc^3
    V = 4.0 * numpy.pi * lal.HubbleDistance(omega) * I #/ (1e3)**3

    #convert V to distance
    D = (3*V/(4*numpy.pi))**(1/3)


    return D,V