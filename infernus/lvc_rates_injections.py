#!/usr/bin/env python
#
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
#

## @file lvc_rates_injections
# The rates injection generator.
#
# ### Usage examples
# - Please add some!
#
# ### Command line interface

__author__ = 'Jolien Creighton <jolien.creighton@ligo.org>'

import sys
from optparse import OptionParser
import numpy
import numpy.random
import scipy.integrate
from glue.ligolw import utils
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw.utils import process as ligolw_process
import lal
import lalsimulation

import cosmology_utils as cutils
import injection_utils as injutils

def parse_command_line():
    parser = OptionParser(description = __doc__)
    parser.add_option("--gps-start-time", type="int", help="Start time of the injection data to be created.")
    parser.add_option("--gps-end-time", type="int", help="End time of the injection data to be created.")
    parser.add_option("--use-segments", default=False, action="store_true", help="Not currently implemented. Cut on segments.")
    parser.add_option("--output-tag", metavar="tag", default="injections", help="String to name output injection xml file.")

    # Quantities for constructing injection parameter distributions
    parser.add_option("--seed", default=7, type="int", help="Seed of the random number generator to get reproducible results, default=7")
    parser.add_option("--max-redshift", type="float", help="Maximum redshift out to which injections will be placed.")
    parser.add_option("--redshift-power", type="float", help="Power of (1+z) to multiply the constant-comoving-rate distribution by.")
    parser.add_option("--time-step", type="float", help="Average time step between injections. Required")
    parser.add_option("--time-interval", type="float", help="Size of the interval used to randomly place injections around the time step (s). Ex. 10 will result in +-10s interval either side of time step.")
    parser.add_option("--randomize-start-time", default=False, action="store_true", help="Add a small random jitter to the gps-start-time")
    parser.add_option("--mass-distribution", help="Component mass distribution. Select one of the following: "
                      "1. IMF_PAIR - Primary from Salpeter IMF distribution, secondary drawn uniformly between min_mass and m1. "
                      "2. UNIFORM_PAIR - Both components drawn from uniform distribution so that min_mass <= m1,m2 < max_mass. "
                      "3. UNIFORMLNM_PAIR - Both components drawn from uniform-in-log distribution. "
                      "4. NORMAL_PAIR - Both components drawn from normal distribution of mean mean_mass and standard deviation sigma_mass. "
                      "5. UNIFORM_DISTINCT - Components drawn from two distinct uniform mass distributions. "
                                            "Must specify min_mass1, max_mass1, min_mass2, max_mass2. "
                      "6. UNIFORMLNM_DISTINCT - Components drawn from two distinct uniform-in-log mass distributions. "
                                               "Must specify min_mass1, max_mass1, min_mass2, max_mass2. "
                      "7. UNIFORM_UNIFORMLNM_DISTINCT - Primary drawn from a uniform mass distribution, secondary drawn "
                          "from a uniform-in-log mass distribution. Must specify min_mass1, max_mass1, min_mass2, max_mass2. "
                      "8. NORMAL_IMF_DISTINCT - Primary drawn from a normal mass distribution, secondary drawn from a Salpeter IMF distribution. "
                      "9. UNIFORM_IMF_DISTINCT - Primary drawn from a uniform mass distribution, secondary drawn from a Salpeter IMF distribution. "
                      "10. POWER_PAIR - Primary from power law, secondary from another power law conditional on m2 < m1. "
                      "Default power laws m1**2.35, m2**2 reproduce the IMF_M2SQ distribution.")
    parser.add_option("--min-mass", type="float", help="Minimum mass of the two compact objects.")
    parser.add_option("--max-mass", type="float", help="Maximum mass of the two compact objects.")
    parser.add_option("--max-mtotal", type = "float", help="Set a cutoff on the allowed total mass. Currently implemented for "
                      "IMF_PAIR, UNIFORMLNM_PAIR, UNIFORM_DISTINCT, UNIFORMLNM_DISTINCT, UNIFORM_UNIFORMLNM_DISTINCT, NORMAL_IMF_DISTINCT.")
    parser.add_option("--mean-mass", type="float", help="Mean of normal mass distribution. Required if using NORMAL_PAIR.")
    parser.add_option("--sigma-mass", type="float", help="Stdev of normal mass distribution. Required if using NORMAL_PAIR.")
    parser.add_option("--min-mass1", type="float", help="Minimum mass of the primary.")
    parser.add_option("--max-mass1", type="float", help="Maximum mass of the primary.")
    parser.add_option("--min-mass2", type="float", help="Minimum mass of the secondary.")
    parser.add_option("--max-mass2", type="float", help="Maximum mass of the secondary.")
    parser.add_option("--mass1-power", type="float", default=2.35, help="Power law index for primary mass. Default 2.35")
    parser.add_option("--mass2-power", type="float", default=2., help="Power law index for secondary mass. Default 2")
    parser.add_option("--spin-distribution", help="Component spin distribution. Select one of the following: "
                      "1. ALIGNED - Spinz uniform in (-max_spin, +max_spin) for each component "
                      "2. ISOTROPIC - (s_x, s_y, s_z) isotropically distributed with uniform magnitude distribution for each component "
                      "3. ALIGNED_ALIGNED - Components uniform in (-max_spin1, +max_spin1) and (-max_spin2, +max_spin2)" +
                      "4. ISOTROPIC_ALIGNED - Component 2 spinz uniform in (-max_spin1, +max_spin1), component 1 isotropically distributed "
                      "5. ISOTROPIC_ISOTROPIC - Both components isotropically distributed with uniform magnitude distribution "
                      "6. ALIGNED_EQUAL - Spinz uniform in (-max_spin, +max_spin) for component 1, spin2z = spin1z.")
    parser.add_option("--max-spin", type="float", help="Maximum component spin magnitude. Required if using ALIGNED, ALIGNED_EQUAL or ISOTROPIC.")
    parser.add_option("--max-spin1", type="float", help="Maximum component spin magnitude of mass_1. Required if using ALIGNED_ISOTROPIC.")
    parser.add_option("--max-spin2", type="float", help="Maximum component spin magnitude of mass_2. Required if using ALIGNED_ISOTROPIC.")
    parser.add_option("--waveform", help="Waveform family name to populate the sim_inspiral table.")

    # Quantities for computing expected SNRs
    parser.add_option("--snr-calculation",
                      help="Method for computing the approximate injection snr. Select one of the following: "
                      "1. OPTIMALLY_ORIENTED_1MPC - Assumes inclination = 0, distance = 1Mpc. Computes only hp "
                      "2. INJ_PARAMS - Computes hp and hc using injection parameters, uses ChooseFDWaveform "
                      "3. GENERIC - Computes hp and hc using injections' parameters and can take FD and TD waveforms "
                      "4. SKIP - Discard all injections and do not perform SNR calculation")
    parser.add_option("--snr-threshold", default=6., type="float", help="H1L1 network snr threshold to determine "
                      "whether an injection is expected to be found or not. Default=6.")
    parser.add_option("--min-frequency", type="float", help="Lower frequency cutoff for SNR calculation (Hz). Ex. 15")
    parser.add_option("--max-frequency", type="float", help="Upper frequency cutoff for SNR calculation (Hz). Ex. 1500")
    parser.add_option("--delta-frequency", type="float", help="Step in frequency for SNR calculation (Hz). Ex. 1")
    parser.add_option("--approximant", default='SEOBNRv4_ROM', help="Approximant waveform to use in the SNR calculation, "
                      "default = SEOBNRv4_ROM")
#    parser.add_option("--reference-spectrum-file", metavar = "filename", help = "Full path to the file containing two columns for the frequency "
#                      "and the spectral data. It can be ASD or PSD data. The code will try to figure out which data you have provided.")
    parser.add_option("--h1-reference-spectrum-file", metavar="filename", help="Full path to the H1 PSD xmldoc")
    parser.add_option("--l1-reference-spectrum-file", metavar="filename", help="Full path to the L1 PSD xmldoc")

    parser.add_option("-v", "--verbose", action="store_true", help="Be verbose.")

    options, filenames = parser.parse_args()

    process_params = options.__dict__.copy()

    required_options = ("gps_start_time", "gps_end_time", "max_redshift", "time_interval", "time_step", "mass_distribution",
			"spin_distribution", "waveform", "approximant", "h1_reference_spectrum_file", "l1_reference_spectrum_file", 
			"min_frequency", "max_frequency", "delta_frequency")

    missing_options = [option for option in required_options if getattr(options, option) is None]
    if missing_options:
        raise ValueError("missing required option(s) %s" % ", ".join("--%s" % option.replace("_", "-") for option in missing_options))

    return options, process_params, filenames


options, process_params, filenames = parse_command_line()

# seed the random number generator to get reproducible results
numpy.random.seed(options.seed)

# psd and related quantities for computing SNRs
approximant = lalsimulation.SimInspiralGetApproximantFromString(options.approximant)
h1_psd = lal.CreateREAL8FrequencySeries(
    'PSD',
    lal.LIGOTimeGPS(0),
    0.0,
    options.delta_frequency,lal.SecondUnit,
    int(round(options.max_frequency / options.delta_frequency)) + 1)
lalsimulation.SimNoisePSDFromFile(h1_psd, options.min_frequency, options.h1_reference_spectrum_file)
# FIXME: Write code to determine whether we have ASD or PSD
# SimNoisePSDFromFile expects ASD in the file, but this one
# contains the PSD, so take the square root
h1_psd.data.data = h1_psd.data.data ** 0.5
l1_psd = lal.CreateREAL8FrequencySeries(
    'PSD',
    lal.LIGOTimeGPS(0),
    0.0,
    options.delta_frequency,
    lal.SecondUnit,
    int(round(options.max_frequency / options.delta_frequency)) + 1)
lalsimulation.SimNoisePSDFromFile(l1_psd, options.min_frequency, options.l1_reference_spectrum_file)
l1_psd.data.data = l1_psd.data.data ** 0.5

omega = cutils.get_cosmo_params()
VT = cutils.surveyed_spacetime_volume(options.gps_start_time, options.gps_end_time, options.max_redshift, omega)
D = cutils.surveyed_distance(options.max_redshift, omega)
print('surveyed spacetime volume: %g Gpc^3 yr' % VT)
print('surveyed distance: %g Mpc' % D)

xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())
sim_inspiral_table = lsctables.New(lsctables.SimInspiralTable)

process = ligolw_process.register_to_xmldoc(
    xmldoc,
    program="lvc_rates_injections",
    paramdict=process_params,
    comment="Injection parameter generator for rates calculations")

random_sim_inspiral_rows = iter(injutils.draw_sim_inspiral_row(process_params, h1_psd, l1_psd, omega))

for accept, reject, row in random_sim_inspiral_rows:

    row.process_id = process.process_id
    sim_inspiral_table.append(row)
    if (accept % 100) == 0:
        print('(accept,reject)=(%d,%d)\r' % (accept, reject))
        sys.stdout.flush()

        ligolw_process.set_process_end_time(process)

        print('(accept,reject)=(%d,%d)' % (accept, reject))
        sys.stdout.flush()

        acceptance_rate = float(accept)/float(accept + reject)
        print('%s acceptance rate: %g' % (options.output_tag, acceptance_rate))
        if process_params['redshift_power'] is None:  # constant rate in comoving volume-time
            print('%s actual surveyed spacetime volume: %g Gpc^3 yr' % (options.output_tag, VT * acceptance_rate))
sys.stdout.flush()

VT_params = {
    'accept' : accept,
    'reject' : reject,
    'acceptance_rate' : acceptance_rate,
    'VT' : VT
    }

ligolw_process.register_to_xmldoc(
    xmldoc,
    program="lvc_rates_injections_vtparams",
    paramdict=VT_params,
    comment="Acceptance and rejection related statistics for VT calculations.")

xmldoc.childNodes[-1].appendChild(sim_inspiral_table)
output_file = options.output_tag + '-' + str(options.gps_start_time) + '-' + str(options.gps_end_time) + '.xml.gz'
utils.write_filename(xmldoc, output_file, gz=True, verbose=options.verbose)
xmldoc.unlink()

