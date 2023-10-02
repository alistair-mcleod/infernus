import lal
import cosmology_utils as cutils


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

redshift = 0.05

omega = get_cosmo_params()

print(cutils.surveyed_volume(redshift, omega))

print(cutils.surveyed_distance(redshift, omega))