import numpy as np
import healpy as hp
import qpoint as qp
from beamconv.scanning import *

def main():

    ####################################################
    # All the scanning parameters
    ####################################################

    import time
    import healpy as hp

    theta_antisun = 45      # degrees
    theta_boresight = 50    # degrees
    freq_antisun = 192.348  # minutes
    freq_boresight = 0.314  # radians per minutes
    siad = 86400.            # seconds in a day
    today_julian = 1        # I just set 1 by hand
    sample_rate = 19.1      # Hz
    ydays = 20              # duration of the mission (in days)
    nside = 256
    runtime_i = time.time()

    theta_antisun = theta_antisun/radeg
    theta_boresight = theta_boresight/radeg
    freq_antisun = 1./(freq_antisun*60.)
    freq_boresight = freq_boresight/60.

    ####################################################
    # This is a minimal version of the gen_scan_c_mod2
    # function of pyScan
    ####################################################

    # we don't really need the next two lines, I just
    # wanted to look at the nhits map
    npix = hp.nside2npix(nside)
    nhits = np.zeros(npix)
    theta_array = np.zeros(int(sample_rate * siad * ydays))
    phi_array = np.zeros(int(sample_rate * siad * ydays))

    for i in range(ydays):
        # define time related variables
        n_time = int(siad * sample_rate)
        sec2date = (1.e-4/8.64)
        time_julian = np.arange(n_time)/(n_time-1.)*siad*sec2date + today_julian # time in julian [day]

        # cal sun position
        DJD = convert_Julian2Dublin(time_julian)
        phi_asp, theta_asp = sun_position_quick(DJD)
        DJD = 0

        time_i = time_julian /sec2date # time in [sec]

        # from ecliptic lat, lon convention to theta, phi convention
        theta_asp = pi/2.-theta_asp

        omega_pre = 2.*pi*freq_antisun
        omega_spin = 2.*pi*freq_boresight

        p_out = LB_rotmatrix_multi2(theta_asp, phi_asp, theta_antisun,
            theta_boresight, omega_pre, omega_spin, time_i)

        theta_out = np.arctan2(np.sqrt(p_out[0,:]**2 + p_out[1,:]**2), p_out[2,:])
        phi_out = np.arctan2(p_out[1,:],p_out[0,:])

        theta_out = wraparound_npi(theta_out, 1.)
        phi_out = wraparound_npi(phi_out, 2.)

        theta_array[i*int(sample_rate*siad):(i+1)*int(sample_rate*siad)] = theta_out
        phi_array[i*int(sample_rate*siad):(i+1)*int(sample_rate*siad)] = phi_out

        ####################################################
        # we don't really need the next lines, I just
        # wanted to look at the nhits map
        nbPix = hp.nside2npix(nside)
        ipix = hp.ang2pix(nside,theta_out,phi_out)
        beta=0; dphivec=0; dpvec=0; dtvec=0; theta_out=0; phi_out=0
        nhits += np.bincount(ipix, minlength=nbPix)
        ipix=0
        runtime_f = time.time()
        print('day ', (i+1), ':', (time.time()-runtime_i), 'sec')
        print()

        today_julian += 1.

    filename_out = str(ydays) + '_days'
    hp.mollview(nhits, title= str(ydays) + ' days', rot=[0.,0.])
    hp.graticule(dpar=10,dmer=10,coord='E')
    py.savefig(filename_out+'.png')

if __name__ == '__main__':
    main()


