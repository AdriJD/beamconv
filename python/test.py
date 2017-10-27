import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import tools
from instrument import ScanStrategy

def scan1(lmax=700, mmax=5, fwhm=40, nside=256, ra0=-10, dec0=-57.5,
    az_throw=10, scan_speed=1, rot_period=10*60):
    '''
    Simulates a fraction of BICEP2-like scan strategy

    Arguments
    ---------

    lmax : int
        bandlimit
    mmax : int
        assumed azimuthal bandlimit beams (symmetric in this example)
    fwhm : float
        The beam FWHM in arcmin
    nside : int
        The nside value of the output map
    rot_perio : int
        The instrument rotation period [s]


    '''

    # Load up alm and blm
    cls = np.loadtxt('../ancillary/wmap7_r0p03_lensed_uK_ext.txt', unpack=True) # Cl in uK^2
    ell, cls = cls[0], cls[1:]
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    blm, blmm2 = tools.gauss_blm(fwhm, lmax, pol=True)
    blm = tools.get_copol_blm(blm.copy())

    # init scan strategy and instrument
    b2 = ScanStrategy(20*60*60, # mission duration in sec.
                      sample_rate=10, # 10 Hz sample rate
                      location='spole', # South pole instrument
                      )

    # Calculate spinmaps, stored internally
    print('\nCalculating spin-maps...')
    b2.get_spinmaps(alm, blm, mmax, verbose=False)
    print('...spin-maps stored')

    # Initiate a single detector
    b2.set_focal_plane(nrow=8, ncol=8, fov=10)
    # Rotate instrument (period in sec)
    b2.set_instr_rot(period=rot_period)
    # calculate tod in chunks of # samples
    chunks = b2.partition_mission(int(60*60*b2.fsamp))
    # Allocate and assign parameters for mapmaking
    b2.allocate_maps()
    # Generating timestreams + maps and storing as attributes
    b2.scan_instrument()

    # just solve for the unpolarized map for now (condition number is terrible obviously)
    maps = b2.solve_map(vec=b2.vec[0], proj=b2.proj[0], copy=True)

    ## Plotting results
    # plot solved T map
    plt.figure()
    moll_opts = dict(min=-250, max=250)
    hp.mollview(maps, **moll_opts)
    plt.savefig('../scratch/img/test_map_I.png')
    plt.close()

    # plot the input map and spectra
    maps_raw = hp.alm2map(alm, 256)
    plt.figure()
    hp.mollview(maps_raw[0], **moll_opts)
    plt.savefig('../scratch/img/raw_map_I.png')
    plt.close()

    cls[3][cls[3]<=0.] *= -1.
    dell = ell * (ell + 1) / 2. / np.pi
    plt.figure()
    for i, label in enumerate(['TT', 'EE', 'BB', 'TE']):
      plt.semilogy(ell, dell * cls[i], label=label)

    plt.legend()
    plt.ylabel(r'$D_{\ell}$ [$\mu K^2_{\mathrm{CMB}}$]')
    plt.xlabel(r'Multipole [$\ell$]')
    plt.savefig('../scratch/img/cls.png')
    plt.close()


if __name__ == '__main__':

    scan1()
