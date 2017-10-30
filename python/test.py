import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import tools
from instrument import ScanStrategy

def scan1(lmax=700, mmax=5, fwhm=40, ra0=-10, dec0=-57.5,
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
    rot_perio : int
        The instrument rotation period [s]

    '''

    # Load up alm and blm
    cls = np.loadtxt('../ancillary/wmap7_r0p03_lensed_uK_ext.txt', unpack=True) # Cl in uK^2
    ell, cls = cls[0], cls[1:]
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    blm = tools.gauss_blm(fwhm, lmax, pol=False)
    blm = tools.get_copol_blm(blm.copy())

    # init scan strategy and instrument
    b2 = ScanStrategy(2*60*60, # mission duration in sec.
                      sample_rate=10, # 10 Hz sample rate
                      location='spole', # South pole instrument
                      nside_out=256
                      )

    # Calculate spinmaps, stored internally
    print('\nCalculating spin-maps...')
    b2.get_spinmaps(alm, blm, mmax, verbose=False)
    print('...spin-maps stored')

    # Initiate focal plane
    b2.set_focal_plane(nrow=10, ncol=10, fov=10)
    # Rotate instrument (period in sec)  
    b2.set_instr_rot(period=rot_period)
    # calculate tod in chunks of # samples
    chunks = b2.partition_mission(int(60*60*b2.fsamp))
    # Allocate and assign parameters for mapmaking
    b2.allocate_maps()
    # Generating timestreams + maps and storing as attributes
    b2.scan_instrument(az_throw=az_throw, ra0=ra0, dec0=dec0,
                       scan_speed=scan_speed)

    maps = b2.solve_map(vec=b2.vec, proj=b2.proj, copy=True, fill=hp.UNSEEN)
    cond = b2.proj_cond(proj=b2.proj)
    cond[cond == np.inf] = hp.UNSEEN
    
    ## Plotting results
    cart_opts = dict(rot=[ra0, dec0, 0],
                     lonra=[-min(az_throw, 90), min(az_throw, 90)],
                     latra=[-min(0.75*az_throw, 45), min(0.75*az_throw, 45)],
                     unit=r'[$\mu K_{\mathrm{CMB}}$]')

    # plot solved maps
    plt.figure()
    hp.cartview(maps[0], min=-250, max=250, **cart_opts)
    plt.savefig('../scratch/img/test_map_I.png')
    plt.close()

    plt.figure()
    hp.cartview(maps[1], min=-5, max=5, **cart_opts)
    plt.savefig('../scratch/img/test_map_Q.png')
    plt.close()

    plt.figure()
    hp.cartview(maps[2], min=-5, max=5, **cart_opts)
    plt.savefig('../scratch/img/test_map_U.png')
    plt.close()

    nside = hp.get_nside(maps[0])
    # plot the input map and spectra
    maps_raw = hp.alm2map(alm, nside)
    plt.figure()
    hp.cartview(maps_raw[0], min=-250, max=250, **cart_opts)
    plt.savefig('../scratch/img/raw_map_I.png')
    plt.close()

    maps_raw = hp.alm2map(alm, nside)
    plt.figure()
    hp.cartview(maps_raw[1], min=-5, max=5, **cart_opts)
    plt.savefig('../scratch/img/raw_map_Q.png')
    plt.close()

    maps_raw = hp.alm2map(alm, nside)
    plt.figure()
    hp.cartview(maps_raw[2], min=-5, max=5, **cart_opts)
    plt.savefig('../scratch/img/raw_map_U.png')
    plt.close()

    cart_opts.pop('min', None)
    cart_opts.pop('max', None)
    cart_opts.pop('unit', None)
    plt.figure()
    hp.cartview(cond, min=2, max=5, unit='condition number', 
                **cart_opts)
    plt.savefig('../scratch/img/test_map_cond.png')
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

    scan1(lmax=300, mmax=2, fwhm=2, az_throw=50, rot_period=3*60, dec0=-60)

