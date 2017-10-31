import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import healpy as hp
import tools
from instrument import ScanStrategy

def get_cls(fname='../ancillary/wmap7_r0p03_lensed_uK_ext.txt'):

    cls = np.loadtxt('../ancillary/wmap7_r0p03_lensed_uK_ext.txt',
                     unpack=True) # Cl in uK^2
    return cls[0], cls[1:]

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
    rot_period : int
        The instrument rotation period [s]

    '''

    # Load up alm and blm
    ell, cls = get_cls()
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    blm = tools.gauss_blm(fwhm, lmax, pol=False)
    blm = tools.get_copol_blm(blm.copy(), c2_fwhm=fwhm)

    # init scan strategy and instrument
    b2 = ScanStrategy(30*60*60, # mission duration in sec.
                      sample_rate=10, # 10 Hz sample rate
                      location='spole', # South pole instrument
                      nside_spin=256,
                      nside_out=256)

    # Calculate spinmaps, stored internally
    print('\nCalculating spin-maps')
    b2.get_spinmaps(alm, blm, mmax, verbose=False)

    # Initiate focal plane
    b2.set_focal_plane(nrow=3, ncol=3, fov=4)

    # Rotate instrument (period in sec)
    b2.set_instr_rot(period=rot_period, angles=np.arange(0, 360, 10))

    # Set HWP rotation
#    b2.set_hwp_mod(mode='continuous', freq=1.)
    b2.set_hwp_mod(mode='stepped', freq=1/(3*60*60.))

    # calculate tod in chunks of # samples
    chunks = b2.partition_mission(int(30*60*60*b2.fsamp))

    # Allocate and assign parameters for mapmaking
    b2.allocate_maps()

    # Generating timestreams + maps and storing as attributes
    b2.scan_instrument(az_throw=az_throw, ra0=ra0, dec0=dec0,
                       scan_speed=scan_speed)

    maps = b2.solve_map(vec=b2.vec, proj=b2.proj, copy=True,
                        fill=hp.UNSEEN)
    cond = b2.proj_cond(proj=b2.proj)
    cond[cond == np.inf] = hp.UNSEEN


    ## Plotting results
    cart_opts = dict(rot=[ra0, dec0, 0],
            lonra=[-min(0.5*az_throw, 90), min(0.5*az_throw, 90)],
            latra=[-min(0.375*az_throw, 45), min(0.375*az_throw, 45)],
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

    # plot smoothed input maps, diff maps and spectra
    nside = hp.get_nside(maps[0])
    hp.smoothalm(alm, fwhm=np.radians(fwhm/60.), verbose=False)
    maps_raw = hp.alm2map(alm, nside, verbose=False)

    plt.figure()
    hp.cartview(maps_raw[0], min=-250, max=250, **cart_opts)
    plt.savefig('../scratch/img/raw_map_I.png')
    plt.close()

    plt.figure()
    hp.cartview(maps_raw[1], min=-5, max=5, **cart_opts)
    plt.savefig('../scratch/img/raw_map_Q.png')
    plt.close()

    plt.figure()
    hp.cartview(maps_raw[2], min=-5, max=5, **cart_opts)
    plt.savefig('../scratch/img/raw_map_U.png')
    plt.close()

    # plot diff maps
    plt.figure()
    hp.cartview(maps[0] - maps_raw[0], min=-1e-6, max=1e-6, **cart_opts)
    plt.savefig('../scratch/img/diff_map_I.png')
    plt.close()

    plt.figure()
    hp.cartview(maps[1] - maps_raw[1], min=-1e-6, max=1e-6, **cart_opts)
    plt.savefig('../scratch/img/diff_map_Q.png')
    plt.close()

    plt.figure()
    hp.cartview(maps[2] - maps_raw[2], min=-1e-6, max=1e-6, **cart_opts)
    plt.savefig('../scratch/img/diff_map_U.png')
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


def offset_beam(az_off=0, el_off=0, lmax=200, fwhm=100, max_spin=100):
    '''
    Script that scans the sky with a symmetric Gaussian beam that has
    been rotated away from the boresight. This means that the
    beam is highly asymmetric.

    Arguments
    ---------

    az_off :

    el_off :

    lmax :

    fwhm :


    '''

    # Load up alm and blm
    ell, cls = get_cls()

    # set random seed
    np.random.seed(10)
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    # Only use the polarized signal
    alm = (alm[0]*0., alm[1]*1, alm[2]*1)

    blm = tools.gauss_blm(fwhm, lmax, pol=False)
    blm = tools.get_copol_blm(blm.copy(), c2_fwhm=fwhm)

    # init scan strategy and instrument
    b2 = ScanStrategy(4*60, # mission duration in sec.
                      sample_rate=10, # 10 Hz sample rate
                      location='spole', # South pole instrument
                      nside_spin=256,
                      nside_out=256)

    # Calculate spinmaps, stored internally
    print('\nCalculating spin-maps')
    b2.get_spinmaps(alm, blm, max_spin=2, verbose=False)

    # Initiate focal plane
    b2.nrow = 1
    b2.ncol = 1
    b2.ndet = 1
    b2.azs = np.array([az_off])
    b2.els = np.array([el_off])
    b2.polangs = np.array([0])

    # Rotate instrument (period in sec)
    b2.set_instr_rot(period=60)

    # Set HWP rotation
    b2.set_hwp_mod(mode='continuous', freq=25.)

    # calculate tod in chunks of # samples
    chunks = b2.partition_mission(int(4*60*b2.fsamp))

    b2.scan_instrument(mapmaking=False)

    # Store the tod made with symmetric beam
    tod_sym = b2.tod.copy()

    # now repeat with asymmetric beam and no detector offset
    b2.azs = np.array([0])
    b2.els = np.array([0])

    # Beam E and B modes
    blmm2 = blm[1].copy()
    blmE = -blmm2 / 2.
    blmB = -1j * blmm2 / 2.
    blmI = blm[0].copy()

    # Rotate blm to match centroid
    radius = np.arccos(np.cos(np.radians(el_off)) * np.cos(np.radians(az_off)))
    if np.tan(radius) != 0:
        angle = np.arctan2(np.tan(np.radians(el_off)), np.sin(np.radians(az_off))) + np.pi/2.
    else:
        angle = 0.

    print np.degrees(radius), np.degrees(angle)

    q_off = b2.det_offset(az_off, el_off, 0)
    ra, dec, pa = b2.quat2radecpa(b2.det_offset(az_off, el_off, 0))
    print ra, dec, pa
    angle = np.radians(180 - ra)
    radius = np.radians(90 - dec)
    psi = -np.radians(pa)

    print np.degrees(psi), np.degrees(radius), np.degrees(-angle)
    hp.rotate_alm([blmI, blmE, blmB], psi, radius, -angle, lmax=lmax, mmax=lmax)
    #hp.rotate_alm([blmI, blmE, blmB], angle, radius, -angle, lmax=lmax,  mmax=lmax)
    #hp.rotate_alm([blmI, blmE, blmB], psi, radius, -np.radians(ra), lmax=lmax,  mmax=lmax)

    plt.figure()
    plt.plot(np.arange(lmax+1), blm[0][:lmax+1], label='Original Gaussian')
    plt.plot(np.arange(lmax+1), blmI[:lmax+1], label='Rotated Gaussian')
    plt.xlabel('Multipole, $\ell$')
    plt.ylabel('Angular response')
    plt.legend()
    plt.savefig('../scratch/img/bl.png')
    plt.close()

    blmp2 = -1 * (blmE + 1j * blmB)
    blmm2 = -1 * (blmE - 1j * blmB)

    blm = (blmI, blmm2, blmp2)

    print('\nCalculating spin-maps...')
    b2.get_spinmaps(alm, blm, max_spin=max_spin, verbose=False)
    print('...spin-maps stored')

    # Reset instrument rotation and HWP
    b2.set_instr_rot(period=60)
    b2.set_hwp_mod(mode='continuous', freq=25.)
    b2.scan_instrument(mapmaking=False)

    plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[:2, :])
    ax2 = plt.subplot(gs[-1, :])
    samples = np.arange(tod_sym.size)
    ax1.plot(samples, b2.tod, label='Asymmetric Gaussian')
    ax1.plot(samples, tod_sym, label='Symmetric Gaussian', alpha=0.5)#, ls=':')
    ax1.legend()

    ax1.tick_params(labelbottom='off')
    ax2.plot(samples, b2.tod - tod_sym)
    ax1.set_ylabel('Signal')
    ax2.set_ylabel('Difference')
    ax2.set_xlabel('Sample number')

    plt.savefig('../scratch/img/tods.png')
    plt.close()

def single_detector(nsamp=1000):
    '''
    Generates a timeline for a set of individual detectors scanning the sky. The
    spatial response of these detectors is described by a 1) Gaussian, 2) an
    elliptical Gaussian and, 3) a beam map derived by physical optics.

    Arguments
    ---------

    nsamp : int (default: 1000)
        The length of the generated timestreams in number of samples

    '''

    # Load up alm and blm
    cls = np.loadtxt('../ancillary/wmap7_r0p03_lensed_uK_ext.txt', unpack=True) # Cl in uK^2
    ell, cls = cls[0], cls[1:]
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    blm = tools.gauss_blm(fwhm, lmax, pol=False)
    blm = tools.get_copol_blm(blm.copy())

    # init scan strategy and instrument
    ss = ScanStrategy(nsamp/10., # mission duration in sec.
                      sample_rate=10, # 10 Hz sample rate
                      location='spole', # South pole instrument
                      nside_out=256)

    # Calculate spinmaps, stored internally

    ss.get_spinmaps(alm, blm, mmax, verbose=False)

    #### FINISH THIS ####


if __name__ == '__main__':

#    scan1(lmax=1200, mmax=2, fwhm=40, az_throw=50, rot_period=1*60*60, dec0=-10)
    offset_beam(az_off=15, el_off=-5)
