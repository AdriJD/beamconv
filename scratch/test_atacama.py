import os
import sys
import time, datetime
sys.path.append('../python/')
from warnings import catch_warnings, simplefilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import healpy as hp
import tools
from instrument import ScanStrategy, MPIBase, Instrument
from detector import Beam
from plot_tools import *

def get_cls(fname='../ancillary/wmap7_r0p03_lensed_uK_ext.txt'):
    '''
    Load a set of LCDM power spectra.
    
    Keyword arguments
    -----------------
    fname : str
        Absolute path to file
    '''
    
    cls = np.loadtxt('../ancillary/wmap7_r0p03_lensed_uK_ext.txt',
                     unpack=True) # Cl in uK^2
    return cls[0], cls[1:]
 


def scan_atacama(lmax=100, mmax=5, fwhm=40, mlen=24,
                 ra0=[-10, 170], dec0=[-57.5, 0],
                 sample_rate=11.31, nrow=1, ncol=1, fov=5,
                 az_throw=50, scan_speed=1, rot_period=0,
                 hwp_mode='continuous'):
    '''
    Simulates 48h of an atacama-based telescope with a 3 x 3 grid
    of Gaussian beams pairs. Prefers to scan the bicep patch but 
    will try to scan the ABS_B patch if the first is not visible.

    Keyword arguments
    ---------

    lmax : int
        bandlimit (default : 700)
    mmax : int
        assumed azimuthal bandlimit beams (symmetric in this example
        so 2 would suffice) (default : 5)
    fwhm : float
        The beam FWHM in arcmin (default : 40)
    ra0 : float, array-like
        Ra coord of centre region (default : [-10., 85.])
    dec0 : float, array-like
        Ra coord of centre region (default : [-57.5, 0.])
    az_throw : float
        Scan width in azimuth (in degrees) (default : 10)
    scan_speed : float
        Scan speed in deg/s (default : 1)
    rot_period : float
        The instrument rotation period in sec
        (default : 600)
    hwp_mode : str, None
        HWP modulation mode, either "continuous", 
        "stepped" or None. Use freq of 1 or 1/10800 Hz
        respectively (default : continuous)
    '''

    mlen *= 3600 #mission length

    # Create LCDM realization
    ell, cls = get_cls()
    np.random.seed(25) # make sure all MPI ranks use the same seed
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    ctime0 = time.mktime(datetime.datetime.
        strptime('2017 Nov 10', "%Y %b %d").timetuple())

    ac = ScanStrategy(mlen, # mission duration in sec.
        sample_rate=sample_rate, # sample rate in Hz
        ctime0=ctime0, # Time when scanning begins
        location='atacama') # Instrument at south pole 

    # Create a 3 x 3 square grid of Gaussian beams
    ac.create_focal_plane(nrow=nrow, ncol=ncol, fov=fov, 
                          lmax=lmax, fwhm=fwhm)

    # calculate tods in two chunks
    chunks = ac.partition_mission(chunksize=0.02*ac.mlen*ac.fsamp) 

    print('len chunks = {:d}'.format(len(chunks)))

    # Allocate and assign parameters for mapmaking
    ac.allocate_maps(nside=256)

    # set instrument rotation
    ac.set_instr_rot(period=rot_period)

    # Set HWP rotation
    if hwp_mode == 'continuous':
        ac.set_hwp_mod(mode='continuous', freq=1.)
    elif hwp_mode == 'stepped':
        ac.set_hwp_mod(mode='stepped', freq=1/(3*60*60.))

    # Generate timestreams, bin them and store as attributes
    ac.scan_instrument_mpi(alm, verbose=1, ra0=ra0,
        dec0=dec0, az_throw=az_throw, nside_spin=256, el_min=45)
    
    # Solve for the maps
    maps, cond, proj = ac.solve_for_map(fill=np.nan, return_proj=True)

    hits = proj[0]

    # Plotting
    if ac.mpi_rank == 0:
        print 'plotting results'        

        moll_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]')

        # plot rescanned maps
        plot_iqu(maps, 'img/atacama', 'rescan_atacama', 
                 sym_limits=[250, 5, 5], 
                 plot_func=hp.mollview, **moll_opts)

        # plot smoothed input maps
        nside = hp.get_nside(maps[0])
        hp.smoothalm(alm, fwhm=np.radians(fwhm/60.), verbose=False)
        maps_raw = hp.alm2map(alm, nside, verbose=False)

        plot_iqu(maps_raw, 'img/atacama', 'raw_atacama', 
                 sym_limits=[250, 5, 5], 
                 plot_func=hp.mollview, **moll_opts)

        # plot difference maps
        for arr in maps_raw:
            # replace stupid UNSEEN crap
            arr[arr==hp.UNSEEN] = np.nan

        diff = maps_raw - maps

        plot_iqu(diff, 'img/atacama', 'diff_atacama', 
                 sym_limits=[1e-6, 1e-6, 1e-6], 
                 plot_func=hp.mollview, **moll_opts)

        # plot condition number map
        moll_opts.pop('unit', None)

        plot_map(cond, 'img/atacama', 'cond_atacama',
                 min=2, max=5, unit='condition number',
                 plot_func=hp.mollview, **moll_opts)

        plot_map(hits, 'img/atacama', 'hits_atacama',
                 unit='hits',
                 plot_func=hp.mollview, **moll_opts)


        # plot input spectrum
        cls[3][cls[3]<=0.] *= -1.
        dell = ell * (ell + 1) / 2. / np.pi
        plt.figure()
        for i, label in enumerate(['TT', 'EE', 'BB', 'TE']):
          plt.semilogy(ell, dell * cls[i], label=label)

        plt.legend()
        plt.ylabel(r'$D_{\ell}$ [$\mu K^2_{\mathrm{CMB}}$]')
        plt.xlabel(r'Multipole [$\ell$]')
        plt.savefig('img/atacama/cls.png')
        plt.close()


if __name__ == '__main__':

    scan_atacama()
