import os
import sys
from warnings import catch_warnings, simplefilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import healpy as hp
import tools
from beamconv import ScanStrategy, MPIBase, Instrument
from beamconv import Beam
from plot_tools import plot_map, plot_iqu

def get_cls(fname='../ancillary/wmap7_r0p03_lensed_uK_ext.txt'):
    '''
    Load a set of LCDM power spectra.

    Keyword arguments
    -----------------
    fname : str
        Absolute path to file
    '''

    cls = np.loadtxt(fname, unpack=True) # Cl in uK^2
    return cls[0], cls[1:]

def scan_bicep(lmax=700, mmax=5, fwhm=43, ra0=-10, dec0=-57.5,
    az_throw=50, scan_speed=2.8, rot_period=4.5*60*60,
    hwp_mode=None):
    '''
    Simulates a 24h BICEP2-like scan strategy
    using a random LCDM realisation and a 3 x 3 grid
    of Gaussian beams pairs. Bins tods into maps and
    compares to smoothed input maps (no pair-
    differencing). MPI-enabled.

    Keyword arguments
    ---------

    lmax : int,
        bandlimit (default : 700)
    mmax : int,
        assumed azimuthal bandlimit beams (symmetric in this example
        so 2 would suffice) (default : 5)
    fwhm : float,
        The beam FWHM in arcmin (default : 40)
    ra0 : float,
        Ra coord of centre region (default : -10)
    dec0 : float,  (default : -57.5)
        Ra coord of centre region
    az_throw : float,
        Scan width in azimuth (in degrees) (default : 50)
    scan_speed : float,
        Scan speed in deg/s (default : 1)
    rot_period : float,
        The instrument rotation period in sec
        (default : 600)
    hwp_mode : str, None
        HWP modulation mode, either "continuous",
        "stepped" or None. Use freq of 1 or 1/10800 Hz
        respectively (default : None)
    '''

    mlen = 24 * 60 * 60 # hardcoded mission length

    # Create LCDM realization
    ell, cls = get_cls()
    np.random.seed(25) # make sure all MPI ranks use the same seed
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    b2 = ScanStrategy(mlen, # mission duration in sec.
                      sample_rate=12.01, # sample rate in Hz
                      location='spole') # Instrument at south pole

    # Create a 3 x 3 square grid of Gaussian beams
    b2.create_focal_plane(nrow=3, ncol=3, fov=10,
                          lmax=lmax, fwhm=fwhm, symmetric=True)

    # calculate tods in two chunks
    b2.partition_mission(0.5*b2.nsamp)

    # Allocate and assign parameters for mapmaking
    b2.allocate_maps(nside=256)

    # set instrument rotation
    b2.set_instr_rot(period=rot_period, angles=[68, 113, 248, 293])

    # Set elevation stepping
    b2.set_el_steps(rot_period, steps=[0, 2, 4, 6, 8, 10])

    # Set HWP rotation
    if hwp_mode == 'continuous':
        b2.set_hwp_mod(mode='continuous', freq=1.)
    elif hwp_mode == 'stepped':
        b2.set_hwp_mod(mode='stepped', freq=1/(3*60*60.))


    # Generate timestreams, bin them and store as attributes
    b2.scan_instrument_mpi(alm, verbose=1, ra0=ra0,
                           dec0=dec0, az_throw=az_throw,
                           nside_spin=256,
                           max_spin=mmax)

    # Solve for the maps
    maps, cond = b2.solve_for_map(fill=np.nan)

    # Plotting
    if b2.mpi_rank == 0:
        print('plotting results')
        img_out_path = '../scratch/img/' 

        cart_opts = dict(rot=[ra0, dec0, 0],
                lonra=[-min(0.5*az_throw, 90), min(0.5*az_throw, 90)],
                latra=[-min(0.375*az_throw, 45), min(0.375*az_throw, 45)],
                 unit=r'[$\mu K_{\mathrm{CMB}}$]')

        # plot rescanned maps
        plot_iqu(maps, img_out_path, 'rescan_bicep',
                 sym_limits=[250, 5, 5],
                 plot_func=hp.cartview, **cart_opts)

        # plot smoothed input maps
        nside = hp.get_nside(maps[0])
        hp.smoothalm(alm, fwhm=np.radians(fwhm/60.), verbose=False)
        maps_raw = hp.alm2map(alm, nside, verbose=False)

        plot_iqu(maps_raw, img_out_path, 'raw_bicep',
                 sym_limits=[250, 5, 5],
                 plot_func=hp.cartview, **cart_opts)

        # plot difference maps
        for arr in maps_raw:
            # replace stupid UNSEEN crap
            arr[arr==hp.UNSEEN] = np.nan

        diff = maps_raw - maps

        plot_iqu(diff, img_out_path, 'diff_bicep',
                 sym_limits=[1e-6, 1e-6, 1e-6],
                 plot_func=hp.cartview, **cart_opts)

        # plot condition number map
        cart_opts.pop('unit', None)

        plot_map(cond, img_out_path, 'cond_bicep',
                 min=2, max=5, unit='condition number',
                 plot_func=hp.cartview, **cart_opts)

        # plot input spectrum
        cls[3][cls[3]<=0.] *= -1.
        dell = ell * (ell + 1) / 2. / np.pi
        plt.figure()
        for i, label in enumerate(['TT', 'EE', 'BB', 'TE']):
          plt.semilogy(ell, dell * cls[i], label=label)

        plt.legend()
        plt.ylabel(r'$D_{\ell}$ [$\mu K^2_{\mathrm{CMB}}$]')
        plt.xlabel(r'Multipole [$\ell$]')
        plt.savefig('../scratch/img/cls_bicep.png')
        plt.close()
        
        print("Results written to {}".format(os.path.abspath(img_out_path))) 

def scan_atacama(lmax=700, mmax=5, fwhm=40,
                 mlen = 48 * 60 * 60, nrow=3, ncol=3, fov=5.0,
                 ra0=[-10, 170], dec0=[-57.5, 0], el_min=45.,
                 cut_el_min=False, az_throw=50, scan_speed=1, rot_period=0,
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
    mlen : int
        The mission length [seconds] (default : 48 * 60 * 60)
    nrow : int
        Number of detectors along row direction (default : 3)
    ncol : int
        Number of detectors along column direction (default : 3)
    fov : float
        The field of view in degrees (default : 5.0)
    ra0 : float, array-like
        Ra coord of centre region (default : [-10., 85.])
    dec0 : float, array-like
        Ra coord of centre region (default : [-57.5, 0.])
    el_min : float
        Minimum elevation range [deg] (default : 45)
    cut_el_min: bool
            If True, excludes timelines where el would be less than el_min
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

     # hardcoded mission length

    # Create LCDM realization
    ell, cls = get_cls()
    np.random.seed(25) # make sure all MPI ranks use the same seed
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    ac = ScanStrategy(mlen, # mission duration in sec.
                      sample_rate=12.01, # sample rate in Hz
                      location='atacama') # Instrument at south pole

    # Create a 3 x 3 square grid of Gaussian beams
    ac.create_focal_plane(nrow=nrow, ncol=ncol, fov=fov,
                          lmax=lmax, fwhm=fwhm)

    # calculate tods in two chunks
    ac.partition_mission(0.5*ac.mlen*ac.fsamp)

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
    ac.scan_instrument_mpi(alm, verbose=2, ra0=ra0,
                           dec0=dec0, az_throw=az_throw,
                           nside_spin=256,
                           el_min=el_min, cut_el_min=cut_el_min,
                           create_memmap=True)

    # Solve for the maps
    maps, cond = ac.solve_for_map(fill=np.nan)

    # Plotting
    if ac.mpi_rank == 0:
        print('plotting results')
        img_out_path = '../scratch/img/' 

        moll_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]')

        # plot rescanned maps
        plot_iqu(maps, img_out_path, 'rescan_atacama',
                 sym_limits=[250, 5, 5],
                 plot_func=hp.mollview, **moll_opts)

        # plot smoothed input maps
        nside = hp.get_nside(maps[0])
        hp.smoothalm(alm, fwhm=np.radians(fwhm/60.), verbose=False)
        maps_raw = hp.alm2map(alm, nside, verbose=False)

        plot_iqu(maps_raw, img_out_path, 'raw_atacama',
                 sym_limits=[250, 5, 5],
                 plot_func=hp.mollview, **moll_opts)

        # plot difference maps
        for arr in maps_raw:
            # replace stupid UNSEEN crap
            arr[arr==hp.UNSEEN] = np.nan

        diff = maps_raw - maps

        plot_iqu(diff, img_out_path, 'diff_atacama',
                 sym_limits=[1e-6, 1e-6, 1e-6],
                 plot_func=hp.mollview, **moll_opts)

        # plot condition number map
        moll_opts.pop('unit', None)

        plot_map(cond, img_out_path, 'cond_atacama',
                 min=2, max=5, unit='condition number',
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
        plt.savefig('../scratch/img/cls_atacama.png')
        plt.close()

        print("Results written to {}".format(os.path.abspath(img_out_path)))

def offset_beam(az_off=0, el_off=0, polang=0, lmax=100,
                fwhm=200, hwp_freq=25., pol_only=True):
    '''
    Script that scans LCDM realization of sky with a symmetric
    Gaussian beam that has been rotated away from the boresight.
    This means that the beam is highly asymmetric.

    Keyword arguments
    -----------------
    az_off : float,
        Azimuthal location of detector relative to boresight
        (default : 0.)
    el_off : float,
        Elevation location of detector relative to boresight
        (default : 0.)
    polang : float,
        Detector polarization angle in degrees (defined for
        unrotated detector as offset from meridian)
        (default : 0)
    lmax : int,
        Maximum multipole number, (default : 200)
    fwhm : float,
        The beam FWHM used in this analysis [arcmin]
        (default : 100)
    hwp_freq : float,
        HWP spin frequency (continuous mode) (default : 25.)
    pol_only : bool,
        Set unpolarized sky signal to zero (default : True)
    '''

    # Load up alm and blm
    ell, cls = get_cls()
    np.random.seed(30)
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    if pol_only:
        alm = (alm[0]*0., alm[1], alm[2])

    # init scan strategy and instrument
    mlen = 240 # mission length
    ss = ScanStrategy(mlen, # mission duration in sec.
                      sample_rate=1, # sample rate in Hz
                      location='spole') # South pole instrument

    # create single detector
    ss.create_focal_plane(nrow=1, ncol=1, fov=0, no_pairs=True,
                          polang=polang, lmax=lmax, fwhm=fwhm,
                          scatter=True)

    # move detector away from boresight
    try:
        ss.beams[0][0].az = az_off
        ss.beams[0][0].el = el_off
    except IndexError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e

    # Start instrument rotated (just to make things complicated)
    rot_period =  ss.mlen
    ss.set_instr_rot(period=rot_period, start_ang=45)

    # Set HWP rotation
    ss.set_hwp_mod(mode='stepped', freq=1/20., start_ang=45,
                   angles=[34, 12, 67])

    # calculate tod in one go (beam is symmetric so mmax=2 suffices)
    ss.partition_mission()
    ss.scan_instrument_mpi(alm, binning=False, nside_spin=512,
                           max_spin=2)

    # Store the tod and pixel indices made with symmetric beam
    try:
        tod_sym = ss.tod.copy()
        pix_sym = ss.pix.copy()
    except AttributeError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e
    
    # now repeat with asymmetric beam and no detector offset
    # set offsets to zero such that tods are generated using
    # only the boresight pointing.
    try:
        ss.beams[0][0].az = 0
        ss.beams[0][0].el = 0
        ss.beams[0][0].polang = 0

        # Convert beam spin modes to E and B modes and rotate them
        # create blm again, scan_instrument_mpi detetes blms when done
        ss.beams[0][0].gen_gaussian_blm()
        blm = ss.beams[0][0].blm
        blmI = blm[0].copy()
        blmE, blmB = tools.spin2eb(blm[1], blm[2])

        # Rotate blm to match centroid.
        # Note that rotate_alm uses the ZYZ euler convention.
        # Note that we include polang here as first rotation.
        q_off = ss.det_offset(az_off, el_off, polang)
        ra, dec, pa = ss.quat2radecpa(q_off)

        # convert between healpy and math angle conventions
        phi = np.radians(ra)
        theta = np.radians(90 - dec)
        psi = np.radians(-pa)

        # rotate blm
        hp.rotate_alm([blmI, blmE, blmB], psi, theta, phi, lmax=lmax, mmax=lmax)

        # convert beam coeff. back to spin representation.
        blmm2, blmp2 = tools.eb2spin(blmE, blmB)
        ss.beams[0][0].blm = (blmI, blmm2, blmp2)

    except IndexError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e

    ss.reset_instr_rot()
    ss.reset_hwp_mod()

    ss.scan_instrument_mpi(alm, binning=False, nside_spin=512,
                           max_spin=lmax) # now we use all spin modes

    # Figure comparing the raw detector timelines for the two versions
    # For subpixel offsets, the bottom plot shows you that sudden shifts
    # in the differenced tods are due to the pointing for the symmetric
    # case hitting a different pixel than the boresight pointing.
    if ss.mpi_rank == 0:
        plt.figure()
        gs = gridspec.GridSpec(5, 9)
        ax1 = plt.subplot(gs[:2, :6])
        ax2 = plt.subplot(gs[2:4, :6])
        ax3 = plt.subplot(gs[-1, :6])
        ax4 = plt.subplot(gs[:, 6:])

        samples = np.arange(tod_sym.size)
        ax1.plot(samples, ss.tod, label='Asymmetric Gaussian', linewidth=0.7)
        ax1.plot(samples, tod_sym, label='Symmetric Gaussian', linewidth=0.7,
                 alpha=0.5)
        ax1.legend()

        ax1.tick_params(labelbottom='off')
        sigdiff = ss.tod - tod_sym
        ax2.plot(samples, sigdiff,ls='None', marker='.', markersize=2.)
        ax2.tick_params(labelbottom='off')
        ax3.plot(samples, (pix_sym - ss.pix).astype(bool).astype(int),
                 ls='None', marker='.', markersize=2.)
        ax1.set_ylabel(r'Signal [$\mu K_{\mathrm{CMB}}$]')
        ax2.set_ylabel(r'asym-sym. [$\mu K_{\mathrm{CMB}}$]')
        ax3.set_xlabel('Sample number')
        ax3.set_ylabel('different pixel?')
        ax3.set_ylim([-0.25,1.25])
        ax3.set_yticks([0, 1])

        ax4.hist(sigdiff, 128, label='Difference')
        ax4.set_xlabel(r'Difference [$\mu K_{\mathrm{CMB}}$]')
        ax4.tick_params(labelleft='off')

        plt.savefig('../scratch/img/tods.png')
        plt.close()

def offset_beam_interp(az_off=0, el_off=0, polang=0, lmax=100,
                fwhm=200, hwp_freq=25., pol_only=True):
    '''
    Script that scans LCDM realization of sky with a symmetric
    Gaussian beam that has been rotated away from the boresight.
    This means that the beam is highly asymmetric. Scanning using
    interpolation.

    Keyword arguments
    -----------------
    az_off : float,
        Azimuthal location of detector relative to boresight
        (default : 0.)
    el_off : float,
        Elevation location of detector relative to boresight
        (default : 0.)
    polang : float,
        Detector polarization angle in degrees (defined for
        unrotated detector as offset from meridian)
        (default : 0)
    lmax : int,
        Maximum multipole number, (default : 200)
    fwhm : float,
        The beam FWHM used in this analysis [arcmin]
        (default : 100)
    hwp_freq : float,
        HWP spin frequency (continuous mode) (default : 25.)
    pol_only : bool,
        Set unpolarized sky signal to zero (default : True)
    '''

    # Load up alm and blm
    ell, cls = get_cls()
    np.random.seed(30)
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    if pol_only:
        alm = (alm[0]*0., alm[1], alm[2])

    # init scan strategy and instrument
    mlen = 240 # mission length
    ss = ScanStrategy(mlen, # mission duration in sec.
                      sample_rate=1, # sample rate in Hz
                      location='spole') # South pole instrument

    # create single detector
    ss.create_focal_plane(nrow=1, ncol=1, fov=0, no_pairs=True,
                          polang=polang, lmax=lmax, fwhm=fwhm,
                          scatter=True)

    # move detector away from boresight
    try:
        ss.beams[0][0].az = az_off
        ss.beams[0][0].el = el_off
    except IndexError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e

    # Start instrument rotated (just to make things complicated)
    rot_period =  ss.mlen
    ss.set_instr_rot(period=rot_period, start_ang=45)

    # Set HWP rotation
    ss.set_hwp_mod(mode='stepped', freq=1/20., start_ang=45,
                   angles=[34, 12, 67])

    # calculate tod in one go (beam is symmetric so mmax=2 suffices)
    ss.partition_mission()
    ss.scan_instrument_mpi(alm, binning=False, nside_spin=512,
                           max_spin=2, interp=True)

    # Store the tod made with symmetric beam
    try:
        tod_sym = ss.tod.copy()
    except AttributeError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e
    
    # now repeat with asymmetric beam and no detector offset
    # set offsets to zero such that tods are generated using
    # only the boresight pointing.
    try:
        ss.beams[0][0].az = 0
        ss.beams[0][0].el = 0
        ss.beams[0][0].polang = 0

        # Convert beam spin modes to E and B modes and rotate them
        # create blm again, scan_instrument_mpi detetes blms when done
        ss.beams[0][0].gen_gaussian_blm()
        blm = ss.beams[0][0].blm
        blmI = blm[0].copy()
        blmE, blmB = tools.spin2eb(blm[1], blm[2])

        # Rotate blm to match centroid.
        # Note that rotate_alm uses the ZYZ euler convention.
        # Note that we include polang here as first rotation.
        q_off = ss.det_offset(az_off, el_off, polang)
        ra, dec, pa = ss.quat2radecpa(q_off)

        # convert between healpy and math angle conventions
        phi = np.radians(ra)
        theta = np.radians(90 - dec)
        psi = np.radians(-pa)

        # rotate blm
        hp.rotate_alm([blmI, blmE, blmB], psi, theta, phi, lmax=lmax, mmax=lmax)

        # convert beam coeff. back to spin representation.
        blmm2, blmp2 = tools.eb2spin(blmE, blmB)
        ss.beams[0][0].blm = (blmI, blmm2, blmp2)

    except IndexError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e

    ss.reset_instr_rot()
    ss.reset_hwp_mod()
    
    # Now we use all spin modes.
    ss.scan_instrument_mpi(alm, binning=False, nside_spin=512,
                           max_spin=lmax, interp=True) 

    if ss.mpi_rank == 0:
        plt.figure()
        gs = gridspec.GridSpec(5, 9)
        ax1 = plt.subplot(gs[:2, :6])
        ax2 = plt.subplot(gs[2:4, :6])
        ax4 = plt.subplot(gs[:, 6:])

        samples = np.arange(tod_sym.size)
        ax1.plot(samples, ss.tod, label='Asymmetric Gaussian', linewidth=0.7)
        ax1.plot(samples, tod_sym, label='Symmetric Gaussian', linewidth=0.7,
                 alpha=0.5)
        ax1.legend()

        ax1.tick_params(labelbottom='off')
        sigdiff = ss.tod - tod_sym
        ax2.plot(samples, sigdiff,ls='None', marker='.', markersize=2.)
        ax2.tick_params(labelbottom='off')
        ax1.set_ylabel(r'Signal [$\mu K_{\mathrm{CMB}}$]')
        ax2.set_ylabel(r'asym-sym. [$\mu K_{\mathrm{CMB}}$]')
        ax2.set_xlabel('Sample number')

        ax4.hist(sigdiff, 128, label='Difference')
        ax4.set_xlabel(r'Difference [$\mu K_{\mathrm{CMB}}$]')
        ax4.tick_params(labelleft='off')

        plt.savefig('../scratch/img/tods_interp.png', dpi=250)
        plt.close()

        
def offset_beam_ghost(az_off=0, el_off=0, polang=0, lmax=100,
                      fwhm=200, hwp_freq=25., pol_only=True):
    '''
    Script that scans LCDM realization of sky with a detector
    on the boresight that has no main beam but a full-amplitude
    ghost at the specified offset eam. The signal is then binned
    using the boresight pointing and compared to a map made by
    a symmetric Gaussian beam that has been rotated away from
    the boresight. Results should agree.

    Keyword arguments
    ---------

    az_off : float,
        Azimuthal location of detector relative to boresight
        (default : 0.)
    el_off : float,
        Elevation location of detector relative to boresight
        (default : 0.)
    polang : float,
        Detector polarization angle in degrees (defined for
        unrotated detector as offset from meridian)
        (default : 0)
    lmax : int,
        Maximum multipole number, (default : 200)
    fwhm : float,
        The beam FWHM used in this analysis in arcmin
        (default : 100)
    hwp_freq : float,
        HWP spin frequency (continuous mode) (default : 25.)
    pol_only : bool,
        Set unpolarized sky signal to zero (default : True)
    '''

    # Load up alm and blm
    ell, cls = get_cls()
    np.random.seed(30)
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    if pol_only:
        alm = (alm[0]*0., alm[1], alm[2])

    # init scan strategy and instrument
    mlen = 240 # mission length
    ss = ScanStrategy(mlen, # mission duration in sec.
                      sample_rate=1, # sample rate in Hz
                      location='spole') # South pole instrument

    # create single detector on boresight
    ss.create_focal_plane(nrow=1, ncol=1, fov=0, no_pairs=True,
                          polang=polang, lmax=lmax, fwhm=fwhm,
                          scatter=True)

    try:
        beam = ss.beams[0][0]

        # set main beam to zero
        beam.amplitude = 0.

        # create Gaussian beam (would be done by code anyway otherwise)
        beam.gen_gaussian_blm()

        #explicitely set offset to zero
        beam.az = 0.
        beam.el = 0.
        beam.polang = 0.

        # create full-amplitude ghost
        beam.create_ghost(az=az_off, el=el_off, polang=polang,
                          amplitude=1.)
        ghost = beam.ghosts[0]
        ghost.gen_gaussian_blm()

    except IndexError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e


    # Start instrument rotated (just to make things complicated)
    rot_period =  ss.mlen
    ss.set_instr_rot(period=rot_period, start_ang=45)

    # Set HWP rotation
    ss.set_hwp_mod(mode='stepped', freq=1/20., start_ang=45,
                   angles=[34, 12, 67])

    # calculate tod in one go (beam is symmetric so mmax=2 suffices)
    ss.partition_mission()
    ss.scan_instrument_mpi(alm, binning=False, nside_spin=512,
                           max_spin=2, verbose=2)

    # Store the tod and pixel indices made with ghost
    try:
        tod_ghost = ss.tod.copy()
        pix_ghost = ss.pix.copy()
    except AttributeError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e

    # now repeat with asymmetric beam and no detector offset
    # set offsets to zero such that tods are generated using
    # only the boresight pointing.
    try:
        beam = ss.beams[0][0]
        beam.amplitude = 1.
        beam.gen_gaussian_blm()

        # Convert beam spin modes to E and B modes and rotate them
        blm = beam.blm
        blmI = blm[0].copy()
        blmE, blmB = tools.spin2eb(blm[1], blm[2])

        # Rotate blm to match centroid.
        # Note that rotate_alm uses the ZYZ euler convention.
        # Note that we include polang here as first rotation.
        q_off = ss.det_offset(az_off, el_off, polang)
        ra, dec, pa = ss.quat2radecpa(q_off)

        # convert between healpy and math angle conventions
        phi = np.radians(ra)
        theta = np.radians(90 - dec)
        psi = np.radians(-pa)

        # rotate blm
        hp.rotate_alm([blmI, blmE, blmB], psi, theta, phi, lmax=lmax, mmax=lmax)

        # convert beam coeff. back to spin representation.
        blmm2, blmp2 = tools.eb2spin(blmE, blmB)
        beam.blm = (blmI, blmm2, blmp2)

        # kill ghost
        ghost.dead = True
        # spinmaps will still be created, so make as painless as possible
        ghost.lmax = 1
        ghost.mmax = 0

    except IndexError as e:
        if ss.mpi_rank != 0:
            pass
        else:
            raise e


    # reset instr. rot and hwp modulation
    ss.reset_instr_rot()
    ss.reset_hwp_mod()

    ss.scan_instrument_mpi(alm, binning=False, nside_spin=512, verbose=2,
                           max_spin=lmax) # now we use all spin modes

    # Figure comparing the raw detector timelines for the two versions
    # For subpixel offsets, the bottom plot shows you that sudden shifts
    # in the differenced tods are due to the pointing for the symmetric
    # case hitting a different pixel than the boresight pointing.
    if ss.mpi_rank == 0:
        plt.figure()
        gs = gridspec.GridSpec(5, 9)
        ax1 = plt.subplot(gs[:2, :6])
        ax2 = plt.subplot(gs[2:4, :6])
        ax3 = plt.subplot(gs[-1, :6])
        ax4 = plt.subplot(gs[:, 6:])

        samples = np.arange(tod_ghost.size)
        ax1.plot(samples, ss.tod, label='Asymmetric Gaussian', linewidth=0.7)
        ax1.plot(samples, tod_ghost, label='Ghost', linewidth=0.7,
                 alpha=0.5)
        ax1.legend()

        ax1.tick_params(labelbottom='off')
        sigdiff = ss.tod - tod_ghost
        ax2.plot(samples, sigdiff, ls='None', marker='.', markersize=2.)
        ax2.tick_params(labelbottom='off')
        ax3.plot(samples, (pix_ghost - ss.pix).astype(bool).astype(int),
                 ls='None', marker='.', markersize=2.)
        ax1.set_ylabel(r'Signal [$\mu K_{\mathrm{CMB}}$]')
        ax2.set_ylabel(r'asym-sym. [$\mu K_{\mathrm{CMB}}$]')
        ax3.set_xlabel('Sample number')
        ax3.set_ylabel('different pixel?')
        ax3.set_ylim([-0.25,1.25])
        ax3.set_yticks([0, 1])

        ax4.hist(sigdiff, 128, label='Difference')
        ax4.set_xlabel(r'Difference [$\mu K_{\mathrm{CMB}}$]')
        ax4.tick_params(labelleft='off')

        plt.savefig('../scratch/img/tods_ghost.png')
        plt.close()

def test_ghosts(lmax=700, mmax=5, fwhm=43, ra0=-10, dec0=-57.5,
    az_throw=50, scan_speed=2.8, rot_period=4.5*60*60,
    hwp_mode=None):
    '''
    Similar test to `scan_bicep`, but includes reflected ghosts

    Simulates a 24h BICEP2-like scan strategy
    using a random LCDM realisation and a 3 x 3 grid
    of Gaussian beams pairs. Bins tods into maps and
    compares to smoothed input maps (no pair-
    differencing). MPI-enabled.

    Keyword arguments
    ---------

    lmax : int,
        bandlimit (default : 700)
    mmax : int,
        assumed azimuthal bandlimit beams (symmetric in this example
        so 2 would suffice) (default : 5)
    fwhm : float,
        The beam FWHM in arcmin (default : 40)
    ra0 : float,
        Ra coord of centre region (default : -10)
    dec0 : float,  (default : -57.5)
        Ra coord of centre region
    az_throw : float,
        Scan width in azimuth (in degrees) (default : 50)
    scan_speed : float,
        Scan speed in deg/s (default : 1)
    rot_period : float,
        The instrument rotation period in sec
        (default : 600)
    hwp_mode : str, None
        HWP modulation mode, either "continuous",
        "stepped" or None. Use freq of 1 or 1/10800 Hz
        respectively (default : None)
    '''

    mlen = 24 * 60 * 60 # hardcoded mission length

    # Create LCDM realization
    ell, cls = get_cls()
    np.random.seed(25) # make sure all MPI ranks use the same seed
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    b2 = ScanStrategy(mlen, # mission duration in sec.
                      sample_rate=12.01, # sample rate in Hz
                      location='spole') # Instrument at south pole

    # Create a 3 x 3 square grid of Gaussian beams
    b2.create_focal_plane(nrow=3, ncol=3, fov=5,
                          lmax=lmax, fwhm=fwhm)

    # Create reflected ghosts for every detector
    # We create two ghosts per detector. They overlap
    # but have different fwhm. First ghost is just a
    # scaled down version of the main beam, the second
    # has a much wider Gaussian shape.
    # After this initialization, the code takes
    # the ghosts into account without modifications
    b2.create_reflected_ghosts(b2.beams, amplitude=0.01,
                               ghost_tag='ghost_1', dead=False)
    b2.create_reflected_ghosts(b2.beams, amplitude=0.01,
                               fwhm=100, ghost_tag='ghost_2', dead=False)

    # calculate tods in two chunks
    b2.partition_mission(0.5*b2.nsamp)

    # Allocate and assign parameters for mapmaking
    b2.allocate_maps(nside=256)

    # set instrument rotation
    b2.set_instr_rot(period=rot_period, angles=[68, 113, 248, 293])

    # Set HWP rotation
    if hwp_mode == 'continuous':
        b2.set_hwp_mod(mode='continuous', freq=1.)
    elif hwp_mode == 'stepped':
        b2.set_hwp_mod(mode='stepped', freq=1/(3*60*60.))

    # Generate timestreams, bin them and store as attributes
    b2.scan_instrument_mpi(alm, verbose=1, ra0=ra0,
                           dec0=dec0, az_throw=az_throw,
                           nside_spin=256,
                           max_spin=mmax)

    # Solve for the maps
    maps, cond = b2.solve_for_map(fill=np.nan)

    # Plotting
    if b2.mpi_rank == 0:
        print('plotting results')

        cart_opts = dict(rot=[ra0, dec0, 0],
                lonra=[-min(0.5*az_throw, 90), min(0.5*az_throw, 90)],
                latra=[-min(0.375*az_throw, 45), min(0.375*az_throw, 45)],
                 unit=r'[$\mu K_{\mathrm{CMB}}$]')

        # plot rescanned maps
        plot_iqu(maps, '../scratch/img/', 'rescan_ghost',
                 sym_limits=[250, 5, 5],
                 plot_func=hp.cartview, **cart_opts)

        # plot smoothed input maps
        nside = hp.get_nside(maps[0])
        hp.smoothalm(alm, fwhm=np.radians(fwhm/60.), verbose=False)
        maps_raw = hp.alm2map(alm, nside, verbose=False)

        plot_iqu(maps_raw, '../scratch/img/', 'raw_ghost',
                 sym_limits=[250, 5, 5],
                 plot_func=hp.cartview, **cart_opts)

        # plot difference maps
        for arr in maps_raw:
            # replace stupid UNSEEN crap
            arr[arr==hp.UNSEEN] = np.nan

        diff = maps_raw - maps

        plot_iqu(diff, '../scratch/img/', 'diff_ghost',
                 sym_limits=[1e+1, 1e-1, 1e-1],
                 plot_func=hp.cartview, **cart_opts)

        # plot condition number map
        cart_opts.pop('unit', None)

        plot_map(cond, '../scratch/img/', 'cond_ghost',
                 min=2, max=5, unit='condition number',
                 plot_func=hp.cartview, **cart_opts)

        # plot input spectrum
        cls[3][cls[3]<=0.] *= -1.
        dell = ell * (ell + 1) / 2. / np.pi
        plt.figure()
        for i, label in enumerate(['TT', 'EE', 'BB', 'TE']):
          plt.semilogy(ell, dell * cls[i], label=label)

        plt.legend()
        plt.ylabel(r'$D_{\ell}$ [$\mu K^2_{\mathrm{CMB}}$]')
        plt.xlabel(r'Multipole [$\ell$]')
        plt.savefig('../scratch/img/cls_ghost.png')
        plt.close()

def single_detector(nsamp=1000, lmax=700, fwhm=30., ra0=-10, dec0=-57.5,
    az_throw=50, scan_speed=2.8, rot_period=4.5*60*60, mmax=5, nside_spin=512):
    '''
    Generates a timeline for a set of individual detectors scanning the sky. The
    spatial response of these detectors is described by a 1) Gaussian, 2) an
    elliptical Gaussian and, 3) a beam map derived by physical optics.

    Arguments
    ---------

    nsamp : int (default : 1000)
        The length of the generated timestreams in number of samples

    '''

    # Load up alm
    ell, cls = get_cls()
    np.random.seed(39)
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    # create Beam properties and pickle (this is just to test load_focal_plane)
    import tempfile
    import shutil
    import pickle
    opj = os.path.join

    blm_dir = os.path.abspath(opj(os.path.dirname(__file__),
                                       '../tests/test_data/example_blms'))
    po_file = opj(blm_dir, 'blm_hp_X1T1R1C8A_800_800.npy')
    eg_file = opj(blm_dir, 'blm_hp_eg_X1T1R1C8A_800_800.npy')

    tmp_dir = tempfile.mkdtemp()

    beam_file = opj(tmp_dir, 'beam_opts.pkl')
    beam_opts = dict(az=0,
                     el=0,
                     polang=0.,
                     btype='Gaussian',
                     name='X1T1R1C8',
                     fwhm=32.2,
                     lmax=800,
                     mmax=800,
                     amplitude=1.,
                     po_file=po_file,
                     eg_file=eg_file,
                     deconv_q=True,  # blm are SH coeff from hp.alm2map
                     normalize=True)


    with open(beam_file, 'wb') as handle:
        pickle.dump(beam_opts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # init scan strategy and instrument
    ss = ScanStrategy(nsamp/10., # mission duration in sec.
                      sample_rate=10, # 10 Hz sample rate
                      location='spole') # South pole instrument

    ss.load_focal_plane(tmp_dir, no_pairs=True)

    # remove tmp dir and contents
    shutil.rmtree(tmp_dir)

    # init plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Generate timestreams with Gaussian beams and plot them
    ss.scan_instrument_mpi(alm, verbose=1, ra0=ra0, dec0=dec0, az_throw=az_throw,
                           nside_spin=nside_spin, max_spin=mmax, binning=False)
    ax1.plot(ss.tod, label='sym. Gaussian', linewidth=0.7)

    gauss_tod = ss.tod.copy()

    # Generate timestreams with elliptical Gaussian beams and plot them
    ss.beams[0][0].btype = 'EG'
    ss.scan_instrument_mpi(alm, verbose=1, ra0=ra0, dec0=dec0, az_throw=az_throw,
                           nside_spin=nside_spin, max_spin=mmax, binning=False)
    ax1.plot(ss.tod, label='ell. Gaussian', linewidth=0.7)

    eg_tod = ss.tod.copy()

    # Generate timestreams with Physical Optics beams and plot them
    ss.beams[0][0].btype = 'PO'
    ss.scan_instrument_mpi(alm, verbose=1, ra0=ra0, dec0=dec0, az_throw=az_throw,
                           nside_spin=nside_spin, max_spin=mmax, binning=False)
    ax1.plot(ss.tod, label='PO', linewidth=0.7)

    po_tod = ss.tod.copy()

    ax2.plot(eg_tod - gauss_tod, label='EG - G', linewidth=0.7)
    ax2.plot(po_tod - gauss_tod, label='PO - G', linewidth=0.7)

    ax1.legend()
    ax2.legend()
    ax1.set_ylabel(r'Signal [$\mu K_{\mathrm{CMB}}$]')
    ax2.set_ylabel(r'Difference [$\mu K_{\mathrm{CMB}}$]')
    ax2.set_xlabel('Sample number')
    fig.savefig('../scratch/img/sdet_tod.png')

    plt.close()

def idea_jon():

    nside_spin = 512
    ra0 = 0
    dec0 = -90
    az_throw = 10
    max_spin = 5
    fwhm = 32.2
    scan_opts = dict(verbose=1, ra0=ra0, dec0=dec0, az_throw=az_throw,
                     nside_spin=nside_spin, max_spin=max_spin, binning=True)

    lmax = 800

    alm = tools.gauss_blm(1e-5, lmax, pol=False)
    ell = np.arange(lmax+1)
    fl = np.sqrt((2 * ell + 1) / 4. / np.pi)
    hp.almxfl(alm, fl, mmax=None, inplace=True)
    fm = (-1)**(hp.Alm.getlm(lmax)[1])
    alm *= fm
    alm = tools.get_copol_blm(alm)

    # create Beam properties and pickle (this is just to test load_focal_plane)
    import tempfile
    import shutil
    import pickle
    opj = os.path.join

    blm_dir = os.path.abspath(opj(os.path.dirname(__file__),
                                       '../tests/test_data/example_blms'))
    po_file = opj(blm_dir, 'blm_hp_X1T1R1C8A_800_800.npy')
    eg_file = opj(blm_dir, 'blm_hp_eg_X1T1R1C8A_800_800.npy')

    tmp_dir = tempfile.mkdtemp()

    beam_file = opj(tmp_dir, 'beam_opts.pkl')
    beam_opts = dict(az=0,
                     el=0,
                     polang=0.,
                     btype='Gaussian',
                     name='X1T1R1C8',
                     fwhm=fwhm,
                     lmax=800,
                     mmax=800,
                     amplitude=1.,
                     po_file=po_file,
                     eg_file=eg_file)


    with open(beam_file, 'wb') as handle:
        pickle.dump(beam_opts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # init scan strategy and instrument
    ss = ScanStrategy(1., # mission duration in sec.
                      sample_rate=10000,
                      location='spole')

    ss.allocate_maps(nside=1024)
    ss.load_focal_plane(tmp_dir, no_pairs=True)

    # remove tmp dir and contents
    shutil.rmtree(tmp_dir)

    ss.set_el_steps(0.01, steps=np.linspace(-10, 10, 100))

    # Generate maps with Gaussian beams
    ss.scan_instrument_mpi(alm, **scan_opts)
    ss.reset_el_steps()

    # Solve for the maps
    maps_g, cond_g = ss.solve_for_map(fill=np.nan)

    # Generate maps with elliptical Gaussian beams
    ss.allocate_maps(nside=1024)
    ss.beams[0][0].btype = 'EG'
    ss.scan_instrument_mpi(alm, **scan_opts)
    ss.reset_el_steps()

    # Solve for the maps
    maps_eg, cond_eg = ss.solve_for_map(fill=np.nan)

    # Generate map with Physical Optics beams and plot them
    ss.allocate_maps(nside=1024)
    ss.beams[0][0].btype = 'PO'
    ss.scan_instrument_mpi(alm, **scan_opts)
    ss.reset_el_steps()

    # Solve for the maps
    maps_po, cond_po = ss.solve_for_map(fill=np.nan)

    # Plotting
    print('plotting results')

    cart_opts = dict(#rot=[ra0, dec0, 0],
            lonra=[-min(0.5*az_throw, 10), min(0.5*az_throw, 10)],
            latra=[-min(0.375*az_throw, 10), min(0.375*az_throw, 10)],
             unit=r'[$\mu K_{\mathrm{CMB}}$]')

    # plot smoothed input maps
    nside = hp.get_nside(maps_g[0])
    hp.smoothalm(alm, fwhm=np.radians(fwhm/60.), verbose=False)
    maps_raw = hp.alm2map(alm, nside, verbose=False)

    plot_iqu(maps_raw, '../scratch/img/', 'raw_delta',
             sym_limits=[1, 1, 1],
             plot_func=hp.cartview, **cart_opts)

    # plot rescanned maps
    # plot_iqu(maps, '../scratch/img/', 'rescan_bicep',
    #         sym_limits=[250, 5, 5],
    #         plot_func=hp.cartview, **cart_opts)

    # plot difference maps
    # for arr in maps_raw:
    #     replace stupid UNSEEN crap
    #     arr[arr==hp.UNSEEN] = np.nan

    # diff = maps_raw - maps

    # plot_iqu(diff, '../scratch/img/', 'diff_bicep',
    #         sym_limits=[1e-6, 1e-6, 1e-6],
    #         plot_func=hp.cartview, **cart_opts)

    # plot condition number map
    # cart_opts.pop('unit', None)

    # plot_map(cond, '../scratch/img/', 'cond_bicep',
    #         min=2, max=5, unit='condition number',
    #         plot_func=hp.cartview, **cart_opts)

def azel4point(ra0=-10, dec0=-57.5, mlen=365, nsamp=1e4,
    lat=-22.96, lon=-67.79):
    '''
    Simple function to see what fraction of the day you can observe the
    Bicep patch (default value).

    Keyword arguments
    ---------

    ra0 : float
        Ra coord of point on the sky [deg] (default : -10)
    dec0 : float
        Ra coord of point on the sky [deg] (default : -57.5)
    mlen : int
        The mission length [days] (default : 365)
    nsamp : in
        The number of samples used in estimate
    lat : float
        Latitude for observation [deg] (default : -22.96, Atacama)
    lon : flaot
        Longitude for observation [deg] (default : -67.79, Atacama)

    '''

    import time
    import qpoint as qp
    Q = qp.QMap()
    nsec = mlen * 24 * 60 * 60
    dt = nsec / float(nsamp)
    ctime = time.time() + np.arange(0, nsec, dt, dtype=float)
    az, el, _ = Q.radec2azel(ra0, dec0, 0, lon, lat, ctime)

    print('Total observation time is {:4.1f} days'.\
        format((ctime[-1]-ctime[0])/24./3600.))

    observable_ratio = np.sum(el > 45)/float(len(ctime))
    print('This patch is observable {:4.1f}% of the time'.\
        format(100*observable_ratio))

    plt.plot(ctime, az)
    plt.ylabel('Azimuth [deg]')
    plt.savefig('../scratch/img/az.png')
    plt.close()

    plt.plot(ctime, el)
    plt.ylabel('Elevation [deg]')
    plt.savefig('../scratch/img/el.png')
    plt.close()

    plt.plot(az, el)
    plt.savefig('../scratch/img/azel.png')
    plt.close()

def test_satellite_scan(lmax=700, mmax=2, fwhm=43, 
    ra0=-10, dec0=-57.5,
    az_throw=50, scan_speed=2.8, 
    hwp_mode=None, alpha=45., beta=45.,
    alpha_period=5400., beta_period=600., delta_az=0., delta_el=0.,
    delta_psi=0., jitter_amp=1.0):

    '''
    Simulates a satellite scan strategy 
    using a random LCDM realisation and a 3 x 3 grid
    of Gaussian beams pairs. Bins tods into maps and
    compares to smoothed input maps (no pair-
    differencing). MPI-enabled.

    Keyword arguments
    ---------

    lmax : int,
        bandlimit (default : 700)
    mmax : int, 
        assumed azimuthal bandlimit beams (symmetric in this example
        so 2 would suffice) (default : 2)
    fwhm : float,
        The beam FWHM in arcmin (default : 40)
    ra0 : float,
        Ra coord of centre region (default : -10)
    dec0 : float,  (default : -57.5)
        Ra coord of centre region
    az_throw : float,
        Scan width in azimuth (in degrees) (default : 50)
    scan_speed : float,
        Scan speed in deg/s (default : 1)
    hwp_mode : str, None
        HWP modulation mode, either "continuous",
        "stepped" or None. Use freq of 1 or 1/10800 Hz
        respectively (default : None)
    '''

    print('Simulating a satellite...')
    mlen = 3 * 24 * 60 * 60 # hardcoded mission length

    # Create LCDM realization
    ell, cls = get_cls()
    np.random.seed(25) # make sure all MPI ranks use the same seed
    alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

    sat = ScanStrategy(mlen, # mission duration in sec.
        external_pointing=True, # Telling code to use non-standard scanning
        sample_rate=12.01, # sample rate in Hz
        location='space') # Instrument at south pole

    # Create a 3 x 3 square grid of Gaussian beams
    sat.create_focal_plane(nrow=7, ncol=7, fov=15,
                          lmax=lmax, fwhm=fwhm)

    # calculate tods in two chunks
    sat.partition_mission(0.5*sat.nsamp)

    # Allocate and assign parameters for mapmaking
    sat.allocate_maps(nside=256)

    scan_opts = dict(q_bore_func=sat.satellite_scan,        
        ctime_func=sat.satellite_ctime,
        q_bore_kwargs=dict(),
        ctime_kwargs=dict())

    # Generate timestreams, bin them and store as attributes
    sat.scan_instrument_mpi(alm, verbose=1, ra0=ra0,
        dec0=dec0, az_throw=az_throw, nside_spin=256, max_spin=mmax,
        **scan_opts)

    # Solve for the maps
    maps, cond, proj = sat.solve_for_map(fill=np.nan, return_proj=True)

    # Plotting
    if sat.mpi_rank == 0:
        print('plotting results')

        cart_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]')

        # plot rescanned maps


        plot_iqu(maps, '../scratch/img/', 'rescan_satellite',
                 sym_limits=[250, 5, 5],
                 plot_func=hp.mollview, **cart_opts)

        # plot smoothed input maps
        nside = hp.get_nside(maps[0])
        hp.smoothalm(alm, fwhm=np.radians(fwhm/60.), verbose=False)
        maps_raw = hp.alm2map(alm, nside, verbose=False)

        plot_iqu(maps_raw, '../scratch/img/', 'raw_satellite',
                 sym_limits=[250, 5, 5],
                 plot_func=hp.mollview, **cart_opts)

        # plot difference maps
        for arr in maps_raw:
            # replace stupid UNSEEN crap
            arr[arr==hp.UNSEEN] = np.nan

        diff = maps_raw - maps

        plot_iqu(diff, '../scratch/img/', 'diff_satellite',
                 sym_limits=[1e-6, 1e-6, 1e-6],
                 plot_func=hp.mollview, **cart_opts)

        # plot condition number map
        cart_opts.pop('unit', None)

        plot_map(cond, '../scratch/img/', 'cond_satellite',
                 min=2, max=5, unit='condition number',
                 plot_func=hp.mollview, **cart_opts)

        plot_map(proj[0], '../scratch/img/', 'hits_satellite',
                 unit='Hits', plot_func=hp.mollview, **cart_opts)

if __name__ == '__main__':
    #scan_bicep(mmax=2, hwp_mode='stepped', fwhm=28, lmax=1000)
    # scan_atacama(mmax=2, rot_period=60*60, mlen=48*60*60, nrow=3, ncol=3,
    #    fov=8.0, ra0=[-10], dec0=[-57.5], cut_el_min=False)
    # offset_beam(az_off=4, el_off=13, polang=36., pol_only=False)
    offset_beam_interp(az_off=40, el_off=13, polang=36., pol_only=True)    
    # offset_beam_ghost(az_off=4, el_off=13, polang=36., pol_only=True)
    # test_ghosts(mmax=2, hwp_mode='stepped', fwhm=28, lmax=1000)
    # single_detector()
    # idea_jon()
    # azel4point()

    # test_satellite_scan()

    # **************************
    # All tests should work with mpi, so e.g. you can use:
    # $ mpirun -n 10 python test.py
    # **************************
