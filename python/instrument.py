import numpy as np
import qpoint as qp
import healpy as hp
import tools
import os
import sys
import time


class Instrument(object):
    '''
    Initialize a (ground-based) telescope and specify its properties.
    '''

    def __init__(self, lat=None, lon=None, ghost_dc=0.):
        '''
        Set location of telescope on earth.

        Arguments
        ---------
        lon : float
            Longitude in degrees
        lat : float
            Latitude in degrees
        ghost_dc : float
            Ghost level.
        '''

        self.lat = lat
        self.lon = lon

    def set_focal_plane(self, nrow, fov):
        '''
        Create detector pointing offsets on the sky,
        i.e. in azimuth and elevation, for a square
        grid of detectors. Every point on the grid
        houses two detectors with orthogonal polarization
        angles. Also initializes polarization angles.

        Arguments
        ---------
        ndet : int
            Number of detectors per row.
        fov : float
            Angular size of side of square focal plane on
            sky in degrees.
        '''

        self.ndet = 2 * nrow**2
        self.chn_pr_az = np.zeros((nrow, nrow), dtype=float)
        self.chn_pr_el = np.zeros((nrow, nrow), dtype=float)

        x = np.linspace(-fov/2., fov/2., nrow)
        xx, yy = np.meshgrid(x, x)

        self.chn_pr_az = xx
        self.chn_pr_el = yy


    def get_blm(self, channel, lmax, fwhm=None, pol=True):
        '''
        Load or create healpix-formatted blm array(s) for specified
        channels.

        Arguments
        ---------
        channel
        lmax
        fwhm : float
            FWHM of symmetric gaussian beam in arcmin. If this 
            option is set, return blm array(s) with symmetric
            gaussian beam in appropriate slices in blm
            
        Returns
        -------
        blm (blm, blmm2) : (tuple of) array(s). 
            Healpix-formatted beam blm array. 
            Also returns blmm2 if pol is set.           
        '''

        # for now, just create a blm array with sym, gaussian beam
        if fwhm:
            return tools.gauss_blm(fwhm, lmax, pol=True)

    def get_blm_spider(self):
        pass

    def kill_channels(self):
        # function that kills certain detectors, i.e. create bool mask for focal plane
        pass

    def get_ghost(self):
        pass
    # function that introduces ghosts, i.e add detector offsets and corresponding beams



class ScanStrategy(Instrument, qp.QPoint):
    '''
    Given an instrument, create a scan strategy in terms of 
    azimuth, elevation, position and polarization angle.
    '''

    def __init__(self, bicep=False, act=False, **kwargs):

        # extract Instrument class specific kwargs.
        instr_kw = tools.extract_func_kwargs(
                   super(ScanStrategy, self).__init__, kwargs)
        
        # Initialize the instrument.
        if bicep:
            instr_kw['lat'] = -89.9
            instr_kw['lon'] = 169.15
            super(ScanStrategy, self).__init__(lat=-89.9, lon=169.15)
        elif act:
            instr_kw['lat'] = -22.96
            instr_kw['lon'] = -67.79
            super(ScanStrategy, self).__init__(lat=-22.96, lon=-67.79)
        else:
            super(ScanStrategy, self).__init__(**instr_kw)
        
            
        ctime_kw = tools.extract_func_kwargs(set_ctime, kwargs)
        set_ctime(**ctime_kw)        

        self.instr_rot = None
        self.hwp_mod = None
        self.rot_dict = {}
        self.hwp_dict = {}

    def set_ctime(self, ctime0=None):
        '''
        Set starting time.
        
        Arguments
        ---------
        ctime0 : int, optional
            Unix time in seconds. If None, use current time.
        '''
        if ctime0:
            self.ctime0 = ctime0
        else:
            self.ctime0 = time.time()

    def set_sample_rate(self, sample_rate=None):
        '''
        Set detector/pointing sample rate in Hz

        Arguments
        ---------
        sample_rate : float
            Sample rate in Hz
        '''
        
        self.fsamp = sample_rate

    def set_mission_len(self, length=None):
        '''
        Set total duration of mission.

        Arguments
        ---------
        length : float
            Mission length in seconds
        '''

        self.mlen = length

    def set_instr_rot(self, period, angles=None, sequence=None):
        '''
        Have the instrument periodically rotate around
        the boresight. 

        Arguments
        ---------
        period : float
            Rotation period in seconds.
        angles : array-like, optional
            Set of rotation angles. If not set, use 
            45 degree steps.
        sequence : array-like, optional
            Index array for angles array. If left None,
            cycle through angles.
        '''
        
        if self.hwp_mod:
            raise ValueError('Cannot have both instrument '
                             'and hpw modulation.')

        self.instr_rot = True
        self.rot_dict['angles'] = angles
        self.rot_dict['indices'] = sequence

    def set_hwp_mod(self, freq=0., period=None, start_ang=None,
                    angles=None, sequence=None, reflectivity=None):
        '''
        Modulate the polarized sky signal using a stepped or 
        continuously rotating half-wave plate.

        Arguments
        ---------
        freq : float, optional
            Use a continuously rotation HWP with this 
            frequency in Hz.
        period : float, optional
            Use a stepped HWP with this rotation period
            in sec.
        start_ang : float, optional
            Starting angle for the HWP in deg.
        angles : array-like, optional
            Rotation angles for stepped HWP. If not set,
            use 22.5 degree steps.
        sequence : array-like, optional
            Index array for angles array. If left None,
            cycle through angles.
        reflectivity : float, optional
            Not yet implemented.        
        '''

        if not freq and not period:
            raise ValueError('Pick either cont. rotation (freq) '
                             'or stepped (period)')

        if self.instr_rot:
            raise ValueError('Cannot have both instrument '
                             'and hpw modulation.')
        self.hwp_mod = True
        self.hwp_dict['freq'] = freq
        self.hwp_dict['period'] = period
        self.hwp_dict['angles'] = angles
        self.hwp_dict['start_ang'] = start_ang
        self.hwp_dict['indices'] = sequence
        self.hwp_dict['reflectivity'] = reflectivity

    def constant_el_scan(ra0, dec0, throw, el_steps,
                         scan):
        '''
        Let telescope scan back and forth in azimuth 
        while keeping elevation constant. Can do 
        periodic steps in elevation.
        '''

        # use qpoint to find az, el corresponding to ra0, el0

        # Set relative el and az starting values

        # Set scan az velocity profile (triangle wave, sine wave, etc)

        # return quaternion with ra, dec, pa

        pass

    def test(self):
        super(ScanStrategy, self).azel2bore()


#bicep = Instrument()
#bicep.set_location(1,2)
#print bicep.lat, bicep.lon
#bicep.set_focal_plane(3, 10)
