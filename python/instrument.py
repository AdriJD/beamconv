import numpy as np
import qpoint as qp
import healpy as hp
import os
import sys
import time


class Instrument(object):
    '''
    Initialize a (ground-based) telescope and specify its properties.
    '''

    def __init__(self, lat, lon):
        '''
        Set location of telescope on earth.

        Arguments
        ---------
        lon : float
            Longitude in degrees
        lat : float
            Latitude in degrees
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


    def get_blm(self):
        '''
        Load or create healpix-formatted blm array(s) for specified
        channels(s).
        '''
        pass

    def get_blm_spider(self):
        pass

    # function that kills certain detectors, i.e. create bool mask for focal plane
    # function that introduces ghosts, i.e add detector offsets and corresponding beams



class ScanStrategy(Instrument):
    '''
    Given an instrument, simulate a scan.
    '''

    def __init__(self):

        # Inherit methods from Instrument
        super(ScanStrategy, self).__init__()


        pass


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



    def set_sample_rate(self):
        pass

    def set_mission_len(self):
        pass

bicep = Instrument()
bicep.set_location(1,2)
print bicep.lat, bicep.lon
bicep.set_focal_plane(3, 10)
