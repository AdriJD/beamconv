import sys
import time
import warnings
import glob
import numpy as np
import tools

class Beam(object):
    '''
    An object representing detector centroid and spatial information
    '''
    def __init__(self, az=0., el=0., polang=0., name=None,
        pol='A', btype='Gaussian', fwhm=43, lmax=None, dead=False, bdict=None,
        load_map=False):
        '''

        Arguments
        ---------

        az : float (default: 0.)
            Azimuthal location of detector relative to boresight
        el : float (default: 0.)
            Elevation location of detector relative to boresight
        polang : float (default: 0.)
            The polarization orientation of the beam/detector [deg]
        name : str (default: None)
            The callsign of this particular beam
        pol : str (default: A)
            The polarization callsign of the beam (A or B)
        dead : bool (default: False)
            True if the beam is dead (not functioning)
        btype : str (default: gaussian)
            Type of detector spatial response model. Can be one of three
            Gaussian : A symmetric Gaussian beam, definied by centroids and FWHM
            Gaussian_map : A symmetric Gaussian, defined by centroids and a map
            EG       : An elliptical Gaussian
            PO       : A realistic beam based on optical simulations or beam maps
        fwhm : float (default: 43. ) [arcmin]
            Detector beam FWHM
        bdict : dict
            Dictionary with kwargs. Will overwrite all other provided kwargs
        load_map : bool, optional
            Not yet implemented

        '''

        if bdict:
            # Loading from a dictionary

            self.az = bdict['az']
            self.el = bdict['el']
            self.polang = bdict['polang']
            self.name = bdict['name']
            self.pol = bdict['pol']
            self.btype = bdict['type']
            self.fwhm = bdict['fwhm']
            self.dead = bdict['dead']

            self.cr = bdict['cr']
            self.numel = bdict['numel']
            self.bmap_path = bdict['bmap_path']
#            if load_map:
#                self.bmap = bmap # FIX this

            self.blm = bdict['blm']
            self.blmm2 = bdict['blmm2']
            self.blmm2 = bdict['blmp2']

        else:
            # Populating attributes from init call

            self.az = az
            self.el = el
            self.polang = polang
            self.name = name
            self.pol = pol
            self.btype = btype
            self.fwhm = fwhm
            self.dead = dead

            if lmax is None:
                # Going up to the Nyquist frequency set by beam scale
                self.lmax = int(2*np.pi/np.radians(self.fwhm/60.))
            else:
                self.lmax = lmax

            # you don't want this to happen at init
            blm, blmm2 = tools.gauss_blm(self.fwhm, self.lmax, pol=True)

            self.blm = blm
            self.blmm2 = blmm2


    def load_eg_beams(bdir):
        '''
        Loads a collection of elliptical Gaussian beams from parameters stored
        in a pickle file.
        '''

        pass


    def generate_eg_beams(nrow=1, ncol=1, fov=10, fwhm0=43,
        emin=0.01, emax=0.05):
        '''
        Creates a set of elliptical Gaussian beams based on the recipe provided
        in arguments

        Arguments
        ---------

        '''

        pass


class Detector():
    '''
    An object representing a CMB bolometer. Attribute describe detector
    sensitivity, frequency, and spatial response.


    '''

    def __init__(self, pol=True, single_moded=True, nu=100, bw=0.3, oe=0.4,
        fwhm=43, alt=5.2, et=None, NEP_phonon=6, NEP_readout=3, NEP_photon=None,
        P_opt=0, P_atm=None, P_cmb =0.3,  site_dir='old_profiles/', site=None):
        '''

        Arguments
        ----------

        pol : Bool (default: True)
            Is the detector polarized
        single_moded : Bool (default: True)
            Is the detector single moded. Currently, the code only knows
            how to deal with single moded detectors
        nu : float (default: 100) [GHz]
            The detector center frequency
        be : float (default: 0.3)
            Fractional spectral bandwidth
        oe : float (default: 0.4)
            Optical efficiency

        '''

        self.pol = pol
        self.polfact = 1.0 if self.pol else 2.0
        self.nu = nu
        self.bw = bw
        self.oe = oe
        self.fwhm = fwhm

        # Optical properties
        self.bsa = 2*pi*(pi/180/60*(self.fwhm/2.354))**2
        self.effa = (c/(self.nu*1e9))**2/self.bsa*1e4 # Effective area in cm^2

        # Note that you don't need fwhm
        if single_moded:
            self.et = (c/(1e9*self.nu))**2
        else:
            self.et = self.bsa*self.effa/1e4

        # Photon loading of various sorts
        self.P_opt = P_opt                        # pW
        self.P_cmb = P_cmb                        # pW
        self.site_dir = site_dir
        self.site = site
        if site is None and P_atm is None:
            raise ValueError('Must define site or atmospheric loading')

        if site is not None:
            self.P_atm = site_loading(self, site, site_dir=self.site_dir) # pW

        self.P_photon = self.oe * (self.P_atm + self.P_cmb + self.P_opt)
        self.dqdt = self.oe*dqdt(self.nu, bw=self.bw, pol=self.pol)

        # NEP's
        if NEP_photon is None:
            # aW/rHz
            self.NEP_bose = np.sqrt(2*self.P_photon**2 \
                /(self.polfact*self.nu*self.bw*1e9)) * 1e6
            self.NEP_shot = np.sqrt(2*hg*self.nu*self.P_photon) * 1e6
            self.NEP_photon = np.sqrt(self.NEP_bose**2 + self.NEP_shot**2)
        else:
            self.NEP_photon = NEP_photon        # aW/rHz

        self.NEP_phonon = NEP_phonon            # aW/rHz
        self.NEP_readout = NEP_readout          # aW/rHz
        self.NEP_detector = np.sqrt(NEP_phonon**2 + NEP_readout**2)
        self.NEP_total = np.sqrt(self.NEP_phonon**2 + self.NEP_readout**2
            +self.NEP_photon**2)                # aW/rHz

        self.NET = self.NEP_total/(np.sqrt(2)*self.dqdt)
