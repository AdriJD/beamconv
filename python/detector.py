import sys
import time
import warnings
import glob
import numpy as np
import tools

class Beam():
    '''
    An object representing detector centroid and spatial information
    '''
    def __init__(self, az=0., el=0., polangle=0., name=None,
        pol='A', type='Gaussian', fwhm=43, dead=False, bdict=None):
        '''

        Arguments
        ---------

        az : float (default: 0.)
            Azimuthal location of detector relative to boresight
        el : float (default: 0.)
            Elevation location of detector relative to boresight
        polangle : float (default: 0.)
            The polarization orientation of the beam/detector
        name : str (default: None)
            The callsign of this particular beam
        pol : str (default: A)
            The polarization callsign of the beam (A or B)
        dead : bool (default: False)
            True if the beam is dead (not functioning)
        type : str (default: gaussian)
            Type of detector spatial response model. Can be one of three
            Gaussian : A symmetric Gaussian beam
            EG       : An elliptical Gaussian
            PO       : A realistic beam based on optical simulations or beam maps

        fwhm : float (default: 43. ) [arcmin]
            Detector beam FWHM

        '''

        if bdict is not None:

            self.az = bdict['az']
            self.el = bdict['el']
            self.polangle = bdict['polangle']
            self.name = bdict['name']
            self.pol = bdict['pol']
            self.type = bdict['type']
            self.fwhm = bdict['fwhm']
            self.dead = bdict['dead']

            self.set_beam_map(bdict)

        else:

            self.az = az
            self.el = el
            self.type = type
            self.fwhm = fwhm
            self.pol = pol

    def set_beam_map(cr, numel, bmap, load_map=False):

        self.cr = bdict['cr']
        self.numel = bdict['numel']
        self.bmap_path = bdict['bmap_path']
        if load_map:
            self.bmap = bmap

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
