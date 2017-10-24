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

    def __init__(self, lat=None, lon=None, location='spole',
                 ghost_dc=0.):
        '''
        Set location of telescope on earth.

        Arguments
        ---------
        lon : float
            Longitude in degrees
        lat : float
            Latitude in degrees
        s_pole : bool, optional
            init telescope at predefinied south pole location.
            (lat=-89.9, lon=169.15)
        atacama : bool, optional
            init telescope at predefined atacama desert location.
            (lat=-22.96, lon=-67.79)
        ghost_dc : float
            Ghost level. Not implemented yet.
        '''

        self.lat = lat
        self.lon = lon

        if location == 'spole':
            self.lat = -89.9
            self.lon = 169.15

        elif location == 'atacama':
            self.lat = -22.96
            self.lon = -67.79

        if not self.lat or not self.lon:
            raise ValueError('Specify location of telescope')

    def set_focal_plane(self, nrow, fov):
        '''
        Create detector pointing offsets on the sky,
        i.e. in azimuth and elevation, for a square
        grid of detectors. Every point on the grid
        houses two detectors with orthogonal polarization
        angles.

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

        self.chn_pr_az = xx.flatten()
        self.chn_pr_el = yy.flatten()


    def get_blm(self, lmax, channel=None, fwhm=None, pol=True):
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



#class ScanStrategy(Instrument, qp.QPoint):
#class ScanStrategy(qp.QPoint, Instrument): # works
class ScanStrategy(qp.QMap, Instrument):
    '''
    Given an instrument, create a scan strategy in terms of
    azimuth, elevation, position and polarization angle.
    '''

    def __init__(self, duration, sample_rate, **kwargs):
        '''
        Initialize scan parameters

        Arguments
        ---------
        duration : float
            Mission duration in seconds.
        sample_rate : float
             Sample rate in Hz.
        '''

        # extract Instrument class specific kwargs.
#        instr_kw = tools.extract_func_kwargs(
#                   super(ScanStrategy, self).__init__, kwargs)
        instr_kw = tools.extract_func_kwargs(
                   Instrument.__init__, kwargs)
#        instr_kw = tools.extract_func_kwargs(
#                   Instrument.__init__(self), kwargs)

        # Initialize the instrument and qpoint.
#        super(ScanStrategy, self).__init__(**instr_kw)
        Instrument.__init__(self, **instr_kw)
        qp.QPoint.__init__(self, fast_math=True)

        ctime_kw = tools.extract_func_kwargs(self.set_ctime, kwargs)
        self.set_ctime(**ctime_kw)

        self.set_mission_len(duration)
        self.set_sample_rate(sample_rate)

        self.instr_rot = None
        self.hwp_mod = None
        self.rot_dict = {}
        self.hwp_dict = {}

    def __del__(self):
        '''
        Call QPoint destructor explicitely to make sure
        the c code frees up memory before exiting.
        '''
#        super(ScanStrategy, self).__del__()
        self.__del__

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

        self.fsamp = float(sample_rate)

    def set_mission_len(self, duration=None):
        '''
        Set total duration of mission.

        Arguments
        ---------
        duration : float
            Mission length in seconds
        '''

        self.mlen = duration

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
            self.hwp_mod = False

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
            self.instr_rot = False

        self.hwp_mod = True
        self.hwp_dict['freq'] = freq
        self.hwp_dict['period'] = period
        self.hwp_dict['angles'] = angles
        self.hwp_dict['start_ang'] = start_ang
        self.hwp_dict['indices'] = sequence
        self.hwp_dict['reflectivity'] = reflectivity

    def constant_el_scan(self, ra0, dec0, az_throw, scan_speed,
                         start, end, el_off=None, vel_prf='triangle'):
        '''
        Let boresight scan back and forth in azimuth starting
        from point in ra, dec, while keeping elevation constant.

        Arguments
        ---------
        ra0 : float
            Ra coordinate of centre of scan in degrees
        dec0 : float
            Ra coordinate of centre of scan in degrees
        az_throw : float
            Scan width in azimuth (in degrees)
        scan_speed : float
            Max scan speed in degrees per second
        start : int
            Starting sample
        end : int
            End at this sample
        el_off : float, optional
            Offset in elevation (in degrees)
        vel_prf : str
            Velocity profile: "triangle" for triangle wave
        '''

        chunk_len = end - start
        delta_ct = np.arange(chunk_len)
        ctime = start + delta_ct

        # use qpoint to find az, el corresponding to ra0, el0
#        az0, el0, _ = super(ScanStrategy, self).radec2azel(ra0, dec0, 0,
#                        self.lon, self.lat, ctime[0])
#        az0, el0, _ = super(ScanStrategy, self).radec2azel(ra0, dec0, 0,
#                                            self.lon, self.lat, ctime[0])
#        az0, el0, _ = qp.QPoint.radec2azel(self, ra0, dec0, 0,
#                                            self.lon, self.lat, ctime[0])

        az0, el0, _ = self.radec2azel(ra0, dec0, 0,
                                       self.lon, self.lat, ctime[0])

        # Scan
        if vel_prf is 'triangle':
            scan_period = 2 * az_throw / float(scan_speed) # in deg.
            az = np.arcsin(np.sin(2 * np.pi * delta_ct / scan_period / self.fsamp))
            az *= az_throw / (np.pi)
            az += az0

        # return quaternion with ra, dec, pa
        if el_off:
            el = el0 + el_off * np.ones_like(az)
        else:
            el = el0 * np.ones_like(az)

        self.q_bore = self.azel2bore(az, el, None, None, self.lon, self.lat, ctime)

    def get_spinmaps(self, alm, blm, max_spin, nside):
        '''
        Compute convolution of map with different spin modes
        of the beam. Computed per spin, so creates spinmmap
        for every s<= 0 for T and for every s for pol.

        Arguments
        ---------
        alm : tuple of array-like
            Tuple containing alm, almE and almB
        blm : tuple of array-like
            Tuple containing blm, blmp2 and blmm2
        max_spin : int
            Maximum spin value describing the beam
        '''

        self.N = max_spin + 1
        lmax = hp.Alm.getlmax(alm[0].size)

        # Unpolarized sky and beam first
        self.func = np.zeros((self.N, 12*nside**2), dtype=np.complex128) # s <=0 spheres

        start = 0
        for n in xrange(self.N): # note n is s
            end = lmax + 1 - n
            if n == 0: # scalar transform

                flmn = hp.almxfl(alm[0], blm[0][start:start+end], inplace=False)
                self.func[n,:] += hp.alm2map(flmn, nside)

            else: # spin transforms

                bell = np.zeros(lmax+1, dtype=np.complex128)
                # spin n beam
                bell[n:] = blm[0][start:start+end]

                flmn = hp.almxfl(alm[0], bell, inplace=False)
                flmmn = hp.almxfl(alm[0], np.conj(bell), inplace=False)

                flmnp = - (flmn + flmmn) / 2.
                flmnm = 1j * (flmn - flmmn) / 2.
                spinmaps = hp.alm2map_spin([flmnp, flmnm], nside, n, lmax, lmax)
                self.func[n,:] = spinmaps[0] + 1j * spinmaps[1]

            start += end

        # Pol
        self.func_c = np.zeros((2*self.N-1, 12*nside**2), dtype=np.complex128) # all spin spheres

        almp2 = -1 * (alm[1] + 1j * alm[2])
        almm2 = -1 * (alm[1] - 1j * alm[2])

        blmm2 = blm[1]
        blmp2 = blm[2]

        start = 0
        for n in xrange(self.N):
            end = lmax + 1 - n

            bellp2 = np.zeros(lmax+1, dtype=np.complex128)
            bellm2 = bellp2.copy()

            bellp2[np.abs(n):] = blmp2[start:start+end]
            bellm2[np.abs(n):] = blmm2[start:start+end]

            ps_flm_p = hp.almxfl(almp2, bellm2, inplace=False) + \
                hp.almxfl(almm2, np.conj(bellm2), inplace=False)
            ps_flm_p /= -2.

            ps_flm_m = hp.almxfl(almp2, bellm2, inplace=False) - \
                hp.almxfl(almm2, np.conj(bellm2), inplace=False)
            ps_flm_m *= 1j / 2.

            ms_flm_p = hp.almxfl(almm2, bellp2, inplace=False) + \
                hp.almxfl(almp2, np.conj(bellp2), inplace=False)
            ms_flm_p /= -2.

            ms_flm_m = hp.almxfl(almm2, bellp2, inplace=False) - \
                hp.almxfl(almp2, np.conj(bellp2), inplace=False)
            ms_flm_m *= 1j / 2.

            if n == 0:
                spinmaps = [hp.alm2map(-ps_flm_p, nside),
                            hp.alm2map(-ms_flm_m, nside)]

                self.func_c[self.N-n-1,:] = spinmaps[0] - 1j * spinmaps[1]

            else:
                # positive spin
                spinmaps = hp.alm2map_spin([ps_flm_p, ps_flm_m],
                                           nside, n, lmax, lmax)
                self.func_c[self.N+n-1,:] = spinmaps[0] + 1j * spinmaps[1]

                # negative spin
                spinmaps = hp.alm2map_spin([ms_flm_p, ms_flm_m],
                                           nside, n, lmax, lmax)
                self.func_c[self.N-n-1,:] = spinmaps[0] - 1j * spinmaps[1]

            start += end

    def scan(self, az_off, el_off, nside, start, end):
        '''
        Combine the pointing and spinmaps into a tod.
        '''

        q_off = self.det_offset(az_off, el_off, 0)

        chunk_len = end - start
        delta_ct = np.arange(chunk_len)
        ctime = start + delta_ct

        sim_tod = np.zeros(ctime.size, dtype='float64')
        sim_tod2 = np.zeros(ctime.size, dtype=np.complex128)

        ra, dec, pa = self.bore2radec(q_off, ctime, self.q_bore,
                                      q_hwp=None, sindec=False,
                                      return_pa=True)
        pix = tools.radec2ind_hp(ra, dec, nside)

        for nidx, n in enumerate(xrange(-self.N+1, self.N)):

            exppais = np.exp(1j * n * np.radians(pa))
            sim_tod2 += self.func_c[nidx][pix] * exppais

            if n == 0: #avoid expais since its one anyway
                sim_tod += np.real(self.func[n][pix])

            if n > 0:
                sim_tod += 2 * np.real(self.func[n,:][pix]) * np.cos(n * np.radians(pa))
                sim_tod -= 2 * np.imag(self.func[n,:][pix]) * np.sin(n * np.radians(pa))

        # load up hwp and polang arrays
        # combine polarized and unpolarized tods
#        expm2 = np.exp(1j * (4 * np.radians(hwpang) + 2 * np.radians(polang)))
#        sim_tod += np.real(sim_tod2 * expm2 + np.conj(sim_tod2 * expm2)) / 2.


    def tod2map(self):
        # note that we need QMap not QPoint for this.
        pass

#b2 = Instrument(lon=10, lat=10)
b2 = ScanStrategy(1*356*24*3600, 100, atacama=False, s_pole=True)
b2.constant_el_scan(0, 0, 50, 10, 0, 60*100, el_off=0)
#print b2.__mro__

