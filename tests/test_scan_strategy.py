import unittest
import numpy as np
import healpy as hp
from beamconv import ScanStrategy
from beamconv import Beam
import os
import pickle

opj = os.path.join

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Create random alm.
        '''

        lmax = 10
        
        def rand_alm(lmax):
            alm = np.empty(hp.Alm.getsize(lmax), dtype=np.complex128)
            alm[:] = np.random.randn(hp.Alm.getsize(lmax))
            alm += 1j * np.random.randn(hp.Alm.getsize(lmax))
            return alm
        
        cls.alm = tuple([rand_alm(lmax) for i in range(3)])
        cls.lmax = lmax 

    def test_init(self):
        
        scs = ScanStrategy(duration=200, sample_rate=10)
        self.assertEqual(scs.mlen, 200)
        self.assertEqual(scs.fsamp, 10.)
        self.assertEqual(scs.nsamp, 2000)

        self.assertRaises(AttributeError, setattr, scs, 'fsamp', 1)
        self.assertRaises(AttributeError, setattr, scs, 'mlen', 1)
        self.assertRaises(AttributeError, setattr, scs, 'nsamp', 1)
    
    def test_el_steps(self):
        
        scs = ScanStrategy(duration=200)
        scs.set_el_steps(10, steps=np.arange(5)) 

        nsteps = int(np.ceil(scs.mlen / float(scs.step_dict['period'])))
        self.assertEqual(nsteps, 20)

        for step in range(12):
            el = next(scs.el_step_gen)
            scs.step_dict['step'] = el
            self.assertEqual(el, step%5)
            

        self.assertEqual(scs.step_dict['step'], el)
        scs.step_dict['remainder'] = 100
        scs.reset_el_steps()
        self.assertEqual(scs.step_dict['step'], 0)
        self.assertEqual(scs.step_dict['remainder'], 0)

        for step in range(nsteps):
            el = next(scs.el_step_gen)
            self.assertEqual(el, step%5)

        scs.reset_el_steps()
        self.assertEqual(next(scs.el_step_gen), 0)
        self.assertEqual(next(scs.el_step_gen), 1)

    def test_init_detpair(self):
        '''
        Check if spinmaps are correctly created.
        '''
        
        mmax = 3
        nside = 16
        scs = ScanStrategy()
        
        beam_a = Beam(fwhm=0., btype='Gaussian', mmax=mmax)
        beam_b = Beam(fwhm=0., btype='Gaussian', mmax=mmax)

        init_spinmaps_opts = dict(max_spin=5, nside_spin=nside,
                                  verbose=False)

        scs.init_detpair(self.alm, beam_a, beam_b=beam_b,
                         **init_spinmaps_opts)

        # We expect a spinmaps attribute (dict) with
        # main_beam key that contains a list of [func, func_c]
        # where func has shape (mmax + 1, 12nside**2) and
        # func_c has shape (2 mmax + 1, 12nside**2).
        # We expect an empty list for the ghosts.

        # Note empty lists evaluate to False
        self.assertFalse(scs.spinmaps['ghosts'])
        
        func, func_c = scs.spinmaps['main_beam']['maps']
        self.assertEqual(func.shape, (mmax + 1, 12 * nside ** 2))
        self.assertEqual(func_c.shape, (2 * mmax + 1, 12 * nside ** 2))

        # Since we have a infinitely narrow Gaussian the convolved
        # maps should just match the input (up to healpix quadrature
        # wonkyness).
        input_map = hp.alm2map(self.alm, nside, verbose=False) # I, Q, U
        zero_map = np.zeros_like(input_map[0])
        np.testing.assert_array_almost_equal(input_map[0],
                                             func[0], decimal=6)
        # s = 2 Pol map should be Q \pm i U
        np.testing.assert_array_almost_equal(input_map[1] + 1j * input_map[2],
                                             func_c[mmax + 2], decimal=6)

        # Test if rest of maps are zero.
        for i in range(1, mmax + 1):
            np.testing.assert_array_almost_equal(zero_map,
                                                 func[i], decimal=6)
            
        for i in range(1, 2 * mmax + 1):
            if i == mmax + 2:
                continue
            np.testing.assert_array_almost_equal(zero_map,
                                                 func_c[i], decimal=6)

    def test_init_detpair2(self):
        '''
        Check if function works with only A beam.
        '''
        
        mmax = 3
        nside = 16
        scs = ScanStrategy()
        
        beam_a = Beam(fwhm=0., btype='Gaussian', mmax=mmax)
        beam_b = None

        init_spinmaps_opts = dict(max_spin=5, nside_spin=nside,
                                  verbose=False)

        scs.init_detpair(self.alm, beam_a, beam_b=beam_b,
                         **init_spinmaps_opts)

        # Test for correct shapes.
        # Note empty lists evaluate to False
        self.assertFalse(scs.spinmaps['ghosts'])
        
        func, func_c = scs.spinmaps['main_beam']['maps']
        self.assertEqual(func.shape, (mmax + 1, 12 * nside ** 2))
        self.assertEqual(func_c.shape, (2 * mmax + 1, 12 * nside ** 2))

    def test_scan_spole(self):
        '''
        Perform a (low resolution) scan and see if TOD make sense.
        '''

        mlen = 10 * 60 
        rot_period = 120
        mmax = 2        
        ra0=-10
        dec0=-57.5
        fwhm = 200
        nside = 256
        az_throw = 10
        polang = 20.

        ces_opts = dict(ra0=ra0, dec0=dec0, az_throw=az_throw,
                        scan_speed=2.)
        
        scs = ScanStrategy(duration=mlen, sample_rate=10, location='spole')

        # Create a 1 x 1 square grid of Gaussian beams.
        scs.create_focal_plane(nrow=1, ncol=1, fov=4,
                               lmax=self.lmax, fwhm=fwhm,
                               polang=polang)
        beam = scs.beams[0][0]
        scs.init_detpair(self.alm, beam, nside_spin=nside,
                                   max_spin=mmax)
        scs.partition_mission()

        chunk = scs.chunks[0]
        ces_opts.update(chunk)

        # Populate boresight.
        scs.constant_el_scan(**ces_opts)

        # Test without returning anything (default behaviour).
        scs.scan(beam, **chunk)

        tod = scs.scan(beam, return_tod=True, **chunk)
        self.assertEqual(tod.size, chunk['end'] - chunk['start'])

        pix, nside_out, pa, hwp_ang = scs.scan(beam, return_point=True,
                                           **chunk)
        self.assertEqual(pix.size, tod.size)
        self.assertEqual(nside, nside_out)
        self.assertEqual(pa.size, tod.size)
        self.assertEqual(hwp_ang, 0)

        # Turn on HWP
        scs.set_hwp_mod(mode='continuous', freq=1., start_ang=0)
        scs.rotate_hwp(**chunk)
        tod2, pix2, nside_out2, pa2, hwp_ang2 = scs.scan(beam,
                        return_tod=True, return_point=True, **chunk)
        np.testing.assert_almost_equal(pix, pix2)
        np.testing.assert_almost_equal(pix, pix2)
        np.testing.assert_almost_equal(pa, pa2)
        self.assertTrue(np.any(np.not_equal(tod, tod2)), True)
        self.assertEqual(nside_out, nside_out2)
        self.assertEqual(hwp_ang2.size, tod.size)

        # Construct TOD manually.
        polang = beam.polang
        maps_sm = np.asarray(hp.alm2map(self.alm, nside, verbose=False,
                                        fwhm=np.radians(beam.fwhm / 60.)))

        np.testing.assert_almost_equal(maps_sm[0],
                                       scs.spinmaps['main_beam']['maps'][0][0])
        q = np.real(scs.spinmaps['main_beam']['maps'][1][mmax + 2])
        u = np.imag(scs.spinmaps['main_beam']['maps'][1][mmax + 2])
        np.testing.assert_almost_equal(maps_sm[1], q)
        np.testing.assert_almost_equal(maps_sm[2], u)

        tod_man = maps_sm[0][pix]
        tod_man += (maps_sm[1][pix] \
                    * np.cos(-2 * np.radians(pa + polang + 2 * hwp_ang2)))
        tod_man += (maps_sm[2][pix] \
                    * np.sin(-2 * np.radians(pa + polang + 2 * hwp_ang2)))

        np.testing.assert_almost_equal(tod2, tod_man)
        
    def test_scan_spole_bin(self):
        '''
        Perform a (low resolution) scan, bin and compare
        to input.
        '''

        mlen = 10 * 60 
        rot_period = 120
        mmax = 2        
        ra0=-10
        dec0=-57.5
        fwhm = 200
        nside = 256
        az_throw = 10

        scs = ScanStrategy(duration=mlen, sample_rate=10, location='spole')

        # Create a 1 x 2 square grid of Gaussian beams.
        scs.create_focal_plane(nrow=1, ncol=2, fov=4,
                              lmax=self.lmax, fwhm=fwhm)
        
        # Allocate and assign parameters for mapmaking.
        scs.allocate_maps(nside=nside)

        # set instrument rotation.
        scs.set_instr_rot(period=rot_period, angles=[68, 113, 248, 293])

        # Set elevation stepping.
        scs.set_el_steps(rot_period, steps=[0, 2, 4])
        
        # Set HWP rotation.
        scs.set_hwp_mod(mode='continuous', freq=3.)
        
        # Generate timestreams, bin them and store as attributes.
        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2.,
                                nside_spin=nside,
                                max_spin=mmax)

        # Solve for the maps
        maps, cond = scs.solve_for_map(fill=np.nan)

        hp.smoothalm(self.alm, fwhm=np.radians(fwhm/60.), 
                     verbose=False)
        maps_raw = np.asarray(hp.alm2map(self.alm, nside, verbose=False))

        cond[~np.isfinite(cond)] = 10

        np.testing.assert_array_almost_equal(maps_raw[0,cond<2.5],
                                             maps[0,cond<2.5], decimal=10)

        np.testing.assert_array_almost_equal(maps_raw[1,cond<2.5],
                                             maps[1,cond<2.5], decimal=10)

        np.testing.assert_array_almost_equal(maps_raw[2,cond<2.5],
                                             maps[2,cond<2.5], decimal=10)

    def test_interpolate(self):
        '''
        Compare interpoted TOD to raw for extremely bandlimited 
        input such that should agree relatively well.
        '''

        mlen = 60
        mmax = 2        
        ra0=-10
        dec0=-57.5
        fwhm = 1
        nside = 128
        az_throw = 10

        scs = ScanStrategy(duration=mlen, sample_rate=10, location='spole')

        # Create a 1 x 1 square grid of Gaussian beams.
        scs.create_focal_plane(nrow=1, ncol=1, fov=4,
                              lmax=self.lmax, fwhm=fwhm)
        
        # Generate timestreams, bin them and store as attributes.
        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2.,
                                nside_spin=nside,
                                max_spin=mmax,
                                binning=False)

        tod_raw = scs.tod.copy()

        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2.,
                                nside_spin=nside,
                                max_spin=mmax,
                                reuse_spinmaps=False,
                                inerpolate=True,
                                binning=False)

        np.testing.assert_array_almost_equal(tod_raw,
                                             scs.tod, decimal=10)
        
if __name__ == '__main__':
    unittest.main()

