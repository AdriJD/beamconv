import unittest
import numpy as np
import healpy as hp
from beamconv import ScanStrategy
from beamconv import Beam, tools
import os
import pickle

opj = os.path.join

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Create random alm.
        '''

        lmax = 50
        
        def rand_alm(lmax):
            alm = np.empty(hp.Alm.getsize(lmax), dtype=np.complex128)
            alm[:] = np.random.randn(hp.Alm.getsize(lmax))
            alm += 1j * np.random.randn(hp.Alm.getsize(lmax))
            # Make m=0 modes real.
            alm[:lmax+1] = np.real(alm[:lmax+1])            
            return alm
        
        cls.alm = tuple([rand_alm(lmax) for i in range(3)])
        cls.lmax = lmax 

    def test_init(self):
        
        scs = ScanStrategy(duration=200, sample_rate=10)
        self.assertEqual(scs.mlen, 200)
        self.assertEqual(scs.fsamp, 10.)
        self.assertEqual(scs.nsamp, 2000)

        # Test if we are unable to change scan parameters after init.
        self.assertRaises(AttributeError, setattr, scs, 'fsamp', 1)
        self.assertRaises(AttributeError, setattr, scs, 'mlen', 1)
        self.assertRaises(AttributeError, setattr, scs, 'nsamp', 1)

    def test_init_no_mlen(self):
        
        # Test if we can also init without specifying mlen.
        scs = ScanStrategy(sample_rate=20, num_samples=100)

        #  nsamp = mlen * sample_rate
        self.assertEqual(scs.mlen, 5)
        self.assertEqual(scs.fsamp, 20)
        self.assertEqual(scs.nsamp, 100)

    def test_init_no_sample_rate(self):
        
        # Test if we can also init without specifying mlen.
        scs = ScanStrategy(duration=5, num_samples=100)

        #  nsamp = mlen * sample_rate
        self.assertEqual(scs.mlen, 5)
        self.assertEqual(scs.fsamp, 20)
        self.assertEqual(scs.nsamp, 100)

    def test_init_zero_duration(self):
        
        # Sample rate should be zero
        scs = ScanStrategy(duration=0, sample_rate=10)

        #  nsamp = mlen * sample_rate
        self.assertEqual(scs.mlen, 0)
        self.assertEqual(scs.fsamp, 0)
        self.assertEqual(scs.nsamp, 0)

    def test_init_err(self):

        # Test if init raises erorrs when user does not 
        # provide enough info.
        with self.assertRaises(ValueError):
            ScanStrategy(duration=5)
        with self.assertRaises(ValueError):
            ScanStrategy(num_samples=5)
        with self.assertRaises(ValueError):
            ScanStrategy(sample_rate=5)

        # Or if nsamp = mlen * sample_rate is not satisfied.
        with self.assertRaises(ValueError):
            ScanStrategy(duration=10, sample_rate=20, num_samples=100)

        # Or when sample_rate is zero or negative.
        with self.assertRaises(ValueError):
            ScanStrategy(sample_rate=0, duration=10)

        with self.assertRaises(ValueError):
            ScanStrategy(sample_rate=-2, duration=10)
        
    def test_el_steps(self):
        
        scs = ScanStrategy(duration=200, sample_rate=30)
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
        scs = ScanStrategy(duration=1, sample_rate=10)
        
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
        
        func = scs.spinmaps['main_beam']['s0a0']['maps']
        func_c = scs.spinmaps['main_beam']['s2a4']['maps']        
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
        scs = ScanStrategy(duration=1, sample_rate=10)
        
        beam_a = Beam(fwhm=0., btype='Gaussian', mmax=mmax)
        beam_b = None

        init_spinmaps_opts = dict(max_spin=5, nside_spin=nside,
                                  verbose=False)

        scs.init_detpair(self.alm, beam_a, beam_b=beam_b,
                         **init_spinmaps_opts)

        # Test for correct shapes.
        # Note empty lists evaluate to False
        self.assertFalse(scs.spinmaps['ghosts'])
        
        func = scs.spinmaps['main_beam']['s0a0']['maps']
        func_c = scs.spinmaps['main_beam']['s2a4']['maps']        
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
        nside = 128
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
                                       scs.spinmaps['main_beam']['s0a0']['maps'][0])
        q = np.real(scs.spinmaps['main_beam']['s2a4']['maps'][mmax + 2])
        u = np.imag(scs.spinmaps['main_beam']['s2a4']['maps'][mmax + 2])
        np.testing.assert_almost_equal(maps_sm[1], q)
        np.testing.assert_almost_equal(maps_sm[2], u)

        tod_man = maps_sm[0][pix]
        tod_man += (maps_sm[1][pix] \
                    * np.cos(2 * np.radians(pa - polang - 2 * hwp_ang2)))
        tod_man += (maps_sm[2][pix] \
                    * np.sin(2 * np.radians(pa - polang - 2 * hwp_ang2)))
        
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
        nside = 128
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

        # Solve for the maps.
        maps, cond = scs.solve_for_map(fill=np.nan)

        alm = hp.smoothalm(self.alm, fwhm=np.radians(fwhm/60.), 
                     verbose=False)
        maps_raw = np.asarray(hp.alm2map(self.alm, nside, verbose=False))

        cond[~np.isfinite(cond)] = 10

        np.testing.assert_array_almost_equal(maps_raw[0,cond<2.5],
                                             maps[0,cond<2.5], decimal=10)

        np.testing.assert_array_almost_equal(maps_raw[1,cond<2.5],
                                             maps[1,cond<2.5], decimal=10)

        np.testing.assert_array_almost_equal(maps_raw[2,cond<2.5],
                                             maps[2,cond<2.5], decimal=10)

    def test_scan_ghosts(self):
        '''
        Perform a (low resolution) scan with two detectors, 
        compare to detector + ghost.
        '''

        mlen = 10 * 60 
        rot_period = 120
        mmax = 2        
        ra0=-10
        dec0=-57.5
        fwhm = 200
        nside = 128
        az_throw = 10

        scs = ScanStrategy(duration=mlen, sample_rate=10, location='spole')

        # Create two Gaussian (main) beams.
        beam_opts = dict(az=0, el=0, polang=0, fwhm=fwhm, lmax=self.lmax, 
                         symmetric=True)
        ghost_opts = dict(az=-4, el=10, polang=34, fwhm=fwhm, lmax=self.lmax,
                          symmetric=True, amplitude=0.1)

        scs.add_to_focal_plane(Beam(**beam_opts))
        scs.add_to_focal_plane(Beam(**ghost_opts))
        
        # Allocate and assign parameters for mapmaking.
        scs.allocate_maps(nside=nside)

        # Set HWP rotation.
        scs.set_hwp_mod(mode='continuous', freq=3.)
        
        # Generate timestreams, bin them and store as attributes.
        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2., binning=False,
                                nside_spin=nside,
                                max_spin=mmax, save_tod=True)

        tod = scs.data(scs.chunks[0], beam=scs.beams[0][0], data_type='tod')
        tod += scs.data(scs.chunks[0], beam=scs.beams[1][0], data_type='tod')
        tod = tod.copy()

        # Repeat with single beam + ghost.
        scs.remove_from_focal_plane(scs.beams[1][0])
        scs.beams[0][0].create_ghost(**ghost_opts)

        scs.reset_hwp_mod()

        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2., binning=False,
                                nside_spin=nside,
                                max_spin=mmax, save_tod=True)

        tod_w_ghost = scs.data(scs.chunks[0], beam=scs.beams[0][0], 
                               data_type='tod')

        # Sum TOD of two beams must match TOD of single beam + ghost.
        np.testing.assert_array_almost_equal(tod, tod_w_ghost, decimal=10)

    def test_scan_ghosts_map(self):
        '''
        Perform a (low resolution) scan with two detectors, 
        compare map to detector + ghost.
        '''

        mlen = 10 * 60 
        rot_period = 120
        mmax = 2        
        ra0=-10
        dec0=-57.5
        fwhm = 200
        nside = 128
        az_throw = 10

        scs = ScanStrategy(duration=mlen, sample_rate=10, location='spole')

        # Create two Gaussian (main) beams.
        beam_opts = dict(az=0, el=0, polang=28, fwhm=fwhm, lmax=self.lmax, 
                         symmetric=True)
        ghost_opts = dict(az=0, el=0, polang=28, fwhm=fwhm, lmax=self.lmax,
                          symmetric=True, amplitude=1)

        scs.add_to_focal_plane(Beam(**beam_opts))
        scs.add_to_focal_plane(Beam(**ghost_opts))
        
        # Allocate and assign parameters for mapmaking.
        scs.allocate_maps(nside=nside)

        # Set HWP rotation.
        scs.set_hwp_mod(mode='continuous', freq=3.)
        
        # Generate timestreams, bin them and store as attributes.
        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2., binning=False,
                                nside_spin=nside,
                                max_spin=mmax, save_tod=True)

        tod = scs.data(scs.chunks[0], beam=scs.beams[0][0], data_type='tod')
        tod += scs.data(scs.chunks[0], beam=scs.beams[1][0], data_type='tod')
        tod = tod.copy()

        # Solve for the maps.
        maps, cond = scs.solve_for_map(fill=np.nan)

        # To supress warnings
        cond[~np.isfinite(cond)] = 10

        # Repeat with single beam + ghost.
        scs.remove_from_focal_plane(scs.beams[1][0])
        scs.beams[0][0].create_ghost(**ghost_opts)

        scs.reset_hwp_mod()

        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2., binning=False,
                                nside_spin=nside,
                                max_spin=mmax, save_tod=True)

        tod_w_ghost = scs.data(scs.chunks[0], beam=scs.beams[0][0], 
                               data_type='tod')

        # Sum TOD of two beams must match TOD of single beam + ghost.
        np.testing.assert_array_almost_equal(tod, tod_w_ghost, decimal=10)

        # Maps must match.
        maps_w_ghost, cond_w_ghost = scs.solve_for_map(fill=np.nan)

        # To supress warnings
        cond_w_ghost[~np.isfinite(cond_w_ghost)] = 10


        np.testing.assert_array_almost_equal(maps[0,cond<2.5],
                                             maps_w_ghost[0,cond_w_ghost<2.5],
                                             decimal=10)

        np.testing.assert_array_almost_equal(maps[1,cond<2.5],
                                             maps_w_ghost[1,cond_w_ghost<2.5],
                                             decimal=10)

        np.testing.assert_array_almost_equal(maps[2,cond<2.5],
                                             maps_w_ghost[2,cond_w_ghost<2.5],
                                             decimal=10)


    def test_cross_talk(self):
        '''Test if the cross-talk is performing as it should.'''

        mlen = 10 * 60 
        rot_period = 120
        mmax = 2        
        ra0=-10
        dec0=-57.5
        fwhm = 200
        nside = 128
        az_throw = 10

        scs = ScanStrategy(duration=mlen, sample_rate=10, location='spole')

        # Single pair.
        scs.create_focal_plane(nrow=1, ncol=1, fov=0,
                              lmax=self.lmax, fwhm=fwhm)
        
        # Allocate and assign parameters for mapmaking.
        scs.allocate_maps(nside=nside)

        # set instrument rotation.
        scs.set_instr_rot(period=rot_period, angles=[12, 14, 248, 293])

        # Set elevation stepping.
        scs.set_el_steps(rot_period, steps=[0, 2, 4, 8, 10])
        
        # Set HWP rotation.
        scs.set_hwp_mod(mode='stepped', freq=3.)

        beam_a, beam_b = scs.beams[0]


        scs.init_detpair(self.alm, beam_a, beam_b=beam_b, nside_spin=nside,
                         verbose=False)
        
        # Generate timestreams, bin them and store as attributes.
        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2.,
                                max_spin=mmax,
                                reuse_spinmaps=True, 
                                save_tod=True,
                                binning=False,
                                ctalk=0.0)

        tod_a = scs.data(scs.chunks[0], beam=beam_a, data_type='tod').copy()
        tod_b = scs.data(scs.chunks[0], beam=beam_b, data_type='tod').copy()

        # Redo with cross-talk
        ctalk = 0.5

        scs.reset_instr_rot()
        scs.reset_hwp_mod()
        scs.reset_el_steps()

        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2.,
                                max_spin=mmax,
                                reuse_spinmaps=True, 
                                save_tod=True,
                                binning=False,
                                ctalk=ctalk)

        tod_ac = scs.data(scs.chunks[0], beam=beam_a, data_type='tod')
        tod_bc = scs.data(scs.chunks[0], beam=beam_b, data_type='tod')

        np.testing.assert_array_almost_equal(tod_ac, tod_a + ctalk * tod_b)
        np.testing.assert_array_almost_equal(tod_bc, tod_b + ctalk * tod_a)

        # Redo with less cross-talk
        ctalk = 0.000001

        scs.reset_instr_rot()
        scs.reset_hwp_mod()
        scs.reset_el_steps()

        scs.scan_instrument_mpi(self.alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=2.,
                                max_spin=mmax,
                                reuse_spinmaps=True, 
                                save_tod=True,
                                binning=False,
                                ctalk=ctalk)

        tod_acs = scs.data(scs.chunks[0], beam=beam_a, data_type='tod')
        tod_bcs = scs.data(scs.chunks[0], beam=beam_b, data_type='tod')

        np.testing.assert_array_almost_equal(tod_acs, tod_a + ctalk * tod_b)
        np.testing.assert_array_almost_equal(tod_bcs, tod_b + ctalk * tod_a)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 tod_ac, tod_acs)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 tod_bc, tod_bcs)

    def test_interpolate(self):
        '''
        Compare interpolated TOD to default for extremely bandlimited 
        input such that should agree relatively well.
        '''

        mlen = 60
        mmax = 2        
        ra0=-10
        dec0=-57.5
        fwhm = 10 * 60
        nside = 256
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
                                interp=True,
                                binning=False)

        np.testing.assert_array_almost_equal(tod_raw,
                                             scs.tod, decimal=0)

    def test_chunks(self):
        '''Test the _chunk2idx function. '''
        
        mlen = 100 # so 1000 samples
        chunksize = 30
        rot_period = 1.2 # Note, seconds.
        scs = ScanStrategy(duration=mlen, sample_rate=10)

        scs.partition_mission(chunksize=chunksize)

        self.assertEqual(len(scs.chunks), 
                         int(np.ceil(scs.nsamp / float(chunksize))))

        # Take single chunk and subdivide it and check whether we 
        # can correctly access a chunk-sized array.
        scs.set_instr_rot(period=rot_period)

        for chunk in scs.chunks:

            scs.rotate_instr()
            subchunks = scs.subpart_chunk(chunk)
            
            chunklen = chunk['end'] - chunk['start']

            # Start with zero array, let every subchunk add ones
            # to its slice, then test if resulting array is one
            # everywhere.
            arr = np.zeros(chunklen, dtype=int)
            
            for subchunk in subchunks:

                self.assertEqual(subchunk['cidx'], chunk['cidx'])
                self.assertTrue(subchunk['start'] >= chunk['start'])
                self.assertTrue(subchunk['end'] <= chunk['end'])

                qidx_start, qidx_end = scs._chunk2idx(**subchunk)

                arr[qidx_start:qidx_end] += 1

            np.testing.assert_array_equal(arr, np.ones_like(arr))

    def test_preview_pointing_input(self):
        
        # Test if scan_instrument_mpi works with preview_pointing
        # option set.

        scs = ScanStrategy(duration=1, sample_rate=10, location='spole')

        # Should raise error if alm is None with preview_pointing not set.
        alm = None
        with self.assertRaises(TypeError):
            scs.scan_instrument_mpi(alm, verbose=0, 
                                    preview_pointing=False)

        # Should not raise error if alm is provided and preview_pointing set.
        alm = self.alm
        scs.scan_instrument_mpi(alm, verbose=0,
                                preview_pointing=True)

    def test_preview_pointing(self):

        # With preview_pointing set, expect correct proj matrix, 
        # but vec vector should be zero.

        mlen = 6 * 60 
        rot_period = 30
        step_period = rot_period * 2
        mmax = 2        
        ra0=-10
        dec0=-57.5
        fwhm = 10
        nside_out = 32
        az_throw = 10
        scan_speed = 2 # deg / s.

        scs = ScanStrategy(duration=mlen, sample_rate=10, location='spole')

        # Create a 1 x 2 square grid of Gaussian beams.
        scs.create_focal_plane(nrow=1, ncol=2, fov=2,
                              lmax=self.lmax, fwhm=fwhm)
        
        # Allocate and assign parameters for mapmaking.
        scs.allocate_maps(nside=nside_out)

        # set instrument rotation.
        scs.set_instr_rot(period=rot_period, angles=[68, 113, 248, 293])

        # Set elevation stepping.
        scs.set_el_steps(step_period, steps=[0, 1, 2])
        
        # Set HWP rotation.
        scs.set_hwp_mod(mode='continuous', freq=3.)
        
        # First run with preview_pointing set
        alm = None
        preview_pointing = True
        # Generate timestreams, bin them and store as attributes.
        scs.scan_instrument_mpi(alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=scan_speed,
                                nside_spin=nside_out,
                                max_spin=mmax,
                                preview_pointing=preview_pointing)
        
        # Vec should be zero
        np.testing.assert_array_equal(scs.vec, np.zeros((3, 12 * nside_out ** 2)))
        # Save for comparison
        vec_prev = scs.vec
        proj_prev = scs.proj

        # Now run again in default way.
        # Create new dest arrays.
        scs.allocate_maps(nside=nside_out)

        scs.reset_instr_rot()
        scs.reset_hwp_mod()
        scs.reset_el_steps()

        alm = self.alm
        preview_pointing = False

        # Generate timestreams, bin them and store as attributes.
        scs.scan_instrument_mpi(alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=scan_speed,
                                nside_spin=nside_out,
                                max_spin=mmax,
                                preview_pointing=preview_pointing)
        
        # Vec should not be zero now.
        np.testing.assert_equal(np.any(scs.vec), True)

        # Proj should be identical.
        np.testing.assert_array_almost_equal(scs.proj, proj_prev, decimal=9)

        # Run one more time with a ghost. Ghost should not change proj.
        # Create new dest arrays.
        scs.allocate_maps(nside=nside_out)

        alm = self.alm
        preview_pointing = False

        scs.reset_instr_rot()
        scs.reset_hwp_mod()
        scs.reset_el_steps()

        ghost_opts = dict(az=10, el=10, polang=28, fwhm=fwhm, lmax=self.lmax,
                          symmetric=True, amplitude=1)

        scs.beams[0][0].create_ghost(**ghost_opts)

        # Generate timestreams, bin them and store as attributes.
        scs.scan_instrument_mpi(alm, verbose=0, ra0=ra0,
                                dec0=dec0, az_throw=az_throw,
                                scan_speed=scan_speed,
                                nside_spin=nside_out,
                                max_spin=mmax,
                                preview_pointing=preview_pointing)
        
        # Vec should not be zero now.
        np.testing.assert_equal(np.any(scs.vec), True)

        # Proj should be identical.
        np.testing.assert_array_almost_equal(scs.proj, proj_prev, decimal=9)

    def test_offset_beam(self):
        
        mlen = 20 # mission length
        sample_rate = 10
        location='spole'
        lmax = self.lmax
        fwhm = 300
        nside_spin = 256
        polang = 30
        az_off = 20
        el_off = 40
        
        ss = ScanStrategy(mlen, sample_rate=sample_rate,
                          location=location) 

        # Create single detector.
        ss.create_focal_plane(nrow=1, ncol=1, fov=0, no_pairs=True,
                              polang=polang, lmax=lmax, fwhm=fwhm)        

        # Move detector away from boresight.
        ss.beams[0][0].az = az_off
        ss.beams[0][0].el = el_off

        # Start instrument rotated.
        rot_period =  ss.mlen
        ss.set_instr_rot(period=rot_period, start_ang=45)

        ss.set_hwp_mod(mode='stepped', freq=1/20., start_ang=45,
                       angles=[34, 12, 67])

        ss.partition_mission()
        ss.scan_instrument_mpi(self.alm, binning=False, nside_spin=nside_spin,
                               max_spin=2, interp=True)

        # Store the tod and pixel indices made with symmetric beam.
        tod_sym = ss.tod.copy()

        # Now repeat with asymmetric beam and no detector offset.
        # Set offsets to zero such that tods are generated using
        # only the boresight pointing.
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

        # We need to to apply these changes to the angles.
        phi = np.radians(ra)
        theta = np.radians(90 - dec)
        psi = np.radians(-pa)

        # rotate blm
        hp.rotate_alm([blmI, blmE, blmB], psi, theta, phi, lmax=lmax, mmax=lmax)

        # convert beam coeff. back to spin representation.
        blmm2, blmp2 = tools.eb2spin(blmE, blmB)
        ss.beams[0][0].blm = (blmI, blmm2, blmp2)

        ss.reset_instr_rot()
        ss.reset_hwp_mod()

        ss.scan_instrument_mpi(self.alm, binning=False, nside_spin=nside_spin,
                               max_spin=lmax, interp=True) 

        # TODs must agree at least at 2% per sample.
        np.testing.assert_equal(np.abs(ss.tod - tod_sym) < 0.02 * np.std(tod_sym),
                                np.full(tod_sym.size, True))
    
    def test_offset_beam_pol(self):

        mlen = 20 # mission length
        sample_rate = 10
        location='spole'
        lmax = self.lmax
        fwhm = 300
        nside_spin = 256
        polang = 30
        az_off = 20
        el_off = 40

        alm = (self.alm[0]*0., self.alm[1], self.alm[2])
        
        ss = ScanStrategy(mlen, sample_rate=sample_rate,
                          location=location) 

        # Create single detector.
        ss.create_focal_plane(nrow=1, ncol=1, fov=0, no_pairs=True,
                              polang=polang, lmax=lmax, fwhm=fwhm)        

        # Move detector away from boresight.
        ss.beams[0][0].az = az_off
        ss.beams[0][0].el = el_off

        # Start instrument rotated.
        rot_period =  ss.mlen
        ss.set_instr_rot(period=rot_period, start_ang=45)

        ss.set_hwp_mod(mode='stepped', freq=1/20., start_ang=45,
                       angles=[34, 12, 67])

        ss.partition_mission()
        ss.scan_instrument_mpi(alm, binning=False, nside_spin=nside_spin,
                               max_spin=2, interp=True)

        # Store the tod and pixel indices made with symmetric beam.
        tod_sym = ss.tod.copy()

        # Now repeat with asymmetric beam and no detector offset.
        # Set offsets to zero such that tods are generated using
        # only the boresight pointing.
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

        # We need to to apply these changes to the angles.
        phi = np.radians(ra)
        theta = np.radians(90 - dec)
        psi = np.radians(-pa)

        # rotate blm
        hp.rotate_alm([blmI, blmE, blmB], psi, theta, phi, lmax=lmax, mmax=lmax)

        # convert beam coeff. back to spin representation.
        blmm2, blmp2 = tools.eb2spin(blmE, blmB)
        ss.beams[0][0].blm = (blmI, blmm2, blmp2)

        ss.reset_instr_rot()
        ss.reset_hwp_mod()

        ss.scan_instrument_mpi(alm, binning=False, nside_spin=nside_spin,
                               max_spin=lmax, interp=True) 

        # TODs must agree at least at 2% per sample.
        np.testing.assert_equal(np.abs(ss.tod - tod_sym) < 0.02 * np.std(tod_sym),
                                np.full(tod_sym.size, True))
        
    def test_offset_beam_I(self):

        mlen = 20 # mission length
        sample_rate = 10
        location='spole'
        lmax = self.lmax
        fwhm = 300
        nside_spin = 256
        polang = 30
        az_off = 20
        el_off = 40

        alm = (self.alm[0], self.alm[1] * 0., self.alm[2] * 0.)
        
        ss = ScanStrategy(mlen, sample_rate=sample_rate,
                          location=location) 

        # Create single detector.
        ss.create_focal_plane(nrow=1, ncol=1, fov=0, no_pairs=True,
                              polang=polang, lmax=lmax, fwhm=fwhm)        

        # Move detector away from boresight.
        ss.beams[0][0].az = az_off
        ss.beams[0][0].el = el_off

        # Start instrument rotated.
        rot_period =  ss.mlen
        ss.set_instr_rot(period=rot_period, start_ang=45)

        ss.set_hwp_mod(mode='stepped', freq=1/20., start_ang=45,
                       angles=[34, 12, 67])

        ss.partition_mission()
        ss.scan_instrument_mpi(alm, binning=False, nside_spin=nside_spin,
                               max_spin=2, interp=True)

        # Store the tod and pixel indices made with symmetric beam.
        tod_sym = ss.tod.copy()

        # Now repeat with asymmetric beam and no detector offset.
        # Set offsets to zero such that tods are generated using
        # only the boresight pointing.
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

        # We need to to apply these changes to the angles.
        phi = np.radians(ra)
        theta = np.radians(90 - dec)
        psi = np.radians(-pa)

        # rotate blm
        hp.rotate_alm([blmI, blmE, blmB], psi, theta, phi, lmax=lmax, mmax=lmax)

        # convert beam coeff. back to spin representation.
        blmm2, blmp2 = tools.eb2spin(blmE, blmB)
        ss.beams[0][0].blm = (blmI, blmm2, blmp2)

        ss.reset_instr_rot()
        ss.reset_hwp_mod()

        ss.scan_instrument_mpi(alm, binning=False, nside_spin=nside_spin,
                               max_spin=lmax, interp=True) 

        # TODs must agree at least at 2% per sample.
        np.testing.assert_equal(np.abs(ss.tod - tod_sym) < 0.02 * np.std(tod_sym),
                                np.full(tod_sym.size, True))
                
    def test_spinmaps_spin2(self):

        def rand_alm(lmax):
            alm = np.empty(hp.Alm.getsize(lmax), dtype=np.complex128)
            alm[:] = np.random.randn(hp.Alm.getsize(lmax))
            alm += 1j * np.random.randn(hp.Alm.getsize(lmax))
            # Make m=0 modes real.
            alm[:lmax+1] = np.real(alm[:lmax+1])
            return alm

        lmax = 10
        almE, almB = tuple([rand_alm(lmax) for i in range(2)])
        blmE, blmB = tuple([rand_alm(lmax) for i in range(2)])
        spin_values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        nside = 32

        spinmaps = ScanStrategy._spinmaps_spin2(almE, almB, blmE, blmB*0,
                                                spin_values, nside)
        #for sidx, spin in enumerate(spin_values):
        #    print(spin, spinmaps[sidx])

        
if __name__ == '__main__':
    unittest.main()

