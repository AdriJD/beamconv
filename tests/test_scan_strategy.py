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
        
        cls.alm = tuple([rand_alm(lmax) for i in xrange(3)])
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

        for step in xrange(12):
            el = scs.el_step_gen.next()
            scs.step_dict['step'] = el
            self.assertEqual(el, step%5)
            

        self.assertEqual(scs.step_dict['step'], el)
        scs.step_dict['remainder'] = 100
        scs.reset_el_steps()
        self.assertEqual(scs.step_dict['step'], 0)
        self.assertEqual(scs.step_dict['remainder'], 0)

        for step in xrange(nsteps):
            el = scs.el_step_gen.next()
            self.assertEqual(el, step%5)

        scs.reset_el_steps()
        self.assertEqual(scs.el_step_gen.next(), 0)
        self.assertEqual(scs.el_step_gen.next(), 1)

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
        
        func, func_c = scs.spinmaps['main_beam']
        self.assertEqual(func.shape, (mmax + 1, 12 * nside ** 2))
        self.assertEqual(func_c.shape, (2 * mmax + 1, 12 * nside ** 2))

        # Since we have a infinitely narrow Gaussian the convolved
        # maps should just match the input (up to healpix quadrature
        # wonkyness).
        input_map = hp.alm2map(self.alm, nside) # I, Q, U
        zero_map = np.zeros_like(input_map[0])
        np.testing.assert_array_almost_equal(input_map[0],
                                             func[0], decimal=6)
        # s = 2 Pol map should be Q \pm i U
        np.testing.assert_array_almost_equal(input_map[1] + 1j * input_map[2],
                                             func_c[mmax + 2], decimal=6)

        # Test if rest of maps are zero.
        for i in xrange(1, mmax + 1):
            np.testing.assert_array_almost_equal(zero_map,
                                                 func[i], decimal=6)
            
        for i in xrange(1, 2 * mmax + 1):
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
        
        func, func_c = scs.spinmaps['main_beam']
        self.assertEqual(func.shape, (mmax + 1, 12 * nside ** 2))
        self.assertEqual(func_c.shape, (2 * mmax + 1, 12 * nside ** 2))
            
if __name__ == '__main__':
    unittest.main()

