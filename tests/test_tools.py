import unittest
import numpy as np
import healpy as hp
from beamconv import tools

class TestTools(unittest.TestCase):
    
    def test_trunc_alm(self):
        '''

        '''

        np.random.seed(39)
        self.lmax = 10
        self.mmax = 3
        self.alm_size = hp.Alm.getsize(self.lmax, mmax=self.mmax)
        self.alm = np.random.randn(self.alm_size)
        self.alm = self.alm + 1j * np.random.randn(self.alm_size)
        self.alm_tup = (self.alm, self.alm, self.alm)

        # forget mmax
        self.assertRaises(ValueError, tools.trunc_alm, self.alm, self.lmax)

        # Do nothing case
        # single array
        alm_t = tools.trunc_alm(self.alm, self.lmax, mmax_old=self.mmax)
        np.testing.assert_array_almost_equal(self.alm, alm_t)

        # tuple of alms
        alm_tup_t = tools.trunc_alm(self.alm_tup, self.lmax, 
                                    mmax_old=self.mmax)
        for i in range(3):
            np.testing.assert_array_almost_equal(self.alm_tup[i],
                                                 alm_tup_t[i])

        # lmax = 0 case
        # single array
        alm_t = tools.trunc_alm(self.alm, 0, mmax_old=self.mmax)
        self.assertTrue(alm_t.size == 1)
        self.assertEqual(self.alm[0], alm_t[0])

        # tuple of alms
        alm_tup_t = tools.trunc_alm(self.alm_tup, 0, mmax_old=self.mmax)
        for i in range(3):
            self.assertTrue(alm_tup_t[i].size == 1)
            self.assertEqual(self.alm_tup[i][0], alm_tup_t[i][0])

        # check in loop
        new_lmax = 7
        # single array
        alm_t = tools.trunc_alm(self.alm, new_lmax, mmax_old=self.mmax)

        # tuple of alms
        alm_tup_t = tools.trunc_alm(self.alm_tup, new_lmax, 
                                    mmax_old=self.mmax)

        n = 0
        for idx in range(self.alm.size):

            ell, m = hp.Alm.getlm(self.lmax, idx)
            if ell > new_lmax or m > new_lmax or m > self.mmax:
                continue
            idx_new = hp.Alm.getidx(new_lmax, ell, m)

            self.assertEqual(self.alm[idx], alm_t[idx_new])
            
            for i in range(3):
                self.assertEqual(self.alm_tup[i][idx], alm_tup_t[i][idx_new])
            n += 1

        # check if no other elements in alm_t are missed
        self.assertEqual(n, alm_t.size)

        return


    def test_get_copol_blm(self):
        '''
        Test if converting the unpolarized beam to 
        the copolarized expressions follows the expression in 
        Hivon 2016.
        '''
        blm_in = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5],
                          dtype=np.complex128)
        
        blm, blmm2, blmp2 = tools.get_copol_blm(blm_in, normalize=False,
                                                deconv_q=False, c2_fwhm=None)

        # derived using eq.24 in hivon 2016 (gamma=sigma=0)
        blmm2_expd = np.array([0, 0, 3, 3, 3, 0, -2, -2, -2, 1, 1, 1, 2, 2, 3],
                                  dtype=np.complex128)
        blmp2_expd = np.array([0, 0, 3, 3, 3, 0, 0, 4, 4, 0, 0, 5, 0, 0, 0],
                                  dtype=np.complex128)

        np.testing.assert_array_almost_equal(blm_in, blm)
        np.testing.assert_array_almost_equal(blmm2_expd, blmm2)
        np.testing.assert_array_almost_equal(blmp2_expd, blmp2)

    def test_sawtooth_wave(self):

        az_truth = np.array([0, 2, 4, 6, 8, 10, 0, 2, 4, 6, 8, 10],
                             dtype=float)
        az = tools.sawtooth_wave(12, 2, 12)
        
        np.testing.assert_array_equal(az_truth, az)

    def test_cross_talk(self):
        '''Test the cross_talk function. '''
        
        tod_a = np.ones(100, dtype=float)
        tod_b = np.ones(100, dtype=float)
        
        tools.cross_talk(tod_a, tod_b, ctalk=1)

        twos = np.ones(100, dtype=float) * 2

        np.testing.assert_array_equal(tod_a, twos)
        np.testing.assert_array_equal(tod_b, twos)

        # With other ctalk.
        ctalk = 0.2
        tod_a = np.random.randn(100)
        tod_b = np.random.randn(100)

        expt_a = tod_a + ctalk * tod_b
        expt_b = tod_b + ctalk * tod_a
        
        tools.cross_talk(tod_a, tod_b, ctalk=ctalk)        

        np.testing.assert_array_almost_equal(tod_a, expt_a)
        np.testing.assert_array_almost_equal(tod_b, expt_b)

if __name__ == '__main__':
    unittest.main()
        
