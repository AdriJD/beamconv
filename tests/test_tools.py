import unittest
import numpy as np
import healpy as hp
from cmb_beams import tools

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
        for i in xrange(3):
            np.testing.assert_array_almost_equal(self.alm_tup[i],
                                                 alm_tup_t[i])

        # lmax = 0 case
        # single array
        alm_t = tools.trunc_alm(self.alm, 0, mmax_old=self.mmax)
        self.assertTrue(alm_t.size == 1)
        self.assertEqual(self.alm[0], alm_t[0])

        # tuple of alms
        alm_tup_t = tools.trunc_alm(self.alm_tup, 0, mmax_old=self.mmax)
        for i in xrange(3):
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
        for idx in xrange(self.alm.size):

            ell, m = hp.Alm.getlm(self.lmax, idx)
            if ell > new_lmax or m > new_lmax or m > self.mmax:
                continue
            idx_new = hp.Alm.getidx(new_lmax, ell, m)

            self.assertEqual(self.alm[idx], alm_t[idx_new])
            
            for i in xrange(3):
                self.assertEqual(self.alm_tup[i][idx], alm_tup_t[i][idx_new])
            n += 1

        # check if no other elements in alm_t are missed
        self.assertEqual(n, alm_t.size)

        return
    

if __name__ == '__main__':
    unittest.main()
        
