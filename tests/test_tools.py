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

    def test_tukey_window(self):

        # Even input.
        n = 10
        w = tools.tukey_window(n)

        exp_ans = np.asarray(
            [0, 0.292292, 0.827430, 1, 1, 1, 1, 0.827430, 0.292292, 0])
        np.testing.assert_array_almost_equal(w, exp_ans)

        # Odd input
        n = 11
        w = tools.tukey_window(n)

        exp_ans = np.asarray(
            [0, 0.25, 0.75, 1, 1, 1, 1, 1, 0.75, 0.25, 0])
        np.testing.assert_array_almost_equal(w, exp_ans)

    def test_filter_ft_hwp(self):

        fd = np.ones(10, dtype=np.complex128)
        fd += 1j
        center_idx = 4
        filter_width = 4
        tools.filter_ft_hwp(fd, center_idx, filter_width)

        exp_ans = np.zeros(10, dtype=np.complex128)
        exp_ans[3] = .75 + .75j
        exp_ans[4] = 1. + 1.j
        exp_ans[5] = .75 + .75j

        np.testing.assert_array_almost_equal(fd, exp_ans)

    def test_filter_tod_hwp(self):

        fsamp = 10.3
        nsamp = 103 # I.e. 10 sec of data.
        
        t = np.arange(nsamp, dtype=np.float64) # Samples.
        
        tod = np.sin(2 * t * (2 * np.pi) / float(nsamp)) # f = .2 Hz
        tod += np.sin(4 * t * (2 * np.pi) / float(nsamp)) # f = .4 Hz

        hwp_freq = .1
        
        tools.filter_tod_hwp(tod, fsamp, hwp_freq)

        # We should have filtered out the f = .2 Hz component.
        exp_ans = np.sin(4 * t * (2 * np.pi) / float(nsamp))# f = .4 Hz
        np.testing.assert_array_almost_equal(tod, exp_ans)

    def test_mueller2spin_identity(self):

        identity = np.eye(4)
        mat_spin = tools.mueller2spin(identity)
        np.testing.assert_almost_equal(mat_spin, identity)

    def test_mueller2spin(self):

        mat = np.arange(16).reshape(4,4)
        mat_spin = tools.mueller2spin(mat)
        exp_ans = np.asarray(
            [[0, (1 - 2j) / np.sqrt(2), (1 + 2j) / np.sqrt(2), 3],
             [(2 + 4j) * np.sqrt(2), (15 + 3j) / 2., (-5 + 15j) / 2., np.sqrt(-36 + 77j)],
             [(2 - 4j) * np.sqrt(2), (-5 - 15j) / 2., (15 - 3j) / 2., np.sqrt(-36 - 77j)],
             [12, (13 - 14j) / np.sqrt(2), (13 + 14j) / np.sqrt(2), 15]]
            )
        np.testing.assert_almost_equal(mat_spin, exp_ans)

    def test_shift_blm_shift0(self):

        # Test if shift of 0 leaves input unchanged.
        lmax = 5
        blmE = np.random.randn(hp.Alm.getsize(lmax)).astype(np.complex128)
        blmE += 1j * np.random.randn(hp.Alm.getsize(lmax))        
        blmB = np.random.randn(hp.Alm.getsize(lmax)).astype(np.complex128)
        blmB += 1j * np.random.randn(hp.Alm.getsize(lmax))        

        blmE_new, blmB_new = tools.shift_blm(blmE, blmB, 0)

        np.testing.assert_almost_equal(blmE_new, blmE)
        np.testing.assert_almost_equal(blmB_new, blmB)

    def test_shift_blm_shift4(self):

        blmp2 = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 4, 5],
                          dtype=np.complex128)
        blmm2 = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)
        blmE, blmB = tools.spin2eb(blmm2, blmp2)
        
        blmE_new, blmB_new = tools.shift_blm(blmE, blmB, 4)
        blmm2_new, blmp2_new = tools.eb2spin(blmE_new, blmB_new)
        
        blmp2_exp = np.array(
            [0, 0, 0, 0, -5j, 0, 0, 4j, 4j, -3j, -3j, -3j, 2j, 2j, 1],
            dtype=np.complex128)

        blmm2_exp = np.array(
            [0, 0, 0, 0, 5j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.complex128)
        
        np.testing.assert_almost_equal(blmm2_new, blmm2_exp)
        np.testing.assert_almost_equal(blmp2_new, blmp2_exp)        

    def test_shift_nlm_mimic_unpol2pol(self):
        # Test if shift by -2 is equal to unpol2pol.
        blm = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2j, 3, 3, 3j, 4, 4j, 5j],
                          dtype=np.complex128)
        blmm2, blmp2 = tools.shift_blm(blm, blm, -2, eb=False)
        blmm2_exp, blmp2_exp = tools.unpol2pol(blm)

        # Comparison is not perfect, because unpol2pol will set
        # all l=0 and l=1 elemets to zero. shift_blm does not
        # have that restriction. So, let us manually fix that.
        blmm2[5] = 0
        
        np.testing.assert_almost_equal(blmm2, blmm2_exp)
        np.testing.assert_almost_equal(blmp2, blmp2_exp)        
        
    def test_shift_blm_undo_unpol2pol(self):
        # Test if shift by +2 undoes unpol2pol operation.        
        blm = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2j, 3, 3, 3j, 4, 4j, 5j],
                          dtype=np.complex128)
        blmm2, blmp2 = tools.unpol2pol(blm)
        
        blmm2, blmp2 = tools.shift_blm(blmm2, blmp2, 2, eb=False)

        # We now expect that blmm2 and blmp2 are equal to blm again.
        # The only difference is that we have lost some l=0 and l=1
        # elements along the way.

        blmm2_exp = np.array([0, 0, 1, 1, 1, 0, 0, 2, 2j, 0, 0, 3j, 0, 0, 0])
        blmp2_exp = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2j, 3, 3, 3j, 4, 4j, 5j])

        np.testing.assert_almost_equal(blmm2, blmm2_exp)
        np.testing.assert_almost_equal(blmp2, blmp2_exp)        

    def test_shift_blm_s2a4(self):
        # Test if copolar beam does what we expect.
        blmm2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1j, 0, 0, 0],
                         dtype=np.complex128) 
        blmp2 = np.zeros_like(blmm2)
        blmm2, blmp2 = tools.shift_blm(blmm2, blmp2, 4, eb=False)        

        blmm2_exp = np.zeros_like(blmm2)
        blmp2_exp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1j, 0, 0, 0],
                         dtype=np.complex128) 

        np.testing.assert_almost_equal(blmm2, blmm2_exp)
        np.testing.assert_almost_equal(blmp2, blmp2_exp)        
        
if __name__ == '__main__':
    unittest.main()
        
