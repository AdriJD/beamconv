import unittest
import numpy as np
import healpy as hp
from cmb_beams import detector
import os
import pickle

opj = os.path.join
test_fpu_dir = os.path.abspath(opj(os.path.dirname(__file__),
                                    'test_data', 'test_fpu'))

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Create .pkl file with beam options and .npy blm 
        array
        '''
        
        blm_name = opj(test_fpu_dir, 'blm_test.npy')
        cls.blm_name = blm_name

        beam_opts = dict(az=10,
                         el=5,
                         polang=90.,
                         btype='PO',
                         amplitude=0.5,
                         blm_file=blm_name)

        cls.beam_opts = beam_opts
#        # Store options as pickle file
#        with open(opj(test_fpu_dir, 'beam_opts'), 'wb') as handle:
#            pickle.dump(beam_opts, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Store blm array
        cls.lmax = 3
        alm_size = hp.Alm.getsize(cls.lmax)
        blm = np.zeros(alm_size, dtype=np.complex128)

        # ell = 0, 1, 2, 3, m=0
        blm = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 4], 
                       dtype=np.complex128)
        cls.blm = blm

        # derived using eq.24 in hivon 2016 (gamma=sigma=0)
        cls.blmm2_expd = np.array([0, 0, 3, 3, 0, -2, -2, 1, 1, 2],
                                  dtype=np.complex128)
        cls.blmp2_expd = np.array([0, 0, 3, 3, 0, 0, 4, 0, 0, 0],
                                  dtype=np.complex128)

        np.save(blm_name, cls.blm)

    @classmethod
    def tearDownClass(cls):
        '''
        Remove the files again
        '''

        os.remove(cls.blm_name)
    
    def test_load_blm(self):
        '''
        Test loading up a blm array
        '''
        
        beam = detector.Beam(**self.beam_opts)
        beam.load_blm()
        
        # test if unpolarized beam is loaded and scaled
        np.testing.assert_array_almost_equal(beam.blm[0],
                                  beam.amplitude*self.blm)

        # test if copol parts are correct
        blmm2_expd = self.blmm2_expd * beam.amplitude
        blmp2_expd = self.blmp2_expd * beam.amplitude
        np.testing.assert_array_almost_equal(blmm2_expd, beam.blm[1])
        np.testing.assert_array_almost_equal(blmp2_expd, beam.blm[2])
                   

if __name__ == '__main__':
    unittest.main()
