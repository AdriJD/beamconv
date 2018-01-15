import unittest
import numpy as np
import healpy as hp
from cmb_beams import detector
import os
import pickle

opj = os.path.join
test_data_dir = os.path.abspath(opj(os.path.dirname(__file__),
                                    'test_data'))

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Create a .npy blm array
        '''
        
        blm_name = opj(test_data_dir, 'blm_test.npy')
        cls.blm_name = blm_name

        blm_cross_name = opj(test_data_dir, 'blm_cross_test.npy')
        cls.blm_cross_name = blm_cross_name

        beam_opts = dict(az=10,
                         el=5,
                         polang=90.,
                         btype='PO',
                         amplitude=0.5,
                         po_file=blm_name,
                         deconv_q=False,
                         normalize=False)

        cls.beam_opts = beam_opts

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

        # also save explicit co- and cross-polar beams
        # but just use blm three times
        np.save(blm_cross_name, np.asarray([cls.blm, cls.blm,
                                           cls.blm]))

    @classmethod
    def tearDownClass(cls):
        '''
        Remove the file again
        '''

        os.remove(cls.blm_name)
    
    def test_load_blm(self):
        '''
        Test loading up a blm array
        '''
        
        beam = detector.Beam(**self.beam_opts)
        
        # test if unpolarized beam is loaded and scaled
        np.testing.assert_array_almost_equal(beam.blm[0],
                                  beam.amplitude*self.blm)

        # test if copol parts are correct
        blmm2_expd = self.blmm2_expd * beam.amplitude
        blmp2_expd = self.blmp2_expd * beam.amplitude
        np.testing.assert_array_almost_equal(blmm2_expd, beam.blm[1])
        np.testing.assert_array_almost_equal(blmp2_expd, beam.blm[2])
                   
        # test if you can also load up the full beam
        beam.delete_blm()
        beam.po_file = self.blm_cross_name
        np.testing.assert_array_almost_equal(self.blm*beam.amplitude,
                                             beam.blm[0])
        np.testing.assert_array_almost_equal(self.blm*beam.amplitude,
                                             beam.blm[1])
        np.testing.assert_array_almost_equal(self.blm*beam.amplitude,
                                             beam.blm[2])

if __name__ == '__main__':
    unittest.main()
