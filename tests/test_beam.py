import unittest
import numpy as np
import healpy as hp
from beamconv import Beam
import os
import pickle

opj = os.path.join
test_data_dir = os.path.abspath(opj(os.path.dirname(__file__),
                                    'test_data'))

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Create .npy and .fits blm arrays.
        '''
        
        blm_name = opj(test_data_dir, 'blm_test.npy')
        cls.blm_name = blm_name

        blm_cross_name = opj(test_data_dir, 'blm_cross_test.npy')
        cls.blm_cross_name = blm_cross_name

        # .fits versions
        cls.blm_name_fits = cls.blm_name.replace('.npy', '.fits')
        cls.blm_cross_name_fits = cls.blm_cross_name.replace('.npy',
                                                             '.fits')
        # .fits versions that are truncated in m
        cls.blm_name_mmax_fits = opj(test_data_dir,
                                     'blm_test_mmax.fits')
        cls.blm_cross_name_mmax_fits = opj(test_data_dir,
                                           'blm_cross_test_mmax.fits')
        
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

        # Derived using eq.24 in hivon 2016 (gamma=sigma=0).
        cls.blmm2_expd = np.array([0, 0, 3, 3, 0, -2, -2, 1, 1, 2],
                                  dtype=np.complex128)
        cls.blmp2_expd = np.array([0, 0, 3, 3, 0, 0, 4, 0, 0, 0],
                                  dtype=np.complex128)

        np.save(blm_name, cls.blm)
        
        # Older healpy versions have bugs with write_alm writing multiple alm.
        if int(hp.__version__.replace('.','')) >= 1101:
            hp.write_alm(cls.blm_name_fits, cls.blm, overwrite=True)

        # Also save explicit co- and cross-polar beams
        # but just use blm three times.
        np.save(blm_cross_name, np.asarray([cls.blm, cls.blm,
                                           cls.blm]))

        if int(hp.__version__.replace('.','')) >= 1101:
            hp.write_alm(cls.blm_cross_name_fits,
                     [cls.blm, cls.blm, cls.blm], overwrite=True)

        # Write .fits files that have mmax = 2.
        if int(hp.__version__.replace('.','')) >= 1101:
            hp.write_alm(cls.blm_name_mmax_fits, cls.blm, overwrite=True, mmax=2)
            hp.write_alm(cls.blm_cross_name_mmax_fits,
                         [cls.blm, cls.blm, cls.blm], overwrite=True, mmax=2)
        
    @classmethod
    def tearDownClass(cls):
        '''
        Remove the file again
        '''

        os.remove(cls.blm_name)

    def test_init(self):
        '''
        Test initializing a Beam object.
        '''
        
        beam = Beam(fwhm=0.)
        self.assertEqual(0, beam.fwhm)
        
    def test_load_blm(self):
        '''
        Test loading up a blm array
        '''
        
        beam = Beam(**self.beam_opts)
        
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

    def test_load_blm_fits(self):
        '''
        Test loading up a blm .fits array
        '''

        if int(hp.__version__.replace('.','')) < 1101:
            return

        beam = Beam(**self.beam_opts)
        beam.po_file = self.blm_name_fits
        
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
        beam.po_file = self.blm_cross_name_fits
        np.testing.assert_array_almost_equal(self.blm*beam.amplitude,
                                             beam.blm[0])
        np.testing.assert_array_almost_equal(self.blm*beam.amplitude,
                                             beam.blm[1])
        np.testing.assert_array_almost_equal(self.blm*beam.amplitude,
                                             beam.blm[2])

    def test_load_blm_mmax_fits(self):
        '''
        Test loading up a blm .fits array that has mmax < lmax
        '''

        if int(hp.__version__.replace('.','')) < 1101:
            return

        beam = Beam(**self.beam_opts)
        beam.po_file = self.blm_name_mmax_fits

        # test if unpolarized beam is loaded and scaled
        blm_expd = beam.amplitude * self.blm
        blm_expd[-1] = 0
        np.testing.assert_array_almost_equal(beam.blm[0],
                                             blm_expd)

        # After loading we expect mmax to be equal to the truncated value.
        self.assertEqual(beam.mmax, 2)
        
        # Test if you can also load up the full beam, note that these
        # are just 3 copies of main beam, but truncated.
        beam.delete_blm()
        beam.po_file = self.blm_cross_name_mmax_fits
        
        np.testing.assert_array_almost_equal(blm_expd,
                                             beam.blm[0])
        np.testing.assert_array_almost_equal(blm_expd,
                                             beam.blm[1])
        np.testing.assert_array_almost_equal(blm_expd,
                                             beam.blm[2])

    def test_polang_error(self):
        '''
        Test wheter polang_truth = polang + polang_error
        and correctly updates.
        '''    
        beam = Beam(**self.beam_opts)

        # Default Value.
        self.assertEqual(beam.polang_error, 0.)

        self.assertEqual(beam.polang_truth,
                         beam.polang + beam.polang_error)
        beam.polang = 40
        self.assertEqual(beam.polang_truth,
                         beam.polang + beam.polang_error)
        beam.polang_error = 40
        self.assertEqual(beam.polang_truth,
                         beam.polang + beam.polang_error)

        # User cannot change polang_truth directly.
        try:
            beam.polang_truth = 42
        except:
            AttributeError
        self.assertEqual(beam.polang_truth,
                         beam.polang + beam.polang_error)

if __name__ == '__main__':
    unittest.main()

