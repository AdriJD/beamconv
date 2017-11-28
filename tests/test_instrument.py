import unittest
import numpy as np
import healpy as hp
from cmb_beams import instrument
import os
import pickle

opj = os.path.join
test_fpu_dir = os.path.abspath(opj(os.path.dirname(__file__),
                                    'test_data', 'test_fpu'))

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Create .pkl files with beam options and .npy blm 
        array
        '''
        
        blm_name = opj(test_fpu_dir, 'blm_test.npy')
        cls.blm_name = blm_name
        
        beam_file = opj(test_fpu_dir, 'beam_opts.pkl')
        cls.beam_file = beam_file

        beam_opts = dict(az=10,
                         el=5,
                         polang=90.,
                         btype='PO',
                         amplitude=0.5,
                         blm_file=blm_name,
                         name='aap',
                         pol='B', # ign
                         ghost=False)

        cls.beam_opts = beam_opts
        # Store options as pickle file
        with open(beam_file, 'wb') as handle:
            pickle.dump(beam_opts, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        Remove the file again
        '''

        os.remove(cls.blm_name)
        os.remove(cls.beam_file)

    def test_load_focal_plane(self):

        instr = instrument.Instrument()
        instr.load_focal_plane(test_fpu_dir)

        instr.load_focal_plane(test_fpu_dir, ghost=True, pol='B') # should be ignored
        self.assertEqual(len(instr.beams), 2)

        for pair in instr.beams:

            self.assertEqual(pair[0].az, self.beam_opts['az'])
            self.assertEqual(pair[0].el, self.beam_opts['el'])
            self.assertEqual(pair[0].polang, self.beam_opts['polang'])
            self.assertEqual(pair[0].btype, self.beam_opts['btype'])
            self.assertEqual(pair[0].amplitude, self.beam_opts['amplitude'])
            self.assertEqual(pair[0].blm_file, self.beam_opts['blm_file'])
            self.assertEqual(pair[0].name, self.beam_opts['name']+'A')
            self.assertEqual(pair[0].pol, 'A')
            self.assertEqual(pair[0].ghost, False)

            self.assertEqual(pair[1].az, self.beam_opts['az'])
            self.assertEqual(pair[1].el, self.beam_opts['el'])
            self.assertEqual(pair[1].polang, self.beam_opts['polang']+90)
            self.assertEqual(pair[1].btype, self.beam_opts['btype'])
            self.assertEqual(pair[1].amplitude, self.beam_opts['amplitude'])
            self.assertEqual(pair[1].blm_file, self.beam_opts['blm_file'])
            self.assertEqual(pair[1].name, self.beam_opts['name']+'B')
            self.assertEqual(pair[1].pol, 'B')
            self.assertEqual(pair[1].ghost, False)

        # load up third pair
        instr.load_focal_plane(test_fpu_dir, ghost=True, az=1, el=1, polang=1000)
        self.assertEqual(instr.beams[2][0].az, 1)
        self.assertEqual(instr.beams[2][0].el, 1)
        self.assertEqual(instr.beams[2][0].polang, 1000)
        self.assertEqual(instr.beams[2][1].polang, 1090)

    def test_create_focal_plane(self):
        '''
        Tests whether name, az, el, ghost, pol kwargs 
        are not propagated, but others are.
        '''
        instr = instrument.Instrument()
        instr.create_focal_plane(name='name_that_should_not_appear',
                                 az=11, el=11, polang=10, ghost=True,
                                 pol='B', fov=10)
        self.assertEqual(len(instr.beams), 1)
        self.assertEqual(instr.ndet, 2)

        for bidx, beam in enumerate(instr.beams[0]):
            self.assertNotEqual(beam.name, 'name_that_shoud_not_appear')
            self.assertEqual(beam.az, -5)
            self.assertEqual(beam.el, -5)
            if bidx == 0:
                self.assertEqual(beam.pol, 'A')
                self.assertEqual(beam.polang, 10)
            elif bidx == 1:
                self.assertEqual(beam.pol, 'B')
                self.assertEqual(beam.polang, 100)
            self.assertEqual(beam.ghost, False)

        instr.create_focal_plane()
        self.assertEqual(len(instr.beams), 2)
        self.assertEqual(instr.ndet, 4)
        
    def test_kill_channels(self):
        
        instr = instrument.Instrument()

        # create some channels
        instr.create_focal_plane(nrow=10, ncol=10)
        self.assertEqual(instr.ndet, 200)

        # kill some channels
        instr.kill_channels(killfrac=0.5, pairs=False)
        
        dead_count = 0
        for pair in instr.beams:
            if pair[0].dead:
                dead_count += 1
            if pair[1].dead:
                dead_count += 1

        self.assertEqual(dead_count, instr.ndet / 2.)
        
        # redo but now with killing pairs
        instr_2 = instrument.Instrument()
        instr_2.create_focal_plane(nrow=10, ncol=10)
        instr_2.kill_channels(killfrac=0.5, pairs=True)
        dead_count = 0
        for pair in instr_2.beams:
            if pair[0].dead and pair[1].dead:
                dead_count += 2
            else:
                self.assertEqual(pair[0].dead, False)
                self.assertEqual(pair[1].dead, False)

        self.assertEqual(dead_count, instr_2.ndet / 2.)
