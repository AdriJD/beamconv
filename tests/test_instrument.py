import unittest
import numpy as np
import healpy as hp
from beamconv import instrument
from beamconv import Beam
import os
import pickle
import warnings

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
                         po_file=blm_name,
                         name='aap',
                         pol='B', 
                         ghost=False)

        cls.beam_opts = beam_opts
        # Store options as pickle file.
        with open(beam_file, 'wb') as handle:
            pickle.dump(beam_opts, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Store blm array.
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

    @classmethod
    def tearDownClass(cls):
        '''
        Remove the file again
        '''

        os.remove(cls.blm_name)
        os.remove(cls.beam_file)

    def test_add_to_focal_plane(self):

        instr = instrument.Instrument()

        beam = Beam()
        
        # Add single beam.
        instr.add_to_focal_plane(beam, combine=True)
        self.assertEqual(instr.ndet, 1)

        for pair in instr.beams:

            self.assertEqual(pair[0], beam)
            self.assertEqual(pair[1], None)

        # Add three more individual beam.
        instr.add_to_focal_plane([beam, beam, beam],
                                      combine=True)

        self.assertEqual(instr.ndet, 4)        
        for pair in instr.beams:
            
            self.assertEqual(pair[0], beam)
            self.assertEqual(pair[1], None)

        # Add a pair.
        instr.add_to_focal_plane([[beam, beam]],
                                      combine=True)

        self.assertEqual(instr.ndet, 6)   
        for n, pair in enumerate(instr.beams):
            
            self.assertEqual(pair[0], beam)
            if n > 3:
                self.assertEqual(pair[1], beam)
            
        # Add two pair.
        instr.add_to_focal_plane([[beam, beam], [beam, beam]],
                                      combine=True)

        self.assertEqual(instr.ndet, 10)
        for n, pair in enumerate(instr.beams):
            
            self.assertEqual(pair[0], beam)
            if n > 3:
                self.assertEqual(pair[1], beam)

        # Start new focal plane with pair.
        instr.add_to_focal_plane([[beam, beam]],
                                      combine=False)

        self.assertEqual(instr.ndet, 2)

        for pair in instr.beams:
            
            self.assertEqual(pair[0], beam)
            self.assertEqual(pair[1], beam)

    def test_remove_from_focal_plane(self):

        instr = instrument.Instrument()

        beam1 = Beam()
        beam2 = Beam()
        beam3 = Beam()
        beam4 = Beam()
        
        # Add single beam per pair.
        instr.add_to_focal_plane([beam1, beam2, beam3, beam4],
                                 combine=True)
        self.assertEqual(instr.ndet, 4)
        
        instr.remove_from_focal_plane([beam1, beam2, beam3, beam4])

        self.assertEqual(instr.ndet, 0)
        self.assertEqual(instr.beams, [])

        # Add pairs.
        instr.add_to_focal_plane([[beam1, beam2], [beam3, beam4]],
                                 combine=True)
        self.assertEqual(instr.ndet, 4)
        
        instr.remove_from_focal_plane([beam1, beam2, beam3])

        self.assertEqual(instr.ndet, 1)
        self.assertEqual(instr.beams, [[None, beam4]])

    def test_load_focal_plane(self):

        instr = instrument.Instrument()
        instr.load_focal_plane(test_fpu_dir)

        with warnings.catch_warnings(record=True) as w:
            # Should be ignored
            instr.load_focal_plane(test_fpu_dir, ghost=True, pol='B') 
            self.assertEqual(len(w), 2) # ghost and pol.
        self.assertEqual(len(instr.beams), 2)

        for pair in instr.beams:

            self.assertEqual(pair[0].az, self.beam_opts['az'])
            self.assertEqual(pair[0].el, self.beam_opts['el'])
            self.assertEqual(pair[0].polang, self.beam_opts['polang'])
            self.assertEqual(pair[0].btype, self.beam_opts['btype'])
            self.assertEqual(pair[0].amplitude, self.beam_opts['amplitude'])
            self.assertEqual(pair[0].po_file, self.beam_opts['po_file'])
            self.assertEqual(pair[0].name, self.beam_opts['name']+'A')
            self.assertEqual(pair[0].pol, 'A')
            self.assertEqual(pair[0].ghost, False)

            self.assertEqual(pair[1].az, self.beam_opts['az'])
            self.assertEqual(pair[1].el, self.beam_opts['el'])
            self.assertEqual(pair[1].polang, self.beam_opts['polang'])
            self.assertEqual(pair[1].btype, self.beam_opts['btype'])
            self.assertEqual(pair[1].amplitude, self.beam_opts['amplitude'])
            self.assertEqual(pair[1].po_file, self.beam_opts['po_file'])
            self.assertEqual(pair[1].name, self.beam_opts['name']+'B')
            self.assertEqual(pair[1].pol, 'B')
            self.assertEqual(pair[1].ghost, False)

        # load up third pair
        instr.load_focal_plane(test_fpu_dir, az=1, el=1, polang=1000)
        self.assertEqual(instr.beams[2][0].az, 1)
        self.assertEqual(instr.beams[2][0].el, 1)
        self.assertEqual(instr.beams[2][0].polang, 1000)
        self.assertEqual(instr.beams[2][1].polang, 1000)

        # Test whether you can change btype when loading up beam.
        instr.load_focal_plane(test_fpu_dir, btype='EG')
        self.assertEqual(instr.beams[3][0].btype, 'EG')
        
        # Test wheter you can overwrite existing beams
        instr.load_focal_plane(test_fpu_dir, combine=False)
        self.assertEqual(len(instr.beams), 1)

        # Test polang_A and polang_B.
        instr.load_focal_plane(test_fpu_dir, combine=False,
                               polang=10)
        self.assertEqual(instr.beams[0][0].polang, 10)
        self.assertEqual(instr.beams[0][1].polang, 10)

        instr.load_focal_plane(test_fpu_dir, combine=False,
                               polang=10, polang_A=20,
                               polang_B=0)
        self.assertEqual(instr.beams[0][0].polang, 30)
        self.assertEqual(instr.beams[0][1].polang, 10)

    def test_create_focal_plane(self):
        '''
        Tests whether name, az, el, ghost, pol kwargs 
        are not propagated, but others are.
        '''
        instr = instrument.Instrument()

        with warnings.catch_warnings(record=True) as w:
            instr.create_focal_plane(name='name_that_should_not_appear',
                                     az=11, el=11, polang=10, ghost=True,
                                     pol='B', fov=10)
            self.assertEqual(len(w), 5) # az, el, ghost, pol, name
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

        # Create some channels.
        instr.create_focal_plane(nrow=10, ncol=10)
        self.assertEqual(instr.ndet, 200)

        # Kill some channels
        instr.kill_channels(killfrac=0.5, pairs=False)
        
        dead_count = 0
        for pair in instr.beams:
            if pair[0].dead:
                dead_count += 1
            if pair[1].dead:
                dead_count += 1

        self.assertEqual(dead_count, instr.ndet / 2.)
        
        # Redo but now with killing pairs
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

    def test_set_btypes(self):

        instr = instrument.Instrument()

        # Create some channels.
        instr.create_focal_plane(nrow=10, ncol=10, btype='PO')

        for pair in instr.beams:
            for beam in pair:
                self.assertEqual(beam.btype, 'PO')

        instr.set_btypes(btype='EG')

        for pair in instr.beams:
            for beam in pair:
                self.assertEqual(beam.btype, 'EG')

        # Test default behaviour
        instr.set_btypes()

        for pair in instr.beams:
            for beam in pair:
                self.assertEqual(beam.btype, 'Gaussian')
        
    def test_global_prop(self):
        
        instr = instrument.Instrument()

        # Create some channels.
        instr.create_focal_plane(nrow=10, ncol=10, btype='PO')

        for pair in instr.beams:
            for beam in pair:
                self.assertEqual(beam.btype, 'PO')

        instr.set_global_prop(dict(btype='EG', polang_bias=10))

        for pair in instr.beams:
            for beam in pair:
                self.assertEqual(beam.btype, 'EG')
                self.assertEqual(beam.polang_bias, 10)

        # Add ghosts
        instr.create_reflected_ghosts()
        instr.create_reflected_ghosts()
        
        for pair in instr.beams:        
            for beam in pair:
                for ghost in beam.ghosts:
                    self.assertEqual(ghost.btype, 'EG')
                    self.assertEqual(ghost.polang_bias, 10)

        instr.set_global_prop(dict(btype='PO', polang_bias=5))

        for pair in instr.beams:        
            for beam in pair:
                for ghost in beam.ghosts:
                    self.assertEqual(beam.btype, 'PO')
                    self.assertEqual(beam.polang_bias, 5)


    def test_add_to_prop(self):

        instr = instrument.Instrument()

        # Create some channels.
        instr.create_focal_plane(nrow=10, ncol=10, btype='PO')

        # Add value 
        instr.add_to_prop(dict(amplitude=10))
        
        for pair in instr.beams:
            for beam in pair:
                self.assertEqual(beam.amplitude, 11)
                
                for ghost in beam.ghosts:
                    self.assertEqual(ghost.amplitude, 11)

        # Add value only to main beams.
        instr.add_to_prop(dict(amplitude=10), incl_ghosts=False)
        
        for pair in instr.beams:
            for beam in pair:
                self.assertEqual(beam.amplitude, 21)
                
                for ghost in beam.ghosts:
                    self.assertEqual(ghost.amplitude, 11)

        # Add value only to A beams.
        instr.add_to_prop(dict(amplitude=10), incl_ghosts=True,
                          no_B=True)
        
        for pair in instr.beams:
            self.assertEqual(pair[0].amplitude, 31)
            self.assertEqual(pair[1].amplitude, 21)
            
            for bidx, beam in enumerate(pair):
                
                for ghost in beam.ghosts:
                    if bidx == 0:
                        self.assertEqual(ghost.amplitude, 21)
                    else:
                        self.assertEqual(ghost.amplitude, 11)

        # Add random var per pair
        instr.add_to_prop(dict(polang=100), rand_stdev=30,
                              per_pair=True)

        for pair in instr.beams:
            self.assertAlmostEqual(pair[0].polang, pair[1].polang - 90)
            
            for beam in pair:                
                for ghost in beam.ghosts:
                    self.assertAlmostEqual(beam.polang, ghost.polang)
                        
        # Add random var.
        instr.add_to_prop(dict(polang=100), rand_stdev=30)

        for pair in instr.beams:
            self.assertNotAlmostEqual(pair[0].polang, pair[1].polang - 90)
            
            for beam in pair:                
                for ghost in beam.ghosts:
                    self.assertEqual(beam.polang, ghost.polang)

if __name__ == '__main__':
    unittest.main()
