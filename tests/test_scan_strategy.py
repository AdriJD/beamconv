import unittest
import numpy as np
import healpy as hp
from beamconv import instrument
import os
import pickle

opj = os.path.join

class TestTools(unittest.TestCase):

    def test_init(self):
        
        scs = instrument.ScanStrategy(200, sample_rate=10)
        self.assertEqual(scs.mlen, 200)
        self.assertEqual(scs.fsamp, 10.)
        self.assertEqual(scs.nsamp, 2000)

        self.assertRaises(AttributeError, setattr, scs, 'fsamp', 1)
        self.assertRaises(AttributeError, setattr, scs, 'mlen', 1)
        self.assertRaises(AttributeError, setattr, scs, 'nsamp', 1)
    
    def test_el_steps(self):
        
        scs = instrument.ScanStrategy(200)
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

if __name__ == '__main__':
    unittest.main()

