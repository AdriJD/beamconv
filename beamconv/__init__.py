'''
This package contains tools to simulate a CMB experiment that convolves the sky 
with an asymmetric 4-pi beam.
'''

from .instrument import MPIBase, Instrument, ScanStrategy
from .detector import Beam


