import numpy as np
import healpy as hp
import tools
from instrument import ScanStrategy

lmax = 100
fwhm = 300
alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
alm = (alm, alm.copy(), alm.copy())

blm, blmm2 = tools.gauss_blm(fwhm, lmax, pol=True)
blm2 = tools.get_copol_blm(blm.copy())

SC = ScanStrategy(10*60, 1, s_pole=True)

SC.set_focal_plane(3, 10)

#for cidx in xrange(SC.ndet):
#    blm, blmm2 = SC.get_blm(lmax, fwhm=fwhm)

    
