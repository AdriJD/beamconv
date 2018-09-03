import time
import numpy as np
import healpy as hp
from beamconv import ScanStrategy
from beamconv import Beam

scan_opts = dict(duration=3600,
                 sample_rate=100)

lmax_range = np.logspace(np.log10(500), np.log10(1000), 20, dtype=int)
mmax_range = np.arange(2, 11, dtype=int)
timings = np.ones((mmax_range.size, lmax_range.size)) * np.nan

alm = np.zeros((3, hp.Alm.getsize(lmax=4000)), dtype=np.complex128)


S = ScanStrategy(**scan_opts)

for lidx, lmax in enumerate(lmax_range):

    n = 0
    nside = 2
    while nside < 0.5 * lmax:
        n += 1
        nside = 2 ** n

    beam_opts = dict(az=0, el=0, polang=0,
                         fwhm=40, btype='Gaussian', 
                         lmax=lmax)

    beam = Beam(**beam_opts)
    beam.blm
        
    for enumerate(mmax in mmax_range):

        t0 = time.time()
        S.init_detpair(alm, beam, beam_b=None, nside_spin=nside, max_spin=mmax,
                       verbose=False)
        t1 = time.time()

        print('{}, {}, {}: {}'.format(lmax, mmax, nside, t1-t0))

        timings[midx,lidx] = t1 - t0

np.save('./timings.npy', timings)
np.save('./lmax_range.npy', lmax_range)
np.save('./mmax_range.npy', mmax_range)
