import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import healpy as hp
from beamconv import ScanStrategy
import qpoint as qp

cls = np.loadtxt('../ancillary/wmap7_r0p03_lensed_uK_ext.txt',
    unpack=True) # Cl in uK^2

lmax=700
ell, cls = cls[0], cls[1:]
np.random.seed(25)
alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

fwhm = 40
beam_opts = dict(lmax=lmax,
                 fwhm=fwhm,       # Gaussian co-pol beam, so only specify FWHM (arcmin)
                 btype='Gaussian')

map0 = hp.alm2map(alm, 128)
hp.mollview(map0[0])
plt.savefig('map0.png')
plt.close()

ctime0 = 1510000000
mlen = 3 * 24 * 60 * 60        # Mission length in seconds

mmax = 2
nside_spin = 256
preview_pointing = False
verbose = True

ss = ScanStrategy(duration=mlen,
	external_pointing=True,
    sample_rate=12.01, # sample rate in Hz
    location='space',
    ctime0=ctime0)

scan_opts = dict(
    q_bore_func=ss.litebird_scan,
    ctime_func=ss.litebird_ctime,
    use_litebird_scan=True,
    q_bore_kwargs=dict(),
    ctime_kwargs=dict(),
    max_spin=mmax,
    nside_spin=nside_spin,
    preview_pointing=preview_pointing,
    verbose=verbose)

# Create a 3 x 3 square grid of Gaussian beams (f.o.v. is 3 degrees)

#ss.create_focal_plane(nrow=1, ncol=1, fov=1, **beam_opts)
ss.create_focal_plane(nrow=3, ncol=3, fov=3, **beam_opts)

# Calculate tods in two chunks
chunks = ss.partition_mission(0.1 * ss.mlen * ss.fsamp)

# Allocate and assign parameters for mapmaking
ss.allocate_maps(nside=256)

ss.scan_instrument_mpi(alm, **scan_opts)

# Solve for the maps
maps, cond = ss.solve_for_map()

# ss = ScanStrategy(mlen, external_pointing=True, sample_rate=sample_rate,
#     location='space', ctime0=ctime0, nside_out=nside_out, comm=comm)

cond[cond == np.inf] = hp.UNSEEN
cart_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]')#, lonra=[-60, 40], latra=[-70, -40])
plt.figure(1)
hp.mollview(cond, min=2, max=5, **cart_opts)
plt.savefig('cond.png')
plt.close()

print(np.min(maps[0].flatten()))
print(np.max(maps[0].flatten()))

hp.mollview(maps[0], min=-400, max=400, **cart_opts)
plt.savefig('map_I.png')
plt.close()

hp.mollview(maps[1], min=-10, max=10, **cart_opts)
plt.savefig('map_Q.png')
plt.close()
