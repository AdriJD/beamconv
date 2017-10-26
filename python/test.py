import numpy as np
import healpy as hp
import tools
from instrument import ScanStrategy

lmax = 100
fwhm = 300
nside = 256

# Load up alm and blm
cls = np.loadtxt('wmap7_r0p03_lensed_uK_ext.txt', unpack=True)
ell, cls = cls[0], cls[1:]
alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # check if these make any sense (cl vs Dl, units etc.)
# looks like Cl in uK^2

blm, blmm2 = tools.gauss_blm(fwhm, lmax, pol=True)
blm = tools.get_copol_blm(blm.copy())

# South pole instrument
b2 = ScanStrategy(10*60*60, # mission duration in sec.
                  10, # 10 Hz sample rate
                  location='spole')


# calculate spinmaps 
b2.get_spinmaps(alm, blm, 5)

# Initiate a single detector
b2.set_focal_plane(1, 10)
az_off = b2.chn_pr_az
el_off = b2.chn_pr_el


#b2.set_instr_rot(period=1.1*60) # Rotate instrument (period in sec)

chunks = b2.partition_mission(int(60*60*b2.fsamp)) # calculate tod in chunks of # samples
print chunks

vec = np.zeros((3, 12*b2.nside_out**2), dtype=float)
proj = np.zeros((6, 12*b2.nside_out**2), dtype=float)

for cidx, chunk in enumerate(chunks):
    print 'chunk'
    print chunk
#    b2.constant_el_scan(-10, -57.5, 50, 1, el_step=2 * (cidx % 10 - 5),
#                         **chunk) # this sets the boresight and todsize

    b2.constant_el_scan(-10, -57.5, 10, 1, el_step=1 * (cidx % 1000 - 5),
                         **chunk) # this sets the boresight and todsize

    for subchunk in b2.subpart_chunk(chunk):
        print 'slang'
        print subchunk
        b2.scan(az_off=az_off, el_off=el_off, **subchunk)
 
        b2.bin_tod(az_off, el_off)
       
        vec += b2.depo['vec']
        proj += b2.depo['proj']

print vec[0][np.where(vec[0])]
print vec
maps = b2.solve_map(vec=vec[0], proj=proj[0], copy=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# plot the input map and spectra
plt.figure()
hp.mollview(maps)
plt.savefig('test_map_I.png')
plt.close()

maps_raw = hp.alm2map(alm, 256)
plt.figure()
hp.mollview(maps_raw[0])
plt.savefig('raw_map_I.png')
plt.close()

dell = ell * (ell + 1) / 2. / np.pi
plt.figure()
plt.plot(ell, dell * cls[0])
plt.plot(ell, dell * cls[1])
plt.plot(ell, dell * cls[2])
plt.plot(ell, dell * cls[3])
plt.savefig('cls.png')
plt.close()
