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

blm, blmm2 = tools.gauss_blm(fwhm, lmax, pol=True)
blm = tools.get_copol_blm(blm.copy())

# South pole instrument
b2 = ScanStrategy(20*60*60, # 10*60 min mission
                  10, # 10 Hz sample rate
                  location='spole')

#b2 = ScanStrategy(10*60, # 10*60 sec mission
#                  10, # 10 Hz sample rate
#                  location='spole')


# calculate spinmaps 
b2.get_spinmaps(alm, blm, 5)

# Initiate a single detector
b2.set_focal_plane(1, 10)
az_off = b2.chn_pr_az
el_off = b2.chn_pr_el

b2.set_instr_rot(period=7*60) # Rotate instrument every 7 min

chunks = b2.partition_mission(10*60*60*int(b2.fsamp)) # calculate tod in chunks of # samples
print chunks
for chunk in chunks:
    print chunk
    b2.constant_el_scan(-10, -57.5, 50, 1, **chunk) # this sets the boresight
    for subchunk in b2.subpart_chunk(chunk):
        print subchunk
        b2.scan(az_off=az_off, el_off=el_off, **subchunk)

    


        print b2.sim_tod.size
#b2.init_point(q_bore=self.q_bore
#print b2.depo
#for cidx in xrange(b2.ndet):
#    blm, blmm2 = b2.get_blm(lmax, fwhm=fwhm)
#b2.tod2map
#print b2.depo
#print b2.init_point()
    
