'''
This script takes a asymmetric beam, Planck alms and calculates
tods using healpy.
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import spider_analysis as sa
import spider_analysis.map as sam
import healpy as hp
import time
import os
import sys
sys.path.insert(0, './../../src/python')
import so3_tools as so3t

#plt.style.use('ggplot')
matplotlib.rcParams['agg.path.chunksize'] = 10000

ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171013_asym_centroid/'

fwhm = 300
lmax = 100
mmax = 100
nside = 256

az_off = 14
el_off = 3

# placeholder for quickbeam output
blm_jon = np.zeros((lmax + 1, lmax + 1), dtype=np.complex128)
blm_jon[:,0] = hp.sphtfunc.gauss_beam(np.radians(fwhm / 60.), lmax=lmax, pol=False)
ell = np.arange(lmax+1, dtype=float)

blm = np.zeros(hp.Alm.getsize(lmax, mmax=lmax), dtype=np.complex128)
blmm2 = blm.copy()
# convert to healpix format
lm = hp.Alm.getlm(lmax)
for idx in xrange(blm.size):
    blm[idx] = blm_jon[lm[0][idx], lm[1][idx]]
    try:
        blmm2[idx] = blm_jon[lm[0][idx], lm[1][idx] - 2] # NOTE THAT THIS IS DANGEROUS WITH ASYM BEAMS
    except IndexError:
        pass

# scalar E and B beam
blmE = -blmm2 / 2.
blmB = -1j * blmm2 / 2.

alm = sam.get_planck_alm(100, lmax=2000, coord='C', pol=True)
alm = (alm[0] * 1., alm[1] * 0., alm[2] * 0.)
bell = sam.hfi_beam(100)
# Deconvolve Planck beam
alm = sam.smoothalm(alm, beam=1/bell[0:2000+1], inplace=True)
alm = so3t.trunc_alm(alm, lmax)
fl_map = hp.alm2map(alm, nside)

# Convert the E and B alms to spin \pm 2 alms.
almp2 = -1 * (alm[1] + 1j * alm[2])
almm2 = -1 * (alm[1] - 1j * alm[2])

# smooth source map
fl_map_sm = sam.smoothing(fl_map, fwhm=np.radians(fwhm / 60.))

unimap_opts = dict(default_latest=True, source_map=fl_map_sm,
                   offsets='centroids_deproject02', cal='cal',
                   net='net',
                   pol=True, coord='C', source_units='K',
                   source_coord='C', source_pol=True,
                   polang='trpns_pol_angles_measured')
M = sa.UnifileMap(**unimap_opts)
sopts = M.hwp_partition(event='full_flight', index_mode='sync')
sopts[0].update({'end': 87344020, 'start': 87333300})
sopts = [sopts[0]]

# allocate tod and pointing arrays
tod_size = 0
for sopt in sopts:
    tod_size += M.get_sample_count(spf=20, start=sopt['start'],
                                   end=sopt['end'], index_mode='sync')

test_tod = np.ones(tod_size, dtype=float) * np.nan
ra_ff = test_tod.copy()
dec_ff = test_tod.copy()
pa_ff = test_tod.copy()

#chan = 'x6r09c15'
chan = 'x2r23c06'
cal = M.get_cal(channels=chan)

t2 = time.time()

# Get pointing and spider_tools tod
c_idx = 0
for sidx, sopt in enumerate(sopts):

    sopt_nohwp = sopt.copy()
    sopt_nohwp.pop('hwpidx', None)
    sopt_nohwp.pop('hwpang', None)

    # Update centroids
    M.update_sim_offsets(sim_az=az_off, sim_el=el_off, channels=chan)
    M.update_hwp(hwpang=sopt['hwpang'],
                 hwpidx=sopt['hwpidx'])

    # disable hwp_mueller stuff
    M.update_sim_hwp_mueller(channels=chan) # not important
    M.default_sim_hwp_mueller()
    M.default_hwp_mueller() # not important

    chunk_len = M.get_sample_count(spf=20, start=sopt['start'],
                                   end=sopt['end'], index_mode='sync')
    test_tod[c_idx:chunk_len+c_idx] = M.to_tod(chan,
                                               **sopt)[0] # (nchan, nsamp)

    # Save pointing
    q_bore = M.depo['q_bore']
    ra_bore, dec_bore, pa_bore = M.quat2radecpa(q_bore)

    hwpang = M.get_hwpang(channels=chan)
    polang =  M.get_sim_polang(channels=chan)

    ra_ff[c_idx:chunk_len+c_idx] = ra_bore
    dec_ff[c_idx:chunk_len+c_idx] = dec_bore
    pa_ff[c_idx:chunk_len+c_idx] = pa_bore

    c_idx += chunk_len

test_tod *= cal / 1e6 # Now also in K

# Rotate blm to match centroid
radius = np.arccos(np.cos(np.radians(el_off)) * np.cos(np.radians(az_off)))
if np.tan(radius) != 0:
    angle = np.arctan2(np.tan(np.radians(el_off)), np.sin(np.radians(az_off))) + np.pi/2.
else:
    angle = 0.

# compensate for rotation along phi (angle)
pa_off = angle # option 3
print pa_off
hp.rotate_alm([blm, blmE, blmB], pa_off, (radius), -angle, lmax=lmax,  mmax=lmax)

def radec2ind_hp(ra, dec, nside):

    # Get indices
    ra *= (np.pi / 180.)
#    ra *= -1.
#    ra += 2 * np.pi
    ra = np.mod(ra, 2 * np.pi, out=ra)

    # convert from latitude to colatitude
    dec *= (np.pi / 180.)
    dec *= -1.
    dec += np.pi / 2.
    dec = np.mod(dec, np.pi, out=dec)

    pix = hp.ang2pix(nside, dec, ra)

    return pix

pix = radec2ind_hp(ra_ff, dec_ff, nside)
sim_tod = np.zeros(ra_ff.size, dtype='float64')
sim_tod2 = np.zeros(ra_ff.size, dtype=np.complex128)

# first T
N = mmax + 1
t1 = time.time()
#func_r = np.zeros(L*(2*L-1), dtype='float64')
func_r = np.zeros(12*nside**2, dtype='float64')
func = np.zeros((2*N-1, 12*nside**2), dtype=np.complex128) # all spin spheres

start = 0 
for n in xrange(N):
    end = lmax + 1 - n
    if n == 0: # scalar transform
        flmn = hp.almxfl(alm[0], np.conj(blm[start:start+end]), inplace=False)
        func_r = hp.alm2map(flmn, nside)
        func[N+n-1,:] = func_r

    else: # These are only n > 0 (since alm * conj(bls) is real)

        bell = np.zeros(lmax+1, dtype=np.complex128)
#        bell[n:] = np.conj(blm[start:start+end])
        
        # spin n beam
        bell[n:] =blm[start:start+end]

        flmn = hp.almxfl(alm[0], bell, inplace=False)
        flmmn = hp.almxfl(alm[0], np.conj(bell), inplace=False)

        flmnp = - (flmn + flmmn) / 2.
        flmnm = 1j * (flmn - flmmn) / 2.
        spinmaps = hp.alm2map_spin([flmnp, flmnm], nside, n, lmax, lmax)
        func[N+n-1, :] = spinmaps[0] + 1j * spinmaps[1]


    start += end

    

#sim_tod += func_r[indices]
for n in xrange(N):
#    if n == 0:
#        continue
    if n == 0: #avoid expais since its 1
        sim_tod += np.real(func[N+n-1][pix])

    else:
        exppais = np.exp(1j * n * np.radians(pa_ff))
        sim_tod += np.real(func[N+n-1,:][pix] * exppais + \
                               np.conj(func[N+n-1,:][pix] * exppais))

# Pol
start = 0 
for n in xrange(N):
    end = lmax + 1 - n

    if n == 0: # scalar transform
        flmn = hp.almxfl(alm[0], np.conj(blm[start:start+end]), inplace=False)
        func_r = hp.alm2map(flmn, nside)
        func[N+n-1,:] = func_r

    else: # These are only n > 0 (since alm * conj(bls) is real)

        bell = np.zeros(lmax+1, dtype=np.complex128)
#        bell[n:] = np.conj(blm[start:start+end])
        
        # spin n beam
        bell[n:] =blm[start:start+end]

        flmn = hp.almxfl(alm[0], bell, inplace=False)
        flmmn = hp.almxfl(alm[0], np.conj(bell), inplace=False)

        flmnp = - (flmn + flmmn) / 2.
        flmnm = 1j * (flmn - flmmn) / 2.
        spinmaps = hp.alm2map_spin([flmnp, flmnm], nside, n, lmax, lmax)
        func[N+n-1, :] = spinmaps[0] + 1j * spinmaps[1]


    start += end



samples = np.arange(pa_ff.size)
fig, ax = plt.subplots(2, 1, sharex=True,
                       gridspec_kw={'height_ratios':[2, 1]})
ax[0].plot(samples, sim_tod / cal * 1e6, label='asym. beam',
         alpha=0.6)
ax[0].plot(samples, test_tod / cal * 1e6, label='M.to_tod()',
         alpha=1, ls='--')
ax[1].plot(samples, (test_tod - sim_tod) / cal * 1e6, label='M.to_tod() - asym.beam',
         alpha=1)
ax[1].axhline(y=0., c='black', alpha=0.5)
ax[0].annotate(u'(az,el) = ({:.1f}\u00B0,{:.1f}\u00B0)'.format(az_off, el_off),
               xy=(0.025, 0.950),
               xycoords='axes fraction')
ax[0].annotate("fwhm  = {:d}'".format(fwhm), xy=(0.025, 0.900),
             xycoords='axes fraction')
ax[0].annotate("(L,N)   = ({:d},{:d})".format(lmax, mmax), xy=(0.025, 0.850),
             xycoords='axes fraction')
ax[0].annotate("nside  = {:d}".format(nside), xy=(0.025, 0.800),
             xycoords='axes fraction')

ax[1].annotate(u'mean   = {:.1E}'.format(np.mean((test_tod - sim_tod)) / cal * 1e6),
               xy=(0.025, 0.90),
               xycoords='axes fraction')

ax[1].set_xlabel('sample')
ax[0].set_ylabel('[ADU]')
ax[1].set_ylabel('Difference [ADU]')
ax[0].legend()
ax[1].legend()
fig.savefig(ana_dir+'tods_'+chan+'.png')
plt.close()


plt.figure()
plt.plot(ell, blm[0:lmax+1])
plt.savefig(ana_dir+'bell.png')
plt.close()
