'''
This script takes a asymmetric beam, Planck alms and calculates
tods using healpy. Compared to the so3 (or ssht) implementation,
where a single complex inverse transform is needed per s, here
we have to use two real transforms that both work on s <= 0 
since healpy does not allow for complex signals directly.
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import spider_analysis as sa
import spider_analysis.map as sam
import healpy as hp
import os
import sys
sys.path.insert(0, './../../src/python')
import so3_tools as so3t

plt.style.use('ggplot')
matplotlib.rcParams['agg.path.chunksize'] = 10000

ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171013_asym_centroid/'

fwhm = 300
lmax = 100
mmax = 100
nside = 256

# Specify detector offset
az_off = 15
el_off = -7

# Create Gaussian beam
blm, blmm2 = so3t.gauss_blm(fwhm, lmax, pol=True)
blmE = -blmm2 / 2.
blmB = -1j * blmm2 / 2.

# Get Planck alms
alm = sam.get_planck_alm(100, lmax=2000, coord='C', pol=True)
alm = (alm[0] * 0., alm[1] * 1., alm[2] * 1.)

# Deconvolve Planck beam
bell = sam.hfi_beam(100)
alm = sam.smoothalm(alm, beam=1/bell[0:2000+1], inplace=True)
alm = so3t.trunc_alm(alm, lmax)
fl_map = hp.alm2map(alm, nside)

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

# Only consider a small part of the first HWP event
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

    # Get spider_tools version of tod
    chunk_len = M.get_sample_count(spf=20, start=sopt['start'],
                                   end=sopt['end'], index_mode='sync')
    test_tod[c_idx:chunk_len+c_idx] = M.to_tod(chan,
                                               **sopt)[0] # (nchan, nsamp)

    # Save boresight pointing
    q_bore = M.depo['q_bore']
    ra_bore, dec_bore, pa_bore = M.quat2radecpa(q_bore)

    ra_ff[c_idx:chunk_len+c_idx] = ra_bore
    dec_ff[c_idx:chunk_len+c_idx] = dec_bore
    pa_ff[c_idx:chunk_len+c_idx] = pa_bore

    hwpang = M.get_hwpang(channels=chan)
    polang =  M.get_sim_polang(channels=chan)

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

hp.rotate_alm([blm, blmE, blmB], pa_off, (radius), -angle, lmax=lmax,  mmax=lmax)

def radec2ind_hp(ra, dec, nside):
    '''
    Turn qpoint ra and dec output into healpix
    ring map indices. Note, modifies ra and dec
    currently in-place.
    '''


    # Get indices
    ra *= (np.pi / 180.)
    ra = np.mod(ra, 2 * np.pi, out=ra)

    # convert from latitude to colatitude
    dec *= (np.pi / 180.)
    dec *= -1.
    dec += np.pi / 2.
    dec = np.mod(dec, np.pi, out=dec)

    pix = hp.ang2pix(nside, dec, ra, nest=False)

    return pix

pix = radec2ind_hp(ra_ff, dec_ff, nside)
sim_tod = np.zeros(ra_ff.size, dtype='float64')
sim_tod2 = np.zeros(ra_ff.size, dtype=np.complex128)

# Unpolarized sky and beam first
N = mmax + 1
func_r = np.zeros(12*nside**2, dtype='float64') # real sphere for s=0 
func = np.zeros((N, 12*nside**2), dtype=np.complex128) # s <=0 spin spheres

start = 0
for n in xrange(N): # note n is s
    end = lmax + 1 - n
    if n == 0: # scalar transform

        flmn = hp.almxfl(alm[0], blm[start:start+end], inplace=False)
#        func_r = hp.alm2map(flmn, nside)
#        func[n,:] = func_r

        func[n,:] += hp.alm2map(flmn, nside)
#        func[n,:] = func_r

    else: # spin transforms

        bell = np.zeros(lmax+1, dtype=np.complex128)
        # spin n beam
        bell[n:] = blm[start:start+end]

        flmn = hp.almxfl(alm[0], bell, inplace=False)
        flmmn = hp.almxfl(alm[0], np.conj(bell), inplace=False)

        flmnp = - (flmn + flmmn) / 2.
        flmnm = 1j * (flmn - flmmn) / 2.
        spinmaps = hp.alm2map_spin([flmnp, flmnm], nside, n, lmax, lmax)
        func[n,:] = spinmaps[0] + 1j * spinmaps[1]

    start += end

for n in xrange(N):

    if n == 0: #avoid expais since its one anyway
        sim_tod += np.real(func[n][pix])

    else:
#        exppais = np.exp(1j * n * np.radians(pa_ff))
#        sim_tod += np.real(func[n,:][pix] * exppais + \
#                               np.conj(func[n,:][pix] * exppais))

        # Cleaner way to do the above
        sim_tod += 2 * np.real(func[n,:][pix]) * np.cos(n * np.radians(pa_ff))
        sim_tod -= 2 * np.imag(func[n,:][pix]) * np.sin(n * np.radians(pa_ff))



# Pol
func = np.zeros((2*N-1, 12*nside**2), dtype=np.complex128) # all spin spheres
# convert E and B beams to spin pm2
blmp2 = -1 * (blmE + 1j * blmB)
blmm2 = -1 * (blmE - 1j * blmB)

almp2 = -1 * (alm[1] + 1j * alm[2])
almm2 = -1 * (alm[1] - 1j * alm[2])

start = 0
#for nidx, n in enumerate(xrange(-N+1, N)):
for n in xrange(N):
    end = lmax + 1 - np.abs(n)

    bellp2 = np.zeros(lmax+1, dtype=np.complex128)
    bellm2 = bellp2.copy()

    bellp2[np.abs(n):] = blmp2[start:start+end]
    bellm2[np.abs(n):] = blmm2[start:start+end]
    
    print n
    print bellm2
    print bellp2
    print np.array_equal(bellm2, np.conj(bellp2))

    s_flm_p = hp.almxfl(almp2, bellm2, inplace=False) + \
        hp.almxfl(almm2, np.conj(bellm2), inplace=False)
    s_flm_p /= -2.


    s_flm_m = hp.almxfl(almp2, bellm2, inplace=False) - \
        hp.almxfl(almm2, np.conj(bellm2), inplace=False)
    s_flm_m *= 1j / 2.

    if n == 0: # see https://healpix.jpl.nasa.gov/html/subroutinesnode12.htm
        spinmaps = [hp.alm2map(-s_flm_p, nside), 0] 

    else:
        spinmaps = hp.alm2map_spin([s_flm_p, s_flm_m], nside, n, lmax, lmax)

    func[N+n-1,:] = spinmaps[0] + 1j * spinmaps[1] # positive spin

    start += end

hwpang = sopt['hwpang'][M.fpu_index(chan, single=True)]


for nidx, n in enumerate(xrange(-N+1, N)):
    exppais = np.exp(1j * n * np.radians(pa_ff))
    sim_tod2 += func[nidx][pix] * exppais 

func *= 0
# Pol pt 2

start = 0
for n in xrange(N):
    end = lmax + 1 - np.abs(n)

    bellp2 = np.zeros(lmax+1, dtype=np.complex128)
    bellm2 = bellp2.copy()

    bellp2[np.abs(n):] = blmp2[start:start+end]
    bellm2[np.abs(n):] = blmm2[start:start+end]

    print n
    print bellm2
    print bellp2
    print np.array_equal(bellm2, np.conj(bellp2))

    s_flm_p = hp.almxfl(almm2, bellp2, inplace=False) + \
        hp.almxfl(almp2, np.conj(bellp2), inplace=False)
    s_flm_p /= -2.

    s_flm_m = hp.almxfl(almm2, bellp2, inplace=False) - \
        hp.almxfl(almp2, np.conj(bellp2), inplace=False)
    s_flm_m *= 1j / 2.

    if n == 0: # see https://healpix.jpl.nasa.gov/html/subroutinesnode12.htm
        spinmaps = [0, hp.alm2map(-s_flm_m, nside)] # works

    else:
        spinmaps = hp.alm2map_spin([s_flm_p, s_flm_m], nside, n, lmax, lmax)

    func[N-n-1,:] = spinmaps[0] - 1j * spinmaps[1] # negative spin

    start += end


hwpang = sopt['hwpang'][M.fpu_index(chan, single=True)]

for nidx, n in enumerate(xrange(-N+1, N)):
    exppais = np.exp(1j * n * np.radians(pa_ff))
    sim_tod2 += func[nidx][pix] * exppais 

expm2 = np.exp(1j * (4 * np.radians(hwpang) + 2 * np.radians(polang)))
sim_tod += np.real(sim_tod2 * expm2 + np.conj(sim_tod2 * expm2)) / 2.


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

ell = np.arange(lmax+1, dtype=float)
plt.figure()
plt.plot(ell, blm[0:lmax+1])
plt.savefig(ana_dir+'bell.png')
plt.close()




sys.exit()

