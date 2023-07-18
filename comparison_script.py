import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import mueller_convolver
import ducc0

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)

def make_full_random_alm(lmax, mmax, rng):
    res = rng.uniform(-1., 1., (4, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (4, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    # components 1 and 2 are spin-2, fix them accordingly
    spin=2
    for s in range(spin):
        res[1:3, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res

# code by Marta to get beamconv results for user-specified angles
def get_beamconv_values(lmax, kmax, slm, blm, ptg, hwp_angles, mueller):
    import beamconv
    import qpoint as qp

    # prepare PO beam file
    blm2 = np.zeros((blm.shape[0], hp.Alm.getsize(lmax=lmax)), dtype=np.complex128)
    blm2[:,:blm.shape[1]] = blm
    blmm, blmp = beamconv.tools.eb2spin(blm2[1],blm2[2])
    blm2[1] = blmm
    blm2[2] = blmp
    np.save("temp_beam.npy", blm2)

    # set up beam and HWP mueller matrix (identity, i.e. no HWP)
    beam = beamconv.Beam(btype='PO', lmax=lmax, mmax=lmax, deconv_q=True, normalize=False, po_file="temp_beam.npy", hwp_mueller=mueller)

    nsamp = ptg.shape[0]

    # from (theta,phi) to (ra,dec) convention
    # also, all angles are converted in degrees
    ra = np.degrees(ptg[:,1])
    dec = 90. - np.degrees(ptg[:,0])
    # Adjustment for difference in convention between qpoint and MuellerConvolver?
    psi = 180. - np.degrees(ptg[:,2])

    # calculate the quaternion
    q_bore_array = qp.QPoint().radecpa2quat(ra, dec, psi)

    def ctime_test(**kwargs):
        return np.zeros(kwargs.pop('end')-kwargs.pop('start'))
    
    def q_bore_test(**kwargs):
        return q_bore_array[kwargs.pop('start'):kwargs.pop('end')]

    S = beamconv.ScanStrategy(duration=nsamp, sample_rate=1, external_pointing=True, use_l2_scan=False)
    S.add_to_focal_plane(beam, combine=False)
    S.set_hwp_mod(mode='stepped', freq=1, angles=hwp_angles*180/np.pi)

    # determine nside_spin necessary for good accuracy
    nside_spin = 1
    while nside_spin < 4*lmax:
        nside_spin *= 2

    S.scan_instrument_mpi(slm.copy(), save_tod=True, ctime_func=ctime_test, q_bore_func=q_bore_test,
                      ctime_kwargs={'useless':0}, q_bore_kwargs={'useless':0},nside_spin=nside_spin, interp=True, input_v=True, beam_v=True, max_spin=kmax+4, binning=False, verbose=0)

    return S.data(S.chunks[0], beam=beam, data_type='tod').copy()


np.random.seed(10)
rng = np.random.default_rng(np.random.SeedSequence(42))
lmax = 30
kmax = 18

# completely random sky
slm =make_full_random_alm(lmax, lmax, rng)

# completely random Mueller matrix
mueller = np.random.uniform(-1,1,size=(4,4))
#mueller[1:3,0]=mueller[1:3,-1] = 0
#mueller[0,2]=mueller[2,0] = 0

# completely random beam
blm = make_full_random_alm(lmax, kmax, rng)

nptg=100
# completely random pointings
ptg = np.empty((nptg,3))
ptg[:,0]=np.random.uniform(0,np.pi,size=(nptg,))    # theta
ptg[:,1]=np.random.uniform(0,2*np.pi,size=(nptg,))  # phi
ptg[:,2]=np.random.uniform(0,2*np.pi,size=(nptg,))  # psi
hwp_angles = np.random.uniform(0,2*np.pi,size=(nptg,))  # alpha

# get the signal from beamconv
signal_beamconv = get_beamconv_values(lmax=lmax, kmax=kmax, slm=slm, blm=blm, ptg=ptg, hwp_angles=hwp_angles, mueller=mueller)

# Now do the same thing with MuellerConvolver
fullconv = mueller_convolver.MuellerConvolver(
    lmax=lmax,
    kmax=kmax,
    slm=slm,
    blm=blm,
    mueller=mueller,
    single_precision=False,
    epsilon=1e-7,
    nthreads=1,
)
signal_muellerconvolver = fullconv.signal(ptg=ptg, alpha=hwp_angles)

# L2 error
print("L2 error between results:", ducc0.misc.l2error(signal_beamconv, signal_muellerconvolver))
plt.plot(signal_beamconv)
plt.plot(signal_muellerconvolver)
plt.show()
