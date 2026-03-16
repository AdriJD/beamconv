import numpy as np
import healpy as hp
from beamconv import ScanStrategy
from beamconv import Beam, tools
import ducc0

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res


def convolve(alm1, alm2, lmax, nthreads=1):
    # Go to a Gauss-Legendre grid of sufficient dimensions
    ntheta, nphi = lmax+1, ducc0.fft.good_size(2*lmax+1, True)
    tmap = ducc0.sht.experimental.synthesis_2d(
        alm=alm1.reshape((1,-1)), ntheta=ntheta, nphi=nphi, lmax=lmax,
        geometry="GL", spin=0, nthreads=nthreads)
    tmap *= ducc0.sht.experimental.synthesis_2d(
        alm=alm2.reshape((1,-1)), ntheta=ntheta, nphi=nphi, lmax=lmax,
        geometry="GL", spin=0, nthreads=nthreads)
    # compute the integral over the resulting map
    res = ducc0.sht.experimental.analysis_2d(
        map=tmap, lmax=0, spin=0, geometry="GL", nthreads=nthreads)
    return np.sqrt(4*np.pi)*res[0,0].real


# This does the following:
# - convert s_lm (TEBV) to a sufficiently high resolution IQUV map
# - apply M_alpha^T M_HWP M_alpha to the IQUV vector in each pixel
# - transform the resulting IQUV map back to s_lm (TEBV) and return this
def apply_mueller_to_sky(slm, lmax, m_hwp, alpha, nthreads=1):
    ncomp = slm.shape[0]
    
    # Go to a Gauss-Legendre grid of sufficient dimensions
    ntheta, nphi = lmax+1, ducc0.fft.good_size(2*lmax+1, True)
    skymap = np.zeros((ncomp, ntheta, nphi))
    skymap[0:1] = ducc0.sht.experimental.synthesis_2d(
        alm=slm[0:1], ntheta=ntheta, nphi=nphi, lmax=lmax,
        geometry="GL", spin=0, nthreads=nthreads)
    if ncomp >= 3:
        skymap[1:3] = ducc0.sht.experimental.synthesis_2d(
            alm=slm[1:3], ntheta=ntheta, nphi=nphi, lmax=lmax,
            geometry="GL", spin=2, nthreads=nthreads)
    if ncomp == 4:
       skymap[3:4] = ducc0.sht.experimental.synthesis_2d(
           alm=slm[3:4], ntheta=ntheta, nphi=nphi, lmax=lmax,
           geometry="GL", spin=0, nthreads=nthreads)

    # apply Mueller matrix to sky
    m_alpha = np.zeros((4,4))
    m_alpha[0,0] = m_alpha[3,3] = 1
    m_alpha[1,1] = m_alpha[2,2] = np.cos(2*alpha)
    m_alpha[1,2] = np.sin(2*alpha)
    m_alpha[2,1] = -np.sin(2*alpha)
    m_hwp_alpha = m_alpha.T.dot(m_hwp.dot(m_alpha))
    T = np.zeros((4,4),dtype=np.complex128)
    T[0,0] = T[3,3] = 1.
    T[1,1] = T[2,1] = 1./np.sqrt(2.)
    T[1,2] = 1j/np.sqrt(2.)
    T[2,2] = -1j/np.sqrt(2.)
    C = T.dot(m_hwp.dot(np.conj(T.T)))
    X = T.dot(m_alpha.dot(np.conj(T.T)))
    fullmat = np.conj(T.T).dot(np.conj(X.T).dot(C.dot(X.dot(T))))
    # there must be a better way to do this ...
    for i in range(ntheta):
        for j in range(nphi):
            tmp = fullmat.dot(skymap[:,i,j])
            skymap[:, i, j] = tmp.real
    # go back to spherical harmonics
    res = np.empty_like(slm)
    res[0:1] = ducc0.sht.experimental.analysis_2d(
        map=skymap[0:1], lmax=lmax, spin=0, geometry="GL", nthreads=nthreads)
    if ncomp >= 3:
        res[1:3] = ducc0.sht.experimental.analysis_2d(
            map=skymap[1:3], lmax=lmax, spin=2, geometry="GL", nthreads=nthreads)
    if ncomp == 4:
        res[3:4] = ducc0.sht.experimental.analysis_2d(
            map=skymap[3:4], lmax=lmax, spin=0, geometry="GL", nthreads=nthreads)
    return res


def explicit_convolution(slm, blm, lmax, theta, phi, psi, alpha, m_hwp, nthreads=1):
    # extend blm from (lmax, mmax) to full (lmax, lmax) size for rotation
    blm2 = np.zeros((blm.shape[0], nalm(lmax, lmax)), dtype=np.complex128)
    blm2[:, 0:blm.shape[1]] = blm
    # rotate beam to desired orientation
    for i in range(blm2.shape[0]):
        blm2[i] = ducc0.sht.rotate_alm(blm2[i], lmax, psi, theta, phi, nthreads)
    # apply Mueller matrix to sky
    slm2 = apply_mueller_to_sky(slm, lmax, m_hwp, alpha, nthreads)

    # convolve sky and beam component-wise and return the sum
    res = 0.
    for i in range(blm2.shape[0]):
        res += convolve(slm2[i], blm2[i], lmax, nthreads)
    return res


def test_basic_convolution():
    rng = np.random.default_rng(41)
    lmax = 10

    slm_in = random_alm(lmax, lmax, 0, 4, rng)
    slm = (slm_in[0], slm_in[1], slm_in[2], slm_in[3])

    # create a scanning strategy (we only care for the very first sample though)
    mlen = 10 * 60
    rot_period = 120
    mmax = 2
    ra0=-10
    dec0=-57.5
    fwhm = 20
    nside = 128
    az_throw = 10
    polang = 20.

    ces_opts = dict(ra0=ra0, dec0=dec0, az_throw=az_throw,
                    scan_speed=2.)

    scs = ScanStrategy(duration=mlen, sample_rate=10, location='spole')

    scs.create_focal_plane(nrow=1, ncol=1, fov=0,
                           lmax=lmax, fwhm=fwhm,
                           polang=polang)
    beam = scs.beams[0][0]
    hwp_mueller = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    beam.hwp_mueller = hwp_mueller
    scs.init_detpair(slm, beam, nside_spin=nside,
                               max_spin=mmax)
    scs.partition_mission()

    chunk = scs.chunks[0]
    ces_opts.update(chunk)

    # Populate boresight.
    scs.constant_el_scan(**ces_opts)

    # Turn on HWP
    scs.set_hwp_mod(mode='continuous', freq=1., start_ang=0)
    scs.rotate_hwp(**chunk)

    # scan using beamconv
    tod, pix, nside_out, pa, hwp_ang = scs.scan(beam,
                    return_tod=True, return_point=True, interp=False, **chunk)


    # verify the first returned TOD value by brute force convolution
    theta, phi = hp.pixelfunc.pix2ang(nside_out, pix[0], nest=False)
    psi = np.radians(pa[0])
    alpha = np.radians(hwp_ang[0])

    slm = slm_in.copy()
    # Adjust blm to ducc/healpy conventions
    blm = beam.blm
    blm_ex = np.empty((4, len(blm[0])), dtype=np.complex128)
    blm_ex[0] = blm[0]
    blm_ex[1], blm_ex[2] = tools.spin2eb(blm[1], blm[2])
    blm_ex[3] = blm[2]*0
    lfac = np.sqrt((1.+2*np.arange(lmax+1.))/(4*np.pi))
    ofs=0
    for m in range(lmax+1):
        blm_ex[:, ofs:ofs+lmax+1-m] *= lfac[m:].reshape((1,-1))
        ofs += lmax+1-m

    res = explicit_convolution(slm, blm_ex, lmax, theta, phi, psi, alpha, hwp_mueller, nthreads=1)

    print(tod[0],res, tod[0]/res - 1)
    np.testing.assert_allclose(tod[0],res)


if __name__ == "__main__":
    test_basic_convolution()
