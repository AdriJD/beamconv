import ducc0.totalconvolve
import numpy as np


def _convert_blm(blm_in, lmax, kmax):
    nblm = ((kmax+1)*(kmax+2))//2 + (kmax+1)*(lmax-kmax)
    blm = np.array([x[:nblm].copy() for x in blm_in])
    lfac = np.sqrt((1.+2*np.arange(lmax+1.))/(4*np.pi))
    ofs=0
    for m in range(kmax+1):
       blm[:, ofs:ofs+lmax+1-m] *= lfac[m:].reshape((1,-1))
       ofs += lmax+1-m
    return blm
def _convert_blm2(blm_in, lmax, kmax):
    nblm = ((kmax+1)*(kmax+2))//2 + (kmax+1)*(lmax-kmax)
    blm = blm_in.copy()
    lfac = np.sqrt((1.+2*np.arange(lmax+1.))/(4*np.pi))
    ofs=0
    for m in range(kmax+1):
       blm[:, ofs:ofs+lmax+1-m] = (-1)**m * np.conj(blm[:, ofs:ofs+lmax+1-m])
       ofs += lmax+1-m
    blm= blm[-1::-1]
    return blm


class Interpolator_real:
    def __init__(self, slm, blm, lmax, kmax, epsilon=1e-11, nthreads=1):
        _slm = np.array([x.copy() for x in slm])
        _blm = _convert_blm(blm, lmax, kmax)
        self._inter = ducc0.totalconvolve.Interpolator(
            _slm, _blm, False, lmax, kmax, epsilon, 2., nthreads)

    def interpol(self, theta, phi, psi):
        ptg = np.zeros((theta.shape[0], 3))
        ptg[:, 0] = theta
        ptg[:, 1] = phi
        ptg[:, 2] = psi
        return self._inter.interpol(ptg)[0]


class Interpolator_complex:
    def __init__(self, slmE, slmB, blmE, blmB, lmax, kmax, epsilon=1e-11, nthreads=1):
        _slm = np.array([slmE, slmB])
        _blm = _convert_blm([blmE, blmB], lmax, kmax)
        self._inter = ducc0.totalconvolve.Interpolator(
            _slm, _blm, False, lmax, kmax, epsilon, 2., nthreads)
        _blm2 = _convert_blm2(_blm, lmax, kmax)
        self._inter2 = ducc0.totalconvolve.Interpolator(
            _slm, _blm2, False, lmax, kmax, epsilon, 2., nthreads)

    def interpol(self, theta, phi, psi):
        ptg = np.zeros((theta.shape[0], 3))
        ptg[:, 0] = theta
        ptg[:, 1] = phi
        ptg[:, 2] = psi
        res = self._inter.interpol(ptg)
        res2 = self._inter2.interpol(ptg)
        return res[0]+1j*res2[0]
