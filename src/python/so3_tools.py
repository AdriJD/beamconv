'''
A collection of functions used to interface between the so3 code and 
healpy/spider_tools.
'''
import numpy as np
import so3_wrapper as so3lib
import healpy as hp

def trunc_alm(alm, lmax_new, mmax_old=None):
    '''
    Truncate a (sequence of) healpix-formatted alm array(s)
    from lmax to lmax_new. If sequence: all components must
    share lmax and mmax. No in-place modifications performed.

    Arguments
    ---------
    alm : array-like
        healpy alm array or sequence of alm arrays that
        share lmax and mmax.
    lmax_new : int
        The new bandlmit.
    mmax_old : int
        m-bandlimit alm, optional.

    Returns
    -------
    alm_new: array with same type and dimensionality as
        alm, but now only contains modes up to lmax_new.
        If original dimension > 1, return tuple with
        truncated components.
    '''

    if hp.cookbook.is_seq_of_seq(alm):
        lmax = hp.Alm.getlmax(alm[0].size, mmax=mmax_old)
        seq = True

    else:
        lmax = hp.Alm.getlmax(alm.size, mmax=mmax_old)
        seq = False

    if mmax_old is None:
        mmax_old = lmax

    if lmax < lmax_new:
        raise ValueError('lmax_new should be smaller than old lmax')

    indices = np.zeros(hp.Alm.getsize(
                       lmax_new, mmax=min(lmax_new, mmax_old)),
                       dtype=int)
    start = 0
    nstart = 0

    for m in xrange(min(lmax_new, mmax_old) + 1):
        indices[nstart:nstart+lmax_new+1-m] = \
            np.arange(start, start+lmax_new+1-m)
        start += lmax + 1 - m
        nstart += lmax_new + 1 - m

    # Fancy indexing so numpy makes copy of alm
    if seq:
        return tuple([alm[d][indices] for d in xrange(len(alm))])

    else:
        alm_new = alm[indices]
        return alm_new

def gauss_blm(fwhm, lmax, pol=False):
    '''
    Generate a healpix-shaped blm-array for an
    azimuthally-symmetric Gaussian beam 
    normalized at (ell,m) = (0,0).

    Arguments
    ---------
    fwhm : int
        FWHM specifying Gaussian (arcmin)
    lmax : int
        Band-limit specifying array layout
    pol : bool, optional
        Also return spin -2 blm that represents
        the spin -2 part of the copolarized pol
        beam.
        
    Returns
    -------
    blm : array-like
       Complex numpy array with bell in the
       m=0 slice.

    or
    
    blm, blmm2 : list of array-like
       Complex numpy array with bell in the
       m=0 and m=2 slice.
    '''

    blm = np.zeros(hp.Alm.getsize(lmax, mmax=lmax), 
                   dtype=np.complex128)
    if pol:
        blmm2 = blm.copy()

    bell = hp.sphtfunc.gauss_beam(np.radians(fwhm / 60.),
                                  lmax=lmax, pol=False)

    blm[:lmax+1] = bell

    if pol:
        blmm2[2*lmax+1:3*lmax] = bell[2:]
        return blm, blmm2
    else:
        return blm


def alm2flmn(alm, parameters):
    '''
    Convert healpix-formatted alm array(s) to so3
    flmn. No assymetric beams yet, but can do pol.

    Arguments
    ----------
    alm: array, tuple of arrays.

    Returns
    -------
    flmn
    '''

    flmn_size = so3lib.flmn_size(parameters)
    flmn = np.zeros(flmn_size, dtype=np.complex128)
    lmax = hp.Alm.getlmax(alm.size)
    no_pol = True

    if isinstance(alm, tuple):
        if len(alm) == 3:
            for i in xrange(3):
                assert isinstance(alm[i], np.ndarray), 'alm should be array'

            # Assume I, E and B.
            no_pol = False

            # Convert E and B alm to spin \pm 2 alm.
            almp2 = -1 * (alm[1] + 1j * alm[2]) / 2
            almm2 = -1 * (alm[1] - 1j * alm[2]) / 2
        else:
            print 'Alm tuple shape not supported yet.'

    assert isinstance(alm, np.ndarray), 'alm should be array'
    
    for ind in xrange(flmn_size):
        el, m, n = so3lib.ind2elmn(ind, parameters)
        c_conj = False
        if no_pol and n != 0:
            break
        elif n not in (0, -2, 2):
            break
        
        ## NOTE that once you start with asymmtric beams you should
        ## keep the n != 0,-2,2 indices not zero but constant!

        if n == 0:        
            if m > 0:
                m = -m
                c_conj = True
            n_hp = hp.Alm.getidx(lmax, el, -m)
            if c_conj:
                alm_p = np.conj(alm[n_hp]) * (-1)**m
            else:
                alm_p = alm[n_hp]
                # See paper for (8 * np.pi**2) / (2*el + 1) factor.
            flmn[ind] = alm_p * (-1)**m * np.sqrt((2 * el + 1) / 4. / np.pi) \
                * (8 * np.pi**2) / (2 * el + 1)

        elif n == 2:
            if m > 0:
                m = -m
                c_conj = True
            n_hp = hp.Alm.getidx(lmax, el, -m)
            if c_conj:
                alm_p = np.conj(almm2[n_hp]) * (-1)**m
            else:
                alm_p = almp2[n_hp]

        elif n == -2:
            if m > 0:
                m = -m
                c_conj = True
            n_hp = hp.Alm.getidx(lmax, el, -m)
            if c_conj:
                alm_p = np.conj(almp2[n_hp]) * (-1)**m
            else:
                alm_p = almm2[n_hp]

        flmn[ind] = alm_p * (-1)**m * np.sqrt((2 * el + 1) / 4. / np.pi) \
            * (8 * np.pi**2) / (2 * el + 1)

    return flmn

def get_alm_ind_real(lmax, parameters, ret_wigner_pref=False, ret_spin_pref=False):
    '''
    Get index array: alm_ind and two prefactor arrays such that
    flmn = alm[alm_ind] * pref + np.conj(alm)[alm_ind] * pref_c.
    Here "alm" must correspond to a real-valued field
    on the sphere in healpix format and flmn is so3-format.
    Can also return a multiplicative factor to convert
    spehrical harmonic coefficients to wigner coefficients.

    Arguments
    ----------
    lmax: int
    parameters: so3 parameters struct.

    Returns
    -------
    ind: array-like
        Healpix alm indices array of shape (flmn_size,)
    ind_c: array-like
        Healpix conj(alm) indices array of shape (flmn_size,)
    pref: array-like
        Multiplicative factor array of shape (flmn_size,)
    pref_c: array-like
        Multiplicative factor array of shape (flmn_size,)
    '''

    if not parameters.reality:
        raise ValueError("so3 reality parameter must be 1")

    if ret_wigner_pref and ret_spin_pref:
        raise ValueError("Cannot have both prefactors")

    flmn_size = so3lib.flmn_size(parameters)
    pref = np.zeros(flmn_size, dtype='int8')
    pref_c = pref.copy()
    alm_ind = np.ones(flmn_size, dtype='uint32')
    if ret_wigner_pref or ret_spin_pref:
        w_pref = np.zeros(flmn_size, dtype=float)

    # local variables might be faster
    ind2elmn_real = so3lib.ind2elmn_real
    getidx = hp.Alm.getidx

    for ind in xrange(flmn_size):
        el, m, n = ind2elmn_real(ind, parameters)
        c_conj = False

        if m > 0:
            m = -m
            c_conj = True

        n_hp = getidx(lmax, el, -m)
        alm_ind[ind] = n_hp

        if c_conj:
            pref_c[ind] = (-1)**m
        else:
            pref[ind] = 1

        if ret_wigner_pref:
            # See paper for (8 * np.pi**2) / (2*el + 1) factor.
            w_pref[ind] = (-1)**m * np.sqrt((2 * el + 1) / 4. / np.pi) \
                * (8 * np.pi**2) / (2 * el + 1)

        if ret_spin_pref:
            w_pref[ind] = (-1)**(m + n)

    if ret_wigner_pref or ret_spin_pref:
        return alm_ind, pref, pref_c, w_pref

    return alm_ind, pref, pref_c

def get_alm_ind_complex(lmax, parameters, ret_wigner_pref=False, ret_spin_pref=False):
    '''
    Get index array: alm_ind and four prefactor arrays such that
    almn = almp2[alm_ind] * pref_p2 + np.conj(almm2)[alm_ind] * pref_c_m2
    + almm2[alm_ind] * a_pref_m2 + np.conj(almp2)[alm_ind] * a_pref_c_p2
    Here "almp2" and almm2 correspond to two complex fields p and m
    on the sphere where conj(m) = p (in healpix format).
    flmn is so3-format. Can also return a multiplicative factor to convert
    spehrical harmonic coefficients to wigner coefficients.

    Arguments
    ----------
    lmax: int
    parameters: so3 parameters struct.

    Returns
    -------
    ind: array-like
        Healpix alm indices array of shape (flmn_size,)
    ind_c: array-like
        Healpix conj(alm) indices array of shape (flmn_size,)
    pref: array-like
        Multiplicative factor array of shape (flmn_size,)
    pref_c: array-like
        Multiplicative factor array of shape (flmn_size,)
    '''

    if parameters.reality:
        raise ValueError("so3 reality parameter must be 0")

    if ret_wigner_pref and ret_spin_pref:
        raise ValueError("Cannot have both prefactors")

    flmn_size = so3lib.flmn_size(parameters)
    pref_p2 = np.zeros(flmn_size, dtype='int8')
    pref_c_m2 = pref_p2.copy()
    pref_m2 = pref_p2.copy()
    pref_c_p2 = pref_p2.copy()
    alm_ind = np.ones(flmn_size, dtype=int)
    if ret_wigner_pref or ret_spin_pref:
        w_pref = np.zeros(flmn_size, dtype=float)


    ind2elmn = so3lib.ind2elmn
    getidx = hp.Alm.getidx

    for ind in xrange(flmn_size):
        el, m, n = ind2elmn(ind, parameters)
        c_conj = False

        if m > 0:
            m = -m
            c_conj = True
            
        n_hp = getidx(lmax, el, -m)
        alm_ind[ind] = n_hp

        if c_conj:
            pref_c_m2[ind] = (-1)**m
        else:
            pref_p2[ind] = 1

        if c_conj:
            pref_c_p2[ind] = (-1)**m
        else:
            pref_m2[ind] = 1

        if ret_wigner_pref:
            # See so3 paper for (8 * np.pi**2) / (2*el + 1) factor.
            w_pref[ind] = (-1)**m * np.sqrt((2 * el + 1) / 4. / np.pi) \
                * (8 * np.pi**2) / (2 * el + 1)

        if ret_spin_pref:
            w_pref[ind] = (-1)**(m + n)

    if ret_wigner_pref or ret_spin_pref:
        return alm_ind, pref_p2, pref_c_m2, pref_m2, pref_c_p2, w_pref

    return alm_ind, pref_p2, pref_c_m2, pref_m2, pref_c_p2

def get_blm_ind_complex(lmax, parameters):

    if parameters.reality:
        raise ValueError("so3 reality parameter must be 0")

    flmn_size = so3lib.flmn_size(parameters)
    pref_p2 = np.zeros(flmn_size, dtype='int8')
    pref_c_m2 = pref_p2.copy()
    pref_m2 = pref_p2.copy()
    pref_c_p2 = pref_p2.copy()
    blm_ind = np.ones(flmn_size, dtype=int)

    ind2elmn = so3lib.ind2elmn
    getidx = hp.Alm.getidx

    for ind in xrange(flmn_size):
        el, m, n = ind2elmn(ind, parameters)
        c_conj = False

        if n < 0:
            n = -n
            c_conj = True
            
        n_hp = getidx(lmax, el, n)
        blm_ind[ind] = n_hp

        if c_conj:
            pref_c_m2[ind] = (-1)**n
        else:
            pref_p2[ind] = 1

        if c_conj:
            pref_c_p2[ind] = (-1)**n
        else:
            pref_m2[ind] = 1

    return blm_ind, pref_p2, pref_c_m2, pref_m2, pref_c_p2


def get_blm_ind_real(lmax, parameters, ret_wigner_pref=False):
    '''
    Get index array: blm_ind such that blmn = blm[blm_ind].
    Here "blm" must correspond to a real-valued field
    on the sphere in healpix format and blmn is so3-format.
    Can also return a multiplicative factor to convert
    spherical harmonic coefficients to wigner coefficients.

    Arguments
    ----------
    lmax: int
    parameters: so3 parameters struct.

    Returns
    -------
    blm_ind: array-like
        Healpix alm indices array of shape (flmn_size,)
    pref_c: array-like
        Multiplicative factor array of shape (flmn_size,)
        if ret_wigner_pref is True.
    '''

    if not parameters.reality:
        raise ValueError("so3 reality parameter must be 1")

    flmn_size = so3lib.flmn_size(parameters)
    blm_ind = np.ones(flmn_size, dtype='uint32')
    if ret_wigner_pref:
        w_pref = np.zeros(flmn_size, dtype=float)

    ind2elmn_real = so3lib.ind2elmn_real
    getidx = hp.Alm.getidx

    for ind in xrange(flmn_size):
        el, m, n = ind2elmn_real(ind, parameters)

        n_hp = getidx(lmax, el, n)
        blm_ind[ind] = n_hp

        if ret_wigner_pref:
            # See so3 paper for (8 * np.pi**2) / (2*el + 1) factor.
            w_pref[ind] = (-1)**m * np.sqrt((2 * el + 1) / 4. / np.pi) \
                * (8 * np.pi**2) / (2 * el + 1)

    if ret_wigner_pref:
        return blm_ind, w_pref

    return blm_ind

def get_p2blm_ind_complex_copol(lmax, parameters, ret_wigner_pref=False):

    if parameters.reality:
        raise ValueError("so3 reality parameter must be 0")

    flmn_size = so3lib.flmn_size(parameters)
    pref_p2 = np.zeros(flmn_size, dtype='int8')
    pref_c_m2 = pref_p2.copy()
    blm_ind = np.ones(flmn_size, dtype=int)

    ind2elmn = so3lib.ind2elmn
    getidx = hp.Alm.getidx

    for ind in xrange(flmn_size):
        el, m, n = ind2elmn(ind, parameters)
        c_conj = False

        if n >= 2:
            n_p = n - 2

        if n < 2:
            n_p = -n + 2
            c_conj = True

        if np.abs(n_p) > el:
            continue

        n_hp = getidx(lmax, el, n_p)
        blm_ind[ind] = n_hp

        if c_conj:
            pref_c_m2[ind] = (-1)**n_p 
        else:
            pref_p2[ind] = 1

    return blm_ind, pref_p2, pref_c_m2


def get_m2blm_ind_complex_copol(lmax, parameters, ret_wigner_pref=False):

    if parameters.reality:
        raise ValueError("so3 reality parameter must be 0")

    flmn_size = so3lib.flmn_size(parameters)
    pref_m2 = np.zeros(flmn_size, dtype='int8')
    pref_c_p2 = pref_m2.copy()
    blm_ind = np.ones(flmn_size, dtype=int)

    ind2elmn = so3lib.ind2elmn
    getidx = hp.Alm.getidx

    for ind in xrange(flmn_size):
        el, m, n = ind2elmn(ind, parameters)
        c_conj = False

        if n >= -2:
            n_p = n + 2

        if n < -2:
            n_p = -n - 2
            c_conj = True

        if np.abs(n_p) > el:
            continue

        n_hp = getidx(lmax, el, n_p)
        blm_ind[ind] = n_hp

        if c_conj:
            pref_c_p2[ind] = (-1)**n_p
        else:
            pref_m2[ind] = 1

    return blm_ind, pref_m2, pref_c_p2


def get_blm_ind_complex_copol(lmax, parameters, ret_wigner_pref=False):
    '''
    Get index array: blm_ind and four prefactor arrays such that
    blmn = blm[blm_ind] * pref_p2 + blm[blm_ind] * pref_m2
    + np.conj(blm)[blm_ind] * pref_c_m2
    + np.conj(blm)[blm_ind] * pref_c_p2.
    Here "blm" must correspond to a real-valued field
    on the sphere in healpix format and flmn is so3-format.
    Can also return a multiplicative factor to convert
    spherical harmonic coefficients to wigner coefficients.

    Arguments
    ----------
    lmax: int
    parameters: so3 parameters struct.

    Returns
    -------
    blm_ind: array-like
        Healpix alm indices array of shape (flmn_size,)
    pref_p2: array-like
        Multiplicative factor array of shape (flmn_size,)
    pref_m2: array-like
        Multiplicative factor array of shape (flmn_size,)
    pref_c_m2: array-like
        Multiplicative factor array of shape (flmn_size,)
    pref_c_m2: array-like
        Multiplicative factor array of shape (flmn_size,)
    w_pref: array-like, optional
        Multiplicative factor array of shape (flmn_size,)
    '''

    if parameters.reality:
        raise ValueError("so3 reality parameter must be 0")

    flmn_size = so3lib.flmn_size(parameters)
    pref_p2 = np.zeros(flmn_size, dtype='int8')
    pref_c_m2 = pref_p2.copy()
    pref_m2 = pref_p2.copy()
    pref_c_p2 = pref_p2.copy()
    blm_ind = np.ones(flmn_size, dtype=int)
    if ret_wigner_pref:
        w_pref = np.zeros(flmn_size, dtype=float)

    ind2elmn = so3lib.ind2elmn
    getidx = hp.Alm.getidx

    for ind in xrange(flmn_size):
        el, m, n = ind2elmn(ind, parameters)
        c_conj = False
        normal = False
        if n >= 2:
            n_p = n - 2
            normal = True
        if n <= -2:
            n_p = -n - 2
            c_conj = True

        if 0 < n < 2:
            n_p = -n + 2
            c_conj = True

        if -2 < n < 0:
            n_p = n + 2
            normal = True

        if n == 0:
            n_p = 2
            normal = True
            c_conj = True

        n_hp = getidx(lmax, el, n_p)
        blm_ind[ind] = n_hp

        if m >= 0:
#        if True:
            if c_conj:
                pref_c_m2[ind] = (-1)**n_p
#            else:
            if normal:
                pref_p2[ind] = 1
        if m < 0:
#        if m <= 0:
#        if True:
            if c_conj:
                pref_c_p2[ind] = (-1)**n_p
#            else:
            if normal:
                pref_m2[ind] = 1

        if ret_wigner_pref:
            # See paper for (8 * np.pi**2) / (2*el + 1) factor.
            w_pref[ind] = (-1)**n * np.sqrt((2 * el + 1) / 4. / np.pi) \
                * (8 * np.pi**2) / (2 * el + 1)

    if ret_wigner_pref:
        return blm_ind, pref_p2, pref_c_m2, pref_m2, pref_c_p2, w_pref

    return blm_ind, pref_p2, pref_c_m2, pref_m2, pref_c_p2

def bell2bls(bell, parameters):
    '''
    Convert axisymmetric bell array to flmn-formatted array.
    For Gaussian beam, input is assumed to be normalized to
    the monopole.

    Arguments
    ----------
    bell: array
        Bell array of shape (lmax+1,)
    Returns
    -------
    bls: complex128 array
    '''

    bls_size = so3lib.flmn_size(parameters)
    bls = np.zeros(bls_size, dtype=np.complex128)

    assert isinstance(bell, np.ndarray), 'bell should be array'
    
    for ind in xrange(bls_size):
        el, m, n = so3lib.ind2elmn(ind, parameters)
        if n != 0:
            break
        # You want the m != 0 not to be zero. It should be constant (up to (-1)**m)
 #       if m > 0:
 #           m = -m
        # See paper for (8 * np.pi**2) / (2*el + 1) factor.
        bls[ind] = bell[el] #* (-1)**m * np.sqrt((2 * el + 1) / 4. / np.pi) \
#            * (8 * np.pi**2) / (2 * el + 1)

    return bls

def alm2almn_real(alm, alm_ind, pref, pref_c, w_pref=None, almn_out=None):
    '''
    Convert healpix-formatted alm array to so3 formatted
    array. Arguments are generated by get_alm_ind_real().

    Arguments
    ---------
    alm :  array-like
        Healpy alm array with same lmax as used in 
        get_alm_ind_real().
    
    Returns
    -------
    almn: array-like (complex128)
    '''

    if almn_out is not None:
        almn = almn_out

    almn = alm[alm_ind] * pref + np.conj(alm)[alm_ind] * pref_c

    if w_pref is not None:
        almn *= w_pref

    return almn

def blm2blmn_real(blm, blm_ind, w_pref=None, blmn_out=None):
    '''
    Convert healpix-formatted blm array to so3 formatted
    array. Arguments are generated by get_blm_ind_real().

    Arguments
    ---------
    blm :  array-like
        Healpy alm array with same lmax as used in 
        get_alm_ind_real().
    
    Returns
    -------
    blmn: array-like (complex128)
    '''

    if blmn_out is not None:
        blmn = blmn_out

    blmn = blm[blm_ind]

    if w_pref is not None:
        blmn *= w_pref

    return blmn


def alm2almn_complex(almp2, almm2, alm_ind, pref_p2, pref_c_m2, pref_m2,
                     pref_c_p2, w_pref=None, almn_out=None):
    '''
    Convert healpix-formatted alm array to so3 formatted
    array. Arguments are generated by get_alm_ind_complex().

    Returns
    -------
    almn: array-like (complex128)
    '''

    if almn_out is not None:
        almn = almn_out

    almn = almp2[alm_ind] * pref_p2 + almm2[alm_ind] * pref_m2
    almn += np.conj(almm2)[alm_ind] * pref_c_m2
    almn += np.conj(almp2)[alm_ind] * pref_c_p2

    if w_pref is not None:
        almn *= w_pref

    return almn

def get_so3_angles(parameters):
    '''
    Return the alpha, beta and gamma arrays
    that are sampled using the "MW" scheme.
    Note that the angles are offset by half
    a stepsize to accommodate for np.digitize.
        
    '''
    assert parameters.sampling.name == 'SO3_SAMPLING_MW', 'Use MW sampling'

    nalpha = so3lib.nalpha(parameters)
    nbeta = so3lib.nbeta(parameters)
    ngamma = so3lib.ngamma(parameters)

    alphas = np.arange(nalpha) * 2. * np.pi / float(nalpha)
    alphas += np.pi / float(nalpha)

    betas = np.arange(nbeta) * 2. * np.pi / (2. * nbeta - 1.) + np.pi / (2. * nbeta - 1.)
    betas -= np.pi / (2. * nbeta - 1.)

    gammas = np.arange(ngamma) * 2. * np.pi / float(ngamma)
    gammas += np.pi / float(ngamma)
    
    return alphas, betas, gammas

def get_s2_angles(parameters):
    '''
    Return the alpha and beta arrays
    that are sampled using the "MW" scheme.
    Note that the angles are offset by half
    a stepsize to accommodate for np.digitize.
        
    '''
    assert parameters.sampling.name == 'SO3_SAMPLING_MW', 'Use MW sampling'

    nalpha = so3lib.nalpha(parameters)
    nbeta = so3lib.nbeta(parameters)

    alphas = np.arange(nalpha) * 2. * np.pi / float(nalpha)
    alphas += np.pi / float(nalpha)

    betas = np.arange(nbeta) * 2. * np.pi / (2. * nbeta - 1.) + np.pi / (2. * nbeta - 1.)
    betas -= np.pi / (2. * nbeta - 1.)

    return alphas, betas

def radecpa2ind(ra, dec, pa, parameters, ret_unique=False, ret_inv=False):
    '''
    Convert ra, dec and pa arrays outputted by
    qpoint (bore2radec) to indices of the function
    on so3.

    Arguments
    ---------
    ra, dec, pa: array-like
        Output from M.bore2radec in degrees. These should lie
        between -180 and 180 degrees.
    parameters: so3 parameters struct
    ret_unique: bool, optional
        If True, return ordered unique indices.
        Default: False.
    ret_inv: bool, optional
        If True, return index array that will
        revert the unique index array back to 
        the regular indices array. Default: False.

    Returns
    -------
    indices: array-like
        so3 indices
    u_ind, tuple
        (u_a, u_b, u_g)-shaped tuple with unique a, b, g 
        (ordered) arrays. If ret_unique is True
    inv, array-like
        (3, tod_size)-shaped array with inverse 
        index arrays. Such that e.g. u_a[inv[0]] = a.
        If ret_inv is True
    '''

    nalpha = so3lib.nalpha(parameters)
    nbeta = so3lib.nbeta(parameters)
    ngamma = so3lib.ngamma(parameters)

    # Convert qpoint output to match with so3 sampling 
    # (this does not depend on sampling theorem)
    # apply minus sign and go to (0, 2pi)
    ra *= (np.pi / 180.)
    ra *= -1.
    ra += 2 * np.pi
    ra = np.mod(ra, 2 * np.pi, out=ra)

    # convert from latitude to colatitude
    dec *= (np.pi / 180.)
    dec *= -1.
    dec += np.pi / 2.
    dec = np.mod(dec, np.pi, out=dec)

    pa *= (np.pi / 180.)
    pa = np.mod(pa, 2 * np.pi, out=pa)

    alphas, betas, gammas = get_so3_angles(parameters)
    
    # These are indices of the so3 angle arrays.
    ra2a = np.digitize(ra, alphas) 
    dec2b = np.digitize(dec, betas) 
    pa2g = np.digitize(pa, gammas) 

    # fixing edge cases
    ra2a[ra2a == nalpha] = 0
    dec2b[dec2b == 0] = nbeta
    dec2b -= 1
    pa2g[pa2g == ngamma] = 0

    indices = ra2a + nalpha * dec2b + nalpha * nbeta * pa2g

    # this is only needed for the hamiltonian mc sampler, ignore for now.
    if ret_unique:        
        u_a = np.unique(ra2a, return_inverse=ret_inv)
        u_b = np.unique(dec2b, return_inverse=ret_inv)
        u_g = np.unique(pa2g, return_inverse=ret_inv)
        
        if ret_inv:
            inv = np.stack([u_a[1], u_b[1], u_g[1]])
            return indices, (u_a[0], u_b[0], u_g[0]), inv
        else:
            return indices, (u_a, u_b, u_g)
    else:
        return indices

def radec2ind(ra, dec, parameters, ret_unique=False, ret_inv=False):
    '''
    Convert ra and dec arrays outputted by
    qpoint (bore2radec) to indices of the function
    on s2 (ssht).

    Arguments
    ---------
    ra, dec: array-like
        Output from M.bore2radec in degrees. These should lie
        between -180 and 180 degrees.
    parameters: so3 parameters struct
    ret_unique: bool, optional
        If True, return ordered unique indices.
        Default: False.
    ret_inv: bool, optional
        If True, return index array that will
        revert the unique index array back to 
        the regular indices array. Default: False.

    Returns
    -------
    indices: array-like
        s2 indices
    u_ind, tuple
        (u_a, u_b)-shaped tuple with unique a, b
        (ordered) arrays. If ret_unique is True
    inv, array-like
        (2, tod_size)-shaped array with inverse 
        index arrays. Such that e.g. u_a[inv[0]] = a.
        If ret_inv is True
    '''

    nalpha = so3lib.nalpha(parameters)
    nbeta = so3lib.nbeta(parameters)

    # Convert qpoint output to match with so3 sampling 
    # (this does not depend on sampling theorem)
    # apply minus sign and go to (0, 2pi)
    ra *= (np.pi / 180.)
    ra *= -1.
    ra += 2 * np.pi
    ra = np.mod(ra, 2 * np.pi, out=ra)

    # convert from latitude to colatitude
    dec *= (np.pi / 180.)
    dec *= -1.
    dec += np.pi / 2.
    dec = np.mod(dec, np.pi, out=dec)

    alphas, betas = get_s2_angles(parameters)
    
    # These are indices of the so3 angle arrays.
    ra2a = np.digitize(ra, alphas) 
    dec2b = np.digitize(dec, betas) 

    # fixing edge cases
    ra2a[ra2a == nalpha] = 0
    dec2b[dec2b == 0] = nbeta
    dec2b -= 1

    indices = ra2a + nalpha * dec2b 

    # this is only needed for the hamiltonian mc sampler, ignore for now.
    if ret_unique:        
        u_a = np.unique(ra2a, return_inverse=ret_inv)
        u_b = np.unique(dec2b, return_inverse=ret_inv)
        
        if ret_inv:
            inv = np.stack([u_a[1], u_b[1]])
            return indices, (u_a[0], u_b[0]), inv
        else:
            return indices, (u_a, u_b)
    else:
        return indices


def flmn2tod(flmn, indices, parameters, sim_tod=None, func=None):
    '''
    Calculates the inverse wigner transform and populates
    the sim_tod using the indices.

    Arguments
    ---------
    sim_tod: array-like
        (empty of zero) array that be filled to become tod.
    func: array-like
        (empty of zero) array that be filled to become function
        on so3.

    Returns
    -------
    sim_tod: array-like
    '''

    if func is None:        
        if parameters.reality:
            func = np.zeros(so3lib.f_size(parameters))
        else:
            func = np.zeros(so3lib.f_size(parameters),
                            dtype=np.complex128)
    elif parameters.reality:
        assert func.dtype == 'float64', 'dtype should be float64'

    else:
        assert func.dtype == 'complex128', 'dtype should be complex128'
            

    if sim_tod is None:        
        sim_tod = np.zeros(indices.size, dtype='float64')
    
    if parameters.reality:
        so3lib.inverse_via_ssht_real(func, flmn, parameters)
        sim_tod += func[indices]

    else:
        so3lib.inverse_via_ssht(func, flmn, parameters)
        # imaginary part should just be floating point errors
        # Factor 1/4 should be 1/2, but this works for now....
        # Might actually be 1/4, see notes
        print np.mean(np.imag(sim_tod))
        print np.std(np.imag(sim_tod))
        print np.max(np.abs(np.imag(sim_tod)))
        sim_tod += np.real(func[indices] + np.conj(func[indices])) / 4.
    return sim_tod
    
