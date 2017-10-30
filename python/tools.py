import numpy as np
import healpy as hp
import inspect

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
        beam (spin +2 part is zero).

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

def get_copol_blm(blm, normalize=False, deconv_q=False):
    '''
    Create the spin \pm 2 coefficients of a unpolarized
    beam, assuming a co-polarized beam. See Hivon, Ponthieu
    2016.

    Arguments
    ---------
    blm : array-like
        Healpix-ordered blm array for unpolarized beam.
        Requires lmax=mmax
    normalize : bool
        Normalize unpolarized beam to monopole
    deconv_q : bool
        Divide blm by sqrt(4 pi / (2 ell + 1)) before
        computing spin harmonic coefficients

    Returns
    -------
    blm, blmm2, blmp2 : tuple of array-like
        The spin \pm 2 harmonic coefficients of the
        co-polarized beam.
    '''

    # normalize beams and extract spin-m2 beams
    lmax = hp.Alm.getlmax(blm.size)
    lm = hp.Alm.getlm(lmax)
    getidx = hp.Alm.getidx

    if deconv_q:
        blm *= 2 * np.sqrt(np.pi / (2 * lm[0] + 1))
    if normalize:
        blm /= blm[0]

    blmm2 = np.zeros(blm.size, dtype=np.complex128)
    blmp2 = np.zeros(blm.size, dtype=np.complex128)

    for m in xrange(lmax+1): # loop over spin -2 m's
        start = getidx(lmax, m, m)
        if m < lmax:
            end = getidx(lmax, m+1, m+1)
        else:
            start = end
        if m == 0:
            #+2 here because spin-2, so we can't have nonzero ell=1 bins
            blmm2[start+2:end] = np.conj(blm[2*lmax+1:3*lmax])

            blmp2[start+2:end] = blm[2*lmax+1:3*lmax]

        elif m == 1:
            #+1 here because spin-2, so we can't have nonzero ell=1 bins
            blmm2[start+1:end] = -np.conj(blm[start+1:end])

            blmp2[start+2:end] = blm[3*lmax:4*lmax-2]

        else:
            start_0 = getidx(lmax, m-2, m-2) # spin-0 start and end
            end_0 = getidx(lmax, m-1, m-1)

            blmm2[start:end] = blm[start_0+2:end_0]

            start_p0 = getidx(lmax, m+2, m+2)
            if m + 2 > lmax:
                # stop filling blmp2
                continue
            end_p0 = getidx(lmax, m+3, m+3)

            blmp2[start+2:end] = blm[start_p0:end_p0]

    return blm, blmm2, blmp2


def extract_func_kwargs(func, kwargs, pop=False, others_ok=True, warn=False):
    """
    Extract arguments for a given function from a kwargs dictionary

    Arguments
    ---------
    func : function or callable
        This function's keyword arguments will be extracted.
    kwargs : dict
        Dictionary of keyword arguments from which to extract.
        NOTE: pass the kwargs dict itself, not **kwargs
    pop : bool, optional
        Whether to pop matching arguments from kwargs.
    others_ok : bool
        If False, an exception will be raised when kwargs contains keys
        that are not keyword arguments of func.
    warn : bool
        If True, a warning is issued when kwargs contains keys that are not
        keyword arguments of func.  Use with `others_ok=True`.

    Returns
    -------
    Dict of items from kwargs for which func has matching keyword arguments
    """
    spec = inspect.getargspec(func)
    func_args = set(spec.args[-len(spec.defaults):])
    ret = {}
    for k in kwargs.keys():
        if k in func_args:
            if pop:
                ret[k] = kwargs.pop(k)
            else:
                ret[k] = kwargs.get(k)
        elif not others_ok:
            msg = "Found invalid keyword argument: {}".format(k)
            raise TypeError(msg)
    if warn and kwargs:
        s = ', '.join(kwargs.keys())
        warn("Ignoring invalid keyword arguments: {}".format(s), Warning)
    return ret

def radec2ind_hp(ra, dec, nside):
    '''
    Turn qpoint ra and dec output into healpix ring-order
    map indices. Note, currently modifies ra and dec in-place.

    Arguments
    ---------
    ra : array-like
        Right ascension in degrees.
    dec : array-like
        Declination in degrees.
    nside : int
        nside parameter of healpy map.

    Returns
    -------
    pix : array-like
        Pixel indices corresponding to ra, dec.
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

def angle_gen(angles):
    '''
    Generator that yields cyclic permmuation
    of elements of input array.

    Arguments
    ---------
    angles : array-like
        Array to be cycled through
    '''

    n = 0
    while True:
        yield angles[n]
        n += 1
        n = n % len(angles)

def quat_left_mult(q2, q):
    '''
    Calculate q3 = q2 * q

    Arguments
    ---------
    q2 : array-like
        Float array of shape (4,), representing a
        quaternion
    q : array-like
        Float array of shape (4,), representing a
        quaternion

    Returns
    -------
    q3 : array-like
        Float array of shape (4,), representing a
        quaternion
    '''

    q3 = np.zeros(4, dtype=float)

    q3[0] = q2[0]*q[0] - q2[1]*q[1] - q2[2]*q[2] - q2[3]*q[3]
    q3[1] = q2[1]*q[0] + q2[0]*q[1] - q2[3]*q[2] + q2[2]*q[3]
    q3[2] = q2[2]*q[0] + q2[3]*q[1] + q2[0]*q[2] - q2[1]*q[3]
    q3[3] = q2[3]*q[0] - q2[2]*q[1] + q2[1]*q[2] + q2[0]*q[3]

    return q3

def quat_norm(q, inplace=False):
    '''
    Normalize a quaternion.

    Arguments
    ---------
    q : array-like
        Float array of shape (4,), representing a
        quaternion
    inplace : bool, optional
        Perform normalization in place, default=False

    Returns
    -------
    qn : array-like
        Float array of shape (4,), representing a
        normalized quaternion
    '''

    if not inplace:
        q = q.copy()
    q /= np.sqrt(np.sum(q**2))

    return q

def quat_conj(q):
    '''
    Calculate conjugate quaternion.

    Arguments
    ---------
    q : array-like
        Float array of shape (4,), representing a
        quaternion

    Returns
    -------
    qc : array-like
        Float array of shape (4,), representing the
        conjugate quaternion
    '''

    qc = np.zeros_like(q)

    qc[0] = q[0]
    qc[1] = -q[1]
    qc[2] = -q[2]
    qc[3] = -q[3]

    return qc

def quat_inv(q):

    '''
    Calculate inverse quaternion.

    Arguments
    ---------
    q : array-like
        Float array of shape (4,), representing a
        quaternion

    Returns
    -------
    qi : array-like
        Float array of shape (4,), representing the
        inverse quaternion
    '''

    qi = quat_conj(q)
    qi /= np.sum(qi**2)

    return qi

def quat_conj_by(q, q2):
    '''
    Conjugate q by q2, i.e. returns
    q3 = q2 * q * inv(q2).

    Arguments
    ---------
    q : array-like
        Float array of shape (4,), representing the
        quaternion to be conjugated by q2
    q2 : array-like
        Float array of shape (4,), representing a
        quaternion


    Returns
    -------
    q3 : array-like
        Float array of shape (4,), representing
        q conjugated by q2
    '''

    q3 = quat_left_mult(q, quat_inv(q2))
    q3 = quat_left_mult(q2, q3)

    return q3


#import quaternion as qt
#a = np.array([3., 2., 5., 6.])
#a = quat_norm(a, inplace=True)
#b = np.array([1., 6., 5., 2.])
#b = quat_norm(b, inplace=True)
#c = quat_left_mult(a, b)
#print c
#aq = qt.as_quat_array(a)
#bq = qt.as_quat_array(b)
#print aq * bq
#print quat_left_mult(a, b)
#print quat_left_mult(b, a)
#print a
#print aq
#print b
#print bq

#print quat_conj_by(a, b)
#print bq * aq * np.conj(bq)

#print quat_inv(a)
#print np.conj(aq)

#print quat_norm(c)

