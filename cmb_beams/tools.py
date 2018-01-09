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
        
    Keyword arguments
    -----------------
    mmax_old : int
        m-bandlimit alm. If None, assume mmax_old=lmax
        (default : None)

    Returns
    -------
    alm_new: newly-allocated complex alm array that
        only contains modes up to lmax_new. If alm 
        is sequence of arrays, return tuple with
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

def scale_blm(blm, normalize=False, deconv_q=False):
    '''
    Scale or normalize blm(s)

    Arguments
    ---------
    blm : array-like
        blm or blm(s) ([blm, blmm2, blmp2])

    Keyword arguments
    ---------
    normalize : bool
        Normalize beams(s) to monopole unpolarized beam
        (default : False)
    deconv_q : bool
        Multiply blm(s) by sqrt(4 pi / (2 ell + 1)) before
        computing spin harmonic coefficients (default : False)

    Returns
    -------
    blm : array-like
    '''

    if not normalize and not deconv_q:
        return blm

    blm = np.atleast_2d(blm)

    lmax = hp.Alm.getlmax(blm[0].size)
    ell = np.arange(lmax+1)

    if deconv_q:
        blm *= 2 * np.sqrt(np.pi / (2 * ell + 1))
    if normalize:
        blm /= blm[0,0]

    if blm.shape[0] == 1:
        return blm[0]
    else:
        return blm

def unpol2pol(blm):
    '''
    Compute spin \pm 2 blm coefficients by transforming input 
    blm (HEALPix) array corresponing to a real spin-0 field. 

    Arguments
    ---------
    blm : array-like
        Spin-0 harmonic coefficients of real field in HEALPix format.

    Returns
    -------
    blmm2, blmp2 : tuple of array-like
        The spin \pm 2 harmonic coefficients computed from blm.

    Notes
    -----
    Uses the approximation introduced in Hivon, Mottet, Ponthieu 2016 
    (eq. G.8). Should be accurate for input blm that are roughly 
    constant on \Delta ell = 5 for ell < ~20.
    '''

    lmax = hp.Alm.getlmax(blm.size)
    lm = hp.Alm.getlm(lmax)
    getidx = hp.Alm.getidx

    blmm2 = np.zeros(blm.size, dtype=np.complex128)
    blmp2 = np.zeros(blm.size, dtype=np.complex128)

    for m in xrange(lmax+1): # loop over spin -2 m's
        start = getidx(lmax, m, m)
        if m < lmax:
            end = getidx(lmax, m+1, m+1)
        else:
            # we're in the very last element
            start = end
            end += 1 # add one, otherwise you have empty array
            assert end == hp.Alm.getsize(lmax)
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

    return blmm2, blmp2

def get_copol_blm(blm, c2_fwhm=None, **kwargs):
    '''
    Create the spin \pm 2 coefficients of a unpolarized
    beam, assuming purely co-polarized beam. See Hivon, Ponthieu
    2016.

    Arguments
    ---------
    blm : array-like
        Healpix-ordered blm array for unpolarized beam.
        Requires lmax=mmax

    Keyword arguments
    ---------
    c2 : float, None
        fwhm in arcmin. Used to multiply \pm 2 
        coefficients with exp 2 sigma**2. Needed 
        to match healpy Gaussian smoothing for pol.
        (default : None)
    kwargs : {scale_blm_opts}

    Returns
    -------
    blm, blmm2, blmp2 : tuple of array-like
        The spin 0 and \pm 2 harmonic coefficients of the
        co-polarized beam.
    '''
    # normalize beams 
    blm = scale_blm(blm, **kwargs)

    blmm2, blmp2 = unpol2pol(blm)

    if c2_fwhm:
        s2fwhm = 2 * np.sqrt(2 * np.log(2))
        expsig2 = np.exp(2 * (np.radians(c2_fwhm / 60.) / s2fwhm)**2)
        blmm2 *= expsig2
        blmp2 *= expsig2

    return blm, blmm2, blmp2

def get_pol_beam(blm_q, blm_u, **kwargs):
    '''
    Create spin \pm 2 blm coefficients using spin-0 
    SH coefficients of the Q and U beams on a cartesian
    (Ludwig's third convention) basis.

    Arguments
    ---------
    blm_q : array-like
        Healpy blm array 
    blm_q : array-like
        Healpy blm array 

    Keyword arguments
    -----------------
    kwargs : {scale_blm_opts}

    Returns
    -------
    blmm2, blmp2 : tuple of array-like
        The spin \pm 2 harmonic coefficients of the
        polarized beam.    
    '''
    
    blm_q = scale_blm(blm_q, **kwargs)
    blm_u = scale_blm(blm_u, **kwargs)

    blmm2_q, blmp2_q = unpol2pol(blm_q)
    blmm2_u, blmp2_u = unpol2pol(blm_u)

    # note the signs here, phi -> -phi+pi w/ respect to Challinor
    blmm2 = blmm2_q + 1j * blmm2_u
    blmp2 = blmp2_q - 1j * blmp2_u

    return blmm2, blmp2

def spin2eb(almm2, almp2):
    '''
    Convert spin-harmonic coefficients
    to E and B mode coefficients.

    Arguments
    ---------
    almm2 : array-like
       Healpix-ordered complex array with spin-(-2) 
       coefficients
    almp2 : array-like
       Healpix-ordered complex array with spin-(+2) 
       coefficients

    Returns
    -------
    almE : array-like
        Healpix ordered array with E-modes
    almB : array-like
        Healpix ordered array with B-modes
    '''
    
    almE = almp2 + almm2 
    almE /= -2.

    almB = almp2 - almm2
    almB *= (1j / 2.)

    return almE, almB

def eb2spin(almE, almB):
    '''
    Convert to E and B mode coefficients 
    to spin-harmonic coefficients.

    Arguments
    ---------
    almE : array-like
        Healpix ordered array with E-modes
    almB : array-like
        Healpix ordered array with B-modes

    Returns
    -------
    almm2 : array-like
       Healpix-ordered complex array with spin-(-2) 
       coefficients
    almp2 : array-like
       Healpix-ordered complex array with spin-(+2) 
       coefficients
    '''

    almm2 = -1 * (almE - 1j * almB)
    almp2 = -1 * (almE + 1j * almB)

    return almm2, almp2

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
        
    Yields
    ------
    angle : scalar
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

