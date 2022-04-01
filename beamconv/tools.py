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

    for m in range(min(lmax_new, mmax_old) + 1):
        indices[nstart:nstart+lmax_new+1-m] = \
            np.arange(start, start+lmax_new+1-m)
        start += lmax + 1 - m
        nstart += lmax_new + 1 - m

    # Fancy indexing so numpy makes copy of alm
    if seq:
        return tuple([alm[d][indices] for d in range(len(alm))])

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
        qell = 2 * np.sqrt(np.pi / (2. * ell + 1))
        for i in range(blm.shape[0]):
            hp.almxfl(blm[i], qell, inplace=True)
    if normalize:
        blm /= blm[0,0]

    if blm.shape[0] == 1:
        return blm[0]
    else:
        return blm

def shift_blm(blmE, blmB, shift, eb=True):
    '''
    Return copy of input with m values shifted
    up or down.

    \pm2b_lm^new = \pm2bl(m \mp shift)^old 
    
    Arguments
    ---------
    blmE : complex array
        E-mode coefficients.
    blmB : complex array
        B-mdoe coefficients.
    shift : int

    Keyword arguments
    -----------------
    eb : bool
        Whether input/output is E and B, or -2 and +2 spin (default : True).

    Returns
    -------
    blmE : complex array
        Updated E-mode coefficients.
    BlmB : complex array
        Updated B-mode coefficients.
    
    Raises
    ------
    ValueError
        If input sizes do not match.
        If shift > lmax.
    '''

    if blmE.size != blmB.size:
        raise ValueError('Input sizes do not match')

    lmax = hp.Alm.getlmax(blmE.size)

    if shift > lmax:
        raise ValueError('Shift exceeds lmax')

    if eb:
        blmm2, blmp2 = eb2spin(blmE, blmB)
    else:
        blmm2, blmp2 = blmE, blmB
        
    blmm2_new = np.zeros_like(blmm2)
    blmp2_new = np.zeros_like(blmp2)        

    # First we do +2blm^new.    
    # Loop over m modes in new blms
    for m in range(lmax + 1):

        # Slice into new blms.
        start = hp.Alm.getidx(lmax, m, m)
        end = start + lmax + 1 - m        

        m_old = m - shift
        if abs(m_old) > lmax:
            continue
        
        if m_old >= 0:
            bell_old = blm2bl(blmp2, m=m_old, full=True)         
        elif m_old < 0: 
            bell_old = blm2bl(blmm2, m=abs(m_old), full=True)           
            bell_old = np.conj(bell_old) * (-1) ** m_old
 
        # Length bell_old is always (lmax + 1).
        blmp2_new[start:end] = bell_old[m:]

    # Now we do -2blm^new.        
    for m in range(lmax + 1):

        start = hp.Alm.getidx(lmax, m, m)
        end = start + lmax + 1 - m        
        
        m_old = m + shift
        if abs(m_old) > lmax:
            continue
        
        if m_old >= 0:
            bell_old = blm2bl(blmm2, m=m_old, full=True)         
        elif m_old < 0: 
            bell_old = blm2bl(blmp2, m=abs(m_old), full=True)           
            bell_old = np.conj(bell_old) * (-1) ** m_old

        blmm2_new[start:end] = bell_old[m:]

    if eb:
        return spin2eb(blmm2_new, blmp2_new)
    else:
        return blmm2_new, blmp2_new
        
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

    for m in range(lmax+1): # loop over spin -2 m's
        start = getidx(lmax, m, m)
        if m < lmax:
            end = getidx(lmax, m+1, m+1)
        else:
            # We're in the very last element
            start = end
            end += 1 # add one, otherwise you have empty array
            assert end == hp.Alm.getsize(lmax)
        if m == 0:
            # +2 here because spin-2, so we can't have nonzero ell=1 bins
            blmm2[start+2:end] = np.conj(blm[2*lmax+1:3*lmax])

            blmp2[start+2:end] = blm[2*lmax+1:3*lmax]

        elif m == 1:
            # +1 here because spin-2, so we can't have nonzero ell=1 bins
            blmm2[start+1:end] = -np.conj(blm[start+1:end])

            blmp2[start+2:end] = blm[3*lmax:4*lmax-2]

        else:
            start_0 = getidx(lmax, m-2, m-2) # Spin-0 start and end
            end_0 = getidx(lmax, m-1, m-1)

            blmm2[start:end] = blm[start_0+2:end_0]

            start_p0 = getidx(lmax, m+2, m+2)
            if m + 2 > lmax:
                # Stop filling blmp2
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
    c2_fwhm : float, None
        fwhm in arcmin. Used to multiply \pm 2 blm coefficients
        with exp 2 sigma**2. Needed to match healpy Gaussian
        smoothing for pol (see Challinor et al. 2000)
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
        The spin -2 and +2 harmonic coefficients of the
        polarized beam.
    '''

    blm_q = scale_blm(blm_q, **kwargs)
    blm_u = scale_blm(blm_u, **kwargs)

    blmm2_q, blmp2_q = unpol2pol(blm_q)
    blmm2_u, blmp2_u = unpol2pol(blm_u)

    # Note the signs here, phi -> -phi+pi compared to to
    # Challinor et al. 2000
    blmm2 = blmm2_q + 1j * blmm2_u
    blmp2 = blmp2_q - 1j * blmp2_u

    return blmm2, blmp2

def spin2eb(almm2, almp2, spin=2):
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

    Keyword arguments
    -----------------
    spin : int
        Spin of input. Odd spins receive relative
        minus sign between input in order to be consistent
        with HEALPix alm2map_spin.

    Returns
    -------
    almE : array-like
        Healpix ordered array with E-modes
    almB : array-like
        Healpix ordered array with B-modes\

    Raises
    ------
    ValueError
        If spin is not an integer.

    Notes
    -----
    See https://healpix.jpl.nasa.gov/html/subroutinesnode12.htm
    '''

    if int(spin) != spin:
        raise ValueError('Spin must be integer')
    
    almE = almp2 + almm2 * (-1.) ** spin
    almE /= -2.

    almB = almp2 - almm2 * (-1.) ** spin
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

def radec2colatlong(ra, dec):
    '''
    In-place conversion of qpoints ra, dec output to co-latitude
    and longitude used for healpy.

    Long = RA
    Co-lat = -DEC + pi/2

    Arguments
    ---------
    ra : array-like
        Right ascension in degrees.
    dec : array-like
        Declination in degrees.

    '''

    # Convert RA to healpix longitude (=phi)
    ra *= (np.pi / 180.)
    ra = np.mod(ra, 2 * np.pi, out=ra)

    # convert from DEC to co-latitude (=theta)
    dec *= (np.pi / 180.)
    dec *= -1.
    dec += np.pi / 2.
    dec = np.mod(dec, np.pi, out=dec)

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

    radec2colatlong(ra, dec)
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
    qi /= np.sum(qi**2, dtype=float)

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

def blm2bl(blm, m=0, copy=True, full=False):
    '''
    A tool to return blm for a fixed m-mode

    Arguments
    ---------
    blm : array-like
       Complex numpy array corresponding to the spherical harmonics of the beam

    Keyword arguments
    -----------------
    m : int
        The m-mode being requested (note m >= 0) (default : 0)
    copy : bool
        Return copied slice or not (default : True)
    full : bool
        If set, always return full-sized (lmax + 1) array
        Note, this always produces a copy. (default : False)

    Returns
    -------
    bl : array-like
        Array of bl's for the m-mode requested
    '''

    if blm.ndim > 1:
        raise ValueError('blm should have have ndim == 1')
    if m < 0:
        raise ValueError('m cannot be negative')

    lmax = hp.Alm.getlmax(blm.size)

    start = hp.Alm.getidx(lmax, m, m)
    end = start + lmax + 1 - m

    bell = blm[start:end]

    if full:
        bell_full = np.zeros(lmax + 1, dtype=blm.dtype)
        bell_full[m:] = bell
        bell = bell_full
        
    if copy:
        return bell.copy()
    else:
        return bell

def sawtooth_wave(num_samp, scan_speed, period):
    '''
    Return sawtooth wave.

    Arguments
    ---------
    num_samp : int
        Size of output in samples.
    scan_speed : float
        Degrees per sample.
    period : float
        Period of wave in degrees.

    Returns
    -------
    az : array-like
         Azimuth value for each sample in degrees.
    '''

    tot_degrees = scan_speed * num_samp
    az = np.linspace(0, tot_degrees, num=num_samp, dtype=float,
                     endpoint=False)
    np.mod(az, period, out=az)

    return az

def cross_talk(tod_a, tod_b, ctalk=0.01):
    '''
    Add a fraction of data from one time-stream to
    another and vice versa. In place modification.

    Arguments
    ---------
    tod_a, tod_b : array-like
        Equal-sized arrays that will cross-talk to each other.

    Keyword arguments
    -----------------
    ctalk : float
        Amount of cross-talk, i.e. fraction of each time-stream
        that is added to the other. (default : 0.01)
    '''

    tod_c = tod_a.copy()
    tod_c += tod_b
    tod_c *= ctalk
    tod_a *= (1. - ctalk)
    tod_b *= (1. - ctalk)
    tod_a += tod_c
    tod_b += tod_c

def iquv2ippv(mueller_mat):
    '''
    Returns a 4x4 matrix in the (I, P, Pbar, V) base with P = Q+jU

    Argument
    ----------
    mueller_mat : array-like, size (4,4)
    '''

    A = np.array([[1, 0,  0,  0],
                  [0, 1,  1j, 0],
                  [0, 1, -1j, 0],
                  [0, 0,  0,  1]], dtype=complex)

    Ainv = np.array([[1,  0.0,  0.0, 0],
                     [0,  0.5,  0.5, 0],
                     [0, -.5j, .5j,  0],
                     [0,  0.0,  0.0, 1]], dtype=complex)

    mueller_mat = np.matmul(Ainv,np.matmul(mueller_mat,A))
    return mueller_mat

def ippv2iquv(mueller_mat):
    '''
    Returns a 4x4 matrix in the (I, Q, U, V) base with Q = P+Pbar

    Argument
    ----------
    mueller_mat : array-like, size (4,4)
    '''

    Ainv = np.array([[1, 0,  0,  0],
                  [0, 1,  1j, 0],
                  [0, 1, -1j, 0],
                  [0, 0,  0,  1]], dtype=complex)

    A = np.array([[1,  0.0,  0.0, 0],
                     [0,  0.5,  0.5, 0],
                     [0, -.5j, .5j,  0],
                     [0,  0.0,  0.0, 1]], dtype=complex)

    mueller_mat = np.matmul(Ainv,np.matmul(mueller_mat,A))
    return mueller_mat

def tukey_window(n):
    '''
    Return tukey window (alpha=0.5).
    See https://en.wikipedia.org/wiki/Window_function#Tukey_window.

    Arguments
    ---------
    n : int
        Window_length.

    Returns
    -------
    window : ndarray
        Tukey window.
    '''
    alpha = 0.5
    L = float(n + 1)

    window = np.ones(n, dtype=float)
    x = np.arange(int(alpha * L / 2.) + 1, dtype=float)

    window[:int(alpha * L / 2.) + 1] -= 0.5 * np.cos(2 * np.pi * x / alpha / L) + .5
    # Use symmetry of window.
    window[-int(n/2.):] = np.flip(window[:int(n/2.)], 0)

    return window

def filter_ft_hwp(fd, center_idx, filter_width):
    '''
    Apply window to fourier transformed real data around given frequency.

    Note, calculated in-place for memory reasons.

    Arguments
    ---------
    fd : ndarray
        Fourier-transformed (real) data, fd[0] should contain the zero
        frequency term.
    center_idx : int
        Apply filter around this element of the input array.
    filter_width : int
        Wdith of filter measured in elements of fd. If even, 1 is
        added to window width.
    '''

    # We select a view from the input array and apply the window to it,
    # this avoids creating another large array for the window.
    left = center_idx - int(filter_width / 2.)
    right = center_idx + int(filter_width / 2.) + 1
    arr2filter = fd[left:right]

    # Apply the filter.
    w = tukey_window(arr2filter.size)
    arr2filter *= w

    # Set array outside filter to zero.
    fd[:left] *= 0.
    fd[right:] *= 0.

    return

def filter_tod_hwp(tod, fsamp, hwp_freq):
    '''
    Convolve TOD with window around 4 * fHWP (in place).

    Arguments
    ---------
    tod : ndarray
    fsamp : float
        Sample frequency of TOD in Hz
    hwp_freq : float
        HWP rotation frequency (fHWP) in Hz.
    '''

    fd = np.fft.rfft(tod)

    # Find sample that corresponds to 4 * fHWP.
    # We could use numpy.fft.rfftfreq here, but we want to
    # avoid loading another large array in memory.

    center_idx = int(round((4 * hwp_freq) / float(fsamp) * tod.size))

    # For now, we use a window width of 2 * hwp_freq.
    left_idx = round((3 * hwp_freq) / float(fsamp) * tod.size)
    right_idx = round((5 * hwp_freq) / float(fsamp) * tod.size)
    filter_width = right_idx - left_idx + 1

    filter_ft_hwp(fd, center_idx, filter_width)

    # This is not really in place I think.
    tod[:] = np.fft.irfft(fd, n=tod.size)

    return

def mueller2spin(mueller_mat):
    '''
    Transform input Mueller matrix to complex
    spin basis.

    Arguments
    ---------
    mueller_mat : (4, 4) array

    Returns
    -------
    spin_mat : (4, 4) complex array    
    '''
    sqrt2 = np.sqrt(2)
    tmat = np.asarray([[1, 0, 0, 0],
                       [0, 1/sqrt2, 1j/sqrt2, 0],
                       [0, 1/sqrt2, -1j/sqrt2, 0],
                       [0, 0, 0, 1]])
    tmatinv = np.linalg.inv(tmat)
    
    return np.dot(np.dot(tmat, mueller_mat), tmatinv)        

def load_mueller(hwp_filename, freq, vartheta=0.0):
    '''
    Load a pre-computed Mueller matrix from a Pickle file.

    The function looks for the Mueller matrix computed for the frequency and
    incidence angle (vartheta) pair that most closely matches the (freq, vartheta)
    pair that is input to the function.

    mueller is a N x M x 4 x 4 numpy array where N, and M, stand for the number
    of elements along the vartheta and frequency directions.

    Arguments
    ---------
    hwp_filename : string/path
        absolute path to pickle file
    freq : float
        the frequency (in GHz) of the HWP mueller elements

    Keyword arguments
    -----------------
    vartheta : float

    Returns
    -------

    mueller_mat : array-like, size (4,4)

    '''

    fid = open(hwp_filename, 'rb')
    hwp_data = pickle.load(fid)

    print('Debugging:')
    print(hwp_data.keys())

    print('mueller' in hwp_data.keys())

    # assert ['muellers', 'freqs', 'varthetas'] in hwp_data.keys(), \
    #     'keys/parameters (muellers, freqs, and varthetas) should be part of dictionary'

    muellers = hwp_data['muellers']
    freqs = hwp_data['freqs']
    varthetas = hwp_data['varthetas']

    freqi = np.argmin(np.abs(freq - np.array(freqs)))
    vi = np.argmin(np.abs(vartheta - np.array(varthetas)))

    print('Frequency for HWP found to be: {}'.format(freqs[freqi]))

    return np.squeeze(muellers[vi, freqi, :, :])
