'''
ctypes wrapper for (a small part of) the so3 code.
'''

import numpy as np
import ctypes as ct
from enumstruct import StructureWithEnums
import enum

class CtypesEnum(enum.IntEnum):
    """
    A ctypes-compatible IntEnum superclass.
    see: http://www.chriskrycho.com/2015/
    ctypes-structures-and-dll-exports.html
    """
    @classmethod
    def from_param(cls, obj):
        return int(obj)

libpath = '/mn/stornext/u3/adriaand/cmb_beams/src/c/'
libiw = np.ctypeslib.load_library("libso3f", libpath)

# **********************************************************************
# Types
# **********************************************************************
NDP = np.ctypeslib.ndpointer

complex_double = NDP(dtype=np.complex128, ndim=1, flags=['A','C'])
c_double = NDP(dtype=np.double, ndim=1, flags=['A','C'])

class so3_n_order_t(enum.IntEnum):
    SO3_N_ORDER_ZERO_FIRST = 0
    SO3_N_ORDER_NEGATIVE_FIRST = 1
    SO3_N_ORDER_SIZE = 2

class so3_storage_t(enum.IntEnum):
    SO3_STORAGE_PADDED = 0
    SO3_STORAGE_COMPACT = 1
    SO3_STORAGE_SIZE = 2

class so3_n_mode_t(enum.IntEnum):
    SO3_N_MODE_ALL = 0
    SO3_N_MODE_EVE = 1
    SO3_N_MODE_ODD = 2
    SO3_N_MODE_MAXIMUM = 3
    SO3_N_MODE_L = 4
    SO3_N_MODE_SIZE = 5

class so3_sampling_t(enum.IntEnum):
    SO3_SAMPLING_MW = 0
    SO3_SAMPLING_MW_SS = 1
    SO3_SAMPLING_SIZE = 2

#class ssht_dl_method_t(enum.IntEnum):
class ssht_dl_method_t(CtypesEnum):
    SSHT_DL_RISBO = 0
    SSHT_DL_TRAPANI = 1

class ssht_dl_size_t(CtypesEnum):
    SSHT_DL_QUARTER = 0
    SSHT_DL_QUARTER_EXTENDED = 1    
    SSHT_DL_HALF = 2
    SSHT_DL_FULL = 3

class so3_parameters_t(StructureWithEnums):
    _fields_ = [
        ('verbosity', ct.c_int),
        ('reality', ct.c_int),
        ('L0', ct.c_int),
        ('L', ct.c_int),
        ('N', ct.c_int),
        ('sampling_scheme', ct.c_int),
        ('n_order', ct.c_int),
        ('storage', ct.c_int),
        ('n_mode', ct.c_int),
        ('dl_method', ct.c_int),
        ('steerable', ct.c_int),
        ]
    _map = {
        "n_order": so3_n_order_t, "storage": so3_storage_t,
        "n_mode": so3_n_mode_t, "sampling": so3_sampling_t,
        "dl_method": ssht_dl_method_t
        }

so3_parameters_t_p = ct.POINTER(so3_parameters_t)

# **********************************************************************
# Functions
# **********************************************************************

libiw._so3_core_inverse_via_ssht.restype = None
libiw._so3_core_inverse_via_ssht.argtypes = [complex_double,
                          complex_double, so3_parameters_t_p]

def inverse_via_ssht(f, flmn, parameters):
    '''
    Wrapper for c function, so user needs to
    allocate memory for complex output f.

    Arguments
    ---------
    flms: array-like
        Allocate a contiguous (row-major) numpy
        complex128 array (at least len = (2*N-1)*L*L.)
    f: array-like
        Allocate a contiguous (row-major) numpy
        complex128 array (at least len =  (2*L) * (L+1) * (2*N-1).)
    parameters: fully populated so3_parameters_t object
    '''

    return libiw._so3_core_inverse_via_ssht(f, flmn, parameters)

libiw._so3_core_inverse_via_ssht_real.restype = None
libiw._so3_core_inverse_via_ssht_real.argtypes = [c_double,
                          complex_double, so3_parameters_t_p]

def inverse_via_ssht_real(f_real, flmn, parameters):
    '''
    Wrapper for c function, so user needs to
    allocate memory for output f_real.

    Arguments
    ---------
    flms: array-like
        Allocate a contiguous (row-major) numpy
        complex128 array (at least len = (2*N-1)*L*L.)
    f_real: array-like
        Allocate a contiguous (row-major) numpy
        double array (at least len = (2*L) * (L+1) * (2*N-1).)
    parameters: fully populated so3_parameters_t object
    '''

    return libiw._so3_core_inverse_via_ssht_real(
                        f_real, flmn, parameters)

libiw._ssht_core_mw_lb_inverse_sov_sym.restype = None
libiw._ssht_core_mw_lb_inverse_sov_sym.argtypes = [complex_double,
                          complex_double, ct.c_int, ct.c_int,
                          ct.c_int, ssht_dl_method_t, ct.c_int]

def mw_lb_inverse_sov_sym(f, flm, L, spin, dl_method, verbosity=0, L0=0):
    '''
    Wrapper for ssht inverse spin-weighted SH transform.
    Assumes f is complex-valued. MW sampling.

    Arguments
    ---------
    f: array-like
        Allocate a contiguous (row-major) numpy
        complex double array (size = L*(2*L-1) )
    flm: array-like
        Allocate a contiguous (row-major) numpy
        complex128 array (size = L*L)
    L : int
        Max harmonic limit. Note that L = lmax + 1.
    spin: int
        spin
    dl_method : ssht_dl_method_t object
        Recursion method for dl, 0 for Risbo,
        1 for Trapani & Navaza. Trapani is 20%
        faster (but not stable anymore above L~2048,
        use Risbo in that case).
    verbosity : int
        verbisity level in range [0-5].
    L0 : int
        Lower harmonic limit
    '''

    return libiw._ssht_core_mw_lb_inverse_sov_sym(
                 f, flm, L0, L, spin, dl_method, verbosity)

libiw._ssht_core_mw_lb_inverse_sov_sym_real.restype = None
libiw._ssht_core_mw_lb_inverse_sov_sym_real.argtypes = [c_double,
                          complex_double, ct.c_int, ct.c_int,
                          ssht_dl_method_t, ct.c_int]

def mw_lb_inverse_sov_sym_real(f, flm, L, dl_method, verbosity=0, L0=0):
    '''
    Wrapper for ssht inverse scalar SH transform.
    Assumes f is real-valued. MW sampling.

    Arguments
    ---------
    f: array-like
        Allocate a contiguous (row-major) numpy
        double array (size = L*(2*L-1) )
    flm: array-like
        Allocate a contiguous (row-major) numpy
        complex128 array (size = L*L)
    L : int
        Max harmonic limit. Note that L = lmax + 1.
    dl_method : ssht_dl_method_t object
        Recursion method for dl, 0 for Risbo,
        1 for Trapani & Navaza. Trapani is 20%
        faster (but not stable anymore above L~2048,
        use Risbo in that case).
    verbosity : int
        verbisity level in range [0-5].
    L0 : int
        Lower harmonic limit
    '''

    return libiw._ssht_core_mw_lb_inverse_sov_sym_real(
                 f, flm, L0, L, dl_method, verbosity)


libiw._ssht_core_mwdirect_inverse.restype = None
libiw._ssht_core_mwdirect_inverse.argtypes = [complex_double,
                          complex_double, ct.c_int, ct.c_int,
                          ct.c_int]

def mwdirect_inverse(f, flm, L, spin, verbosity=0):
    '''
    Wrapper for ssht inverse spin-weighted SH transform.
    Assumes f is complex-valued. MW sampling.

    Arguments
    ---------
    f: array-like
        Allocate a contiguous (row-major) numpy
        double array (size = L*(2*L-1) )
    flm: array-like
        Allocate a contiguous (row-major) numpy
        complex128 array (size = L*L)
    L : int
        Max harmonic limit. Note that L = lmax + 1.
    spin: int
        spin
    verbosity : int
        verbisity level in range [0-5].
    '''

    return libiw._ssht_core_mwdirect_inverse(f, flm, L, spin,
                                             verbosity)

libiw._ssht_core_mwdirect_inverse_sov.restype = None
libiw._ssht_core_mwdirect_inverse_sov.argtypes = [complex_double,
                          complex_double, ct.c_int, ct.c_int,
                          ct.c_int]

def mwdirect_inverse_sov(f, flm, L, spin, verbosity=0):
    '''
    Wrapper for ssht inverse spin-weighted SH transform.
    Assumes f is complex-valued. MW sampling.

    Arguments
    ---------
    f: array-like
        Allocate a contiguous (row-major) numpy
        double array (size = L*(2*L-1) )
    flm: array-like
        Allocate a contiguous (row-major) numpy
        complex128 array (size = L*L)
    L : int
        Max harmonic limit. Note that L = lmax + 1.
    spin: int
        spin
    verbosity : int
        verbisity level in range [0-5].
    '''

    return libiw._ssht_core_mwdirect_inverse_sov(f, flm, L, spin,
                                             verbosity)

libiw._ssht_dl_beta_risbo_full_table.restype = None
libiw._ssht_dl_beta_risbo_full_table.argtypes = [c_double, ct.c_double,
                                                 ct.c_int, ssht_dl_size_t,
                                                 ct.c_int, c_double]

def dl_beta_risbo_full_table(dl, beta, L, dl_size, el):
    '''
    Calculates (for m = -l:l and mm = -l:l) lth plane of a
    d-matrix for argument beta using Risbo's recursion method.  For
    l>0, require the dl plane to be computed already with values for
    l-1.  Also takes a table of precomputed square roots of integers to
    avoid recomputing them.
    
    Aguments
    --------    
    '''

    sqrt_tbl = np.sqrt(np.arange(2 * el + 1))
    
    return libiw._ssht_dl_beta_risbo_full_table(dl, beta, L,
                         dl_size, el, sqrt_tbl)

libiw._so3_sampling_ind2elmn.restype = None
libiw._so3_sampling_ind2elmn.argtypes = [ct.POINTER(ct.c_int),
                       ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
                       ct.c_int, so3_parameters_t_p]

def ind2elmn(ind, parameters):
    '''
    Arguments
    ---------
    ind, int
        Index of flmn array.
    parameters: so3_parameters_t object

    Returns
    -------
    el, m, n, ints
        SO3 coefficients
    '''

    el = ct.c_int()
    m = ct.c_int()
    n = ct.c_int()

    libiw._so3_sampling_ind2elmn(el, m, n, ind, parameters)
    return el.value, m.value, n.value

libiw._so3_sampling_ind2elmn_real.restype = None
libiw._so3_sampling_ind2elmn_real.argtypes = [ct.POINTER(ct.c_int),
                       ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
                       ct.c_int, so3_parameters_t_p]

def ind2elmn_real(ind, parameters):
    '''
    Arguments
    ---------
    ind, int
        Index of flmn array.
    parameters: so3_parameters_t object

    Returns
    -------
    el, m, n : int
        SO3 coefficients
    '''

    el = ct.c_int()
    m = ct.c_int()
    n = ct.c_int()

    libiw._so3_sampling_ind2elmn_real(el, m, n, ind,
                                      parameters)
    return el.value, m.value, n.value

libiw._so3_sampling_elmn2ind.restype = None
libiw._so3_sampling_elmn2ind.argtypes = [ct.POINTER(ct.c_int),
                                         ct.c_int, ct.c_int, ct.c_int,
                                         so3_parameters_t_p]

def elmn2ind(el, m, n, parameters):
    '''
    Arguments
    -------
    el, m, n : int
        SO3 coefficients
    parameters: so3_parameters_t object

    Returns
    ---------
    ind, int
        Index of flmn array.
    '''

    ind = ct.c_int()
    libiw._so3_sampling_elmn2ind(ind, el, m, n, 
                                 parameters)
    
    return ind.value

libiw._so3_sampling_elmn2ind_real.restype = None
libiw._so3_sampling_elmn2ind_real.argtypes = [ct.POINTER(ct.c_int),
                                         ct.c_int, ct.c_int, ct.c_int,
                                         so3_parameters_t_p]

def elmn2ind_real(el, m, n, parameters):
    '''
    Arguments
    -------
    el, m, n : int
        SO3 coefficients
    parameters: so3_parameters_t object

    Returns
    ---------
    ind, int
        Index of flmn array.
    '''

    ind = ct.c_int()
    libiw._so3_sampling_elmn2ind_real(ind, el, m, n, 
                                 parameters)
    
    return ind.value

libiw._so3_sampling_f_size.restype = ct.c_int
libiw._so3_sampling_f_size.argtypes = [so3_parameters_t_p]

def f_size(parameters):
    '''
    Arguments
    ---------
    parameters: so3_parameters_t object

    Returns
    -------
    f_size: int
        Size of np.double array to be allocated
    '''
    return libiw._so3_sampling_f_size(parameters)

libiw._so3_sampling_n.restype = ct.c_int
libiw._so3_sampling_n.argtypes = [so3_parameters_t_p]

def n(parameters):
    '''
    Get the number of samples describing S3.
    I don't really trust this function..

    Arguments
    ---------
    parameters: so3_parameters_t object

    Returns
    -------
    n: int
        Amount of samples on 3-sphere not counting degenerate
        angles.
    '''
    return libiw._so3_sampling_n(parameters)

libiw._so3_sampling_nalpha.restype = ct.c_int
libiw._so3_sampling_nalpha.argtypes = [so3_parameters_t_p]

def nalpha(parameters):
    '''
    Arguments
    ---------
    parameters: so3_parameters_t object

    Returns
    -------
    nalpha: int
        Amount of alpha samples on 3-sphere (inc. degenerate
        angles). So number of alpha indices in f array.
    '''
    return libiw._so3_sampling_nalpha(parameters)

libiw._so3_sampling_nbeta.restype = ct.c_int
libiw._so3_sampling_nbeta.argtypes = [so3_parameters_t_p]

def nbeta(parameters):
    '''
    Arguments
    ---------
    parameters: so3_parameters_t object

    Returns
    -------
    nbeta: int
        Amount of beta samples on 3-sphere (inc. degenerate
        angles). So number of beta indices in f array.
    '''
    return libiw._so3_sampling_nbeta(parameters)

libiw._so3_sampling_ngamma.restype = ct.c_int
libiw._so3_sampling_ngamma.argtypes = [so3_parameters_t_p]

def ngamma(parameters):
    '''
    Arguments
    ---------
    parameters: so3_parameters_t object

    Returns
    -------
    ngamma: int
        Amount of gamma samples on 3-sphere (inc. degenerate
        angles). So number of gamma indices in f array.
    '''
    return libiw._so3_sampling_ngamma(parameters)


libiw._so3_sampling_a2alpha.restype = ct.c_double
libiw._so3_sampling_a2alpha.argtypes = [ct.c_int,
                                        so3_parameters_t_p]

def a2alpha(a, parameters):
    '''
    Arguments
    ---------
    a: int
        Alpha index in f.
    parameters: so3_parameters_t object

    Returns
    -------
    alpha: float
        Angle in [0, 2pi)
    '''
    return libiw._so3_sampling_a2alpha(a, parameters)

libiw._so3_sampling_b2beta.restype = ct.c_double
libiw._so3_sampling_b2beta.argtypes = [ct.c_int,
                                        so3_parameters_t_p]

def b2beta(b, parameters):
    '''
    Arguments
    ---------
    b: int
        Beta index in f.
    parameters: so3_parameters_t object

    Returns
    -------
    beta: float
        Angle in (0, pi] if sampling is MW, otherwise
        [0, pi].
    '''
    return libiw._so3_sampling_b2beta(b, parameters)

libiw._so3_sampling_g2gamma.restype = ct.c_double
libiw._so3_sampling_g2gamma.argtypes = [ct.c_int,
                                        so3_parameters_t_p]

def g2gamma(g, parameters):
    '''
    Arguments
    ---------
    g: int
        Gamma index in f.
    parameters: so3_parameters_t object

    Returns
    -------
    gamma: float
        Angle in [0, 2pi)
    '''
    return libiw._so3_sampling_g2gamma(g, parameters)

libiw._so3_sampling_flmn_size.restype = ct.c_int
libiw._so3_sampling_flmn_size.argtypes = [so3_parameters_t_p]

def flmn_size(parameters):
    '''
    Arguments
    ---------
    parameters: so3_parameters_t object

    Returns
    -------
    flmn_size: int
        Size of complex128 array to be allocated
    '''
    return libiw._so3_sampling_flmn_size(parameters)




