import os
import matplotlib
import matplotlib.pyplot as plt
from warnings import catch_warnings, filterwarnings
import numpy as np
import healpy as hp

def plot_map(map_arr, write_dir, tag,
    plot_func=hp.mollview, tight=False, dpi=150, transparent=False, **kwargs):
    '''
    Plot map using one of the healpy plotting
    functions and write to disk.

    Arguments
    ---------
    map_arr : array-like
        Healpix map to plot
    write_dir : str
        Path to directory where map is saved
    tight : bool
        call savefig with bbox_inches = 'tight'
    tag : str
        Filename = <tag>.png

    Keyword arguments
    -----------------
    plot_func : <function>
        healpy plotting function (default : mollview)
    kwargs : <healpy_plot_opts>
    '''

    bbox_inches = 'tight' if tight else None
    filename = os.path.join(write_dir, tag)

    plt.figure()
    with catch_warnings():
        filterwarnings('ignore', category=RuntimeWarning)

        plot_func(map_arr, **kwargs)
        plt.savefig(filename+'.png', bbox_inches=bbox_inches, dpi=dpi,
            transparent=transparent)
    plt.close()

def round_sig(x, sig=1):

    return np.round(x, sig-int(np.floor(np.log10(np.abs(x))))-1)

def plot_iqu(maps, write_dir, tag, plot_func=hp.mollview,
    no_limits=False, limits=None, sym_limits=None, symmetric=False,
    mask=None, tight=False, dpi=150, udicts=None, **kwargs):
    '''
    Plot a (set of I, Q, U) map(s) and write each to disk.
    If a list with 4 maps is provided, assume the fourth component is Stokes V.

    Arguments
    ---------
    sym_limits : scalar, array-like
        Colorbar limits assuming symemtric limits.
        If array-like, assume limits for I, Q, U
        maps
    tight : bool
        call savefig with bbox_inches = 'tight'
    write_dir : str
        Path to directory where map is saved
    tag : str
        Filename = <tag>_I/Q/U.png
    udicts : list
        a list of three dictionaries that are passed to i, q, and u map plotting
        tools respectively, these are combined with kwargs which are shared
        for all three plots
        default: None

    Keyword arguments
    -----------------
    kwargs : {plot_map_opts, healpy_plt_opts}
    '''


    dim1 = np.shape(maps)[0]
    stokes = ['I', 'Q', 'U']

    if dim1 != 3 and dim1 !=4:
        raise ValueError('maps should be a sequence of three or four arrays')

    limits_set = False
    if hasattr(sym_limits, "__iter__"):
        symmetric, limits_set = True, True

    if hasattr(limits, "__iter__"):
        limits_set = True
        symmetric = False

    elif (not hasattr(sym_limits, "__iter__")) and sym_limits is not None and symmetric:
        sym_limits = [sym_limits] * 3
        limits_set = True

    if udicts is None and dim1 == 3:
        udicts = [{}, {}, {}]
    elif udicts is None and dim1 == 4:
        udicts = [{}, {}, {}, {}]
        stokes.append('V')

    maxx, minn = None, None
    for pidx, (st, udict) in enumerate(zip(stokes, udicts)):

        if limits_set and symmetric:
            minn = -sym_limits[pidx]
            maxx = sym_limits[pidx]
        elif limits_set:
            minn = limits[pidx][0]
            maxx = limits[pidx][1]
        elif (maxx is None) and (minn is None):
            maxx = kwargs.pop('max', 10)
            minn = kwargs.pop('min', -10)

        zwargs = kwargs.copy()
        zwargs.update(udict)

        map2plot = np.copy(maps[pidx])

        if minn is None:
            minn = round_sig(np.nanmin(map2plot), sig=1)

        if maxx is None:
            maxx = round_sig(np.nanmax(map2plot), sig=1)

        if mask is not None:
            map2plot[~mask] = np.nan

        plot_func = zwargs.pop('plot_func', plot_func)
        if no_limits:
            plot_map(map2plot, write_dir, tag+'_'+st, plot_func=plot_func,
                tight=tight, **zwargs)
        else:
            plot_map(map2plot, write_dir, tag+'_'+st, plot_func=plot_func,
                min=minn, max=maxx, tight=tight, **zwargs)
