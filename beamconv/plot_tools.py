import os
import matplotlib
import matplotlib.pyplot as plt
from warnings import catch_warnings, filterwarnings
import numpy as np
import healpy as hp

def plot_map(map_arr, write_dir, tag,
             plot_func=hp.mollview, tight=False, **kwargs):
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
        plt.savefig(filename+'.png', bbox_inches=bbox_inches)
    plt.close()

def round_sig(x, sig=1):

    return np.round(x, sig-int(np.floor(np.log10(np.abs(x))))-1)

def plot_iqu(maps, write_dir, tag,
             sym_limits=None, mask=None, tight=False, **kwargs):
    '''
    Plot a (set of I, Q, U) map(s) and write each
    to disk.

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

    Keyword arguments
    -----------------
    kwargs : {plot_map_opts, healpy_plt_opts}
    '''

    dim1 = np.shape(maps)[0]
    if dim1 != 3:
        raise ValueError('maps should be a sequence of three arrays')

    if not hasattr(sym_limits, "__iter__"):
        sym_limits = [sym_limits] * 3

    for pidx, pol in enumerate(['I', 'Q', 'U']):

        maxx = kwargs.pop('max', sym_limits[pidx])
        try:
            minn = -maxx
        except TypeError:
            minn = maxx

        map2plot = np.copy(maps[pidx])
        
        if minn is None:
            minn = round_sig(np.nanmin(map2plot), sig=1)

        if maxx is None:
            maxx = round_sig(np.nanmax(map2plot), sig=1)

        if mask is not None:
            map2plot[~mask] = np.nan

        plot_map(map2plot, write_dir, tag+'_'+pol,
                min=minn, max=maxx,  tight=tight, **kwargs)
