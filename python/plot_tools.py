import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from warnings import catch_warnings, simplefilter
import matplotlib.gridspec as gridspec
import numpy as np
import healpy as hp

def plot_map(map_arr, write_dir, tag,
             plot_func=hp.mollview, **kwargs):
    '''
    Plot map using one of the healpy plotting 
    functions and write to disk.
    
    Arguments
    ---------
    map_arr : array-like
        Healpix map to plot
    write_dir : str
        Path to directory where map is saved
    tag : str
        Filename = <tag>.png
    
    Keyword arguments
    -----------------
    plot_func : <function>
        healpy plotting function (default : mollview)
    kwargs : <healpy_plot_opts>        
    '''
    
    filename = os.path.join(write_dir, tag)

    plt.figure()
    with catch_warnings(RuntimeWarning):
        simplefilter("ignore")
        plot_func(map_arr, **kwargs)
        plt.savefig(filename+'.png')
    plt.close()
    

def plot_iqu(maps, write_dir, tag,
             sym_limits=None, **kwargs):
    '''
    Plot a (set of I, Q, U) map(s) and write each
    to disk.
    
    Arguments
    ---------
    sym_limits : scalar, array-like
        Colorbar limits assuming symemtric limits.
        If array-like, assume limits for I, Q, U
        maps
    write_dir : str
        Path to directory where map is saved
    tag : str
        Filename = <tag>.png

    Keyword arguments
    -----------------
    kwargs : {plot_map_opts, healpy_plt_opts}
    '''

    if not hasattr(sym_limits, "__iter__"):
        sym_limits = [sym_limits] * 3

    for pidx, pol in enumerate(['I', 'Q', 'U']):

        maxx = kwargs.pop('max', sym_limits[pidx])
        try:
            minn = -maxx
        except TypeError:
            minn = maxx
            
        plot_map(maps[pidx], write_dir, tag+'_'+pol,
                min=minn, max=maxx, **kwargs)
