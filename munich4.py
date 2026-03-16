import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('Loading modules...')
import os, copy
import numpy as np
import healpy as hp
import qpoint as qp
from beamconv import ScanStrategy, Beam, tools, plot_tools
print('..done')


matplotlib.rcParams.update({'font.size':10})
matplotlib.rcParams.update({'xtick.direction':'in'})
matplotlib.rcParams.update({'ytick.direction':'in'})
matplotlib.rcParams.update({'xtick.top':True})
matplotlib.rcParams.update({'ytick.right':True})
matplotlib.rcParams.update({'legend.fontsize':8})
lfs = 14
hdl = 2

opj = os.path.join
nside = 128
dpi = 300

cw = [np.array([78., 121., 165.])/255.,
np.array([241., 143., 59.])/255.,
np.array([224., 88., 91.])/255.,
np.array([119., 183., 178.])/255.,
np.array([90., 161., 85.])/255.,
np.array([237., 201., 88.])/255.,
np.array([175., 122., 160.])/255.,
np.array([254., 158., 168.])/255.,
np.array([156., 117., 97.])/255.,
np.array([186., 176., 172.])/255.]

M = qp.QMap()
M.init_dest(nside=nside, pol=False, reset=True)


# def get_cls(no_B=False, no_T=False):

#     wmap_dir = '/Users/jon/git_repos/beamconv/ancillary/'
#     fname='wmap7_r0p03_lensed_uK_ext.txt'

#     cls = np.loadtxt(opj(wmap_dir, fname), unpack=True) # Cl in uK^2
#     cls[3] *= 1.25
#     print(cls[2][:50])
#     if no_B:
#         print('NO B')
#         data = np.loadtxt('lensing.txt', unpack=True)
#         lmax = len(cls[0])
#         cls[3][1:] = data[1][:lmax-1]
#         cls[3][0] = 0.0

#     if no_T:
#         cls[1][:] = 0.0        

#     print(cls[2][:50])
#     print(np.shape(cls))

#     return cls[0], cls[1:]

def short_ctime(**kwargs):

    start = kwargs.pop('start')
    end = kwargs.pop('end')
    cidx = kwargs.pop('cidx')

    ctime = np.linspace(0, 10., 1000)

    return ctime

def get_theta(nsamp=1000):
    
    # return np.linspace(0, 4 * np.pi / 6., nsamp)
    return np.pi / 2.0 * np.ones(nsamp)

def get_phi(nsamp=1000):
    
    # return np.linspace(0, np.pi/6., nsamp)
    return np.zeros(nsamp)

def short_scan(nsamp=1000, **kwargs):

    theta = get_theta(nsamp)
    phi = get_phi(nsamp)

    # theta = np.zeros(1000)
    # phi = np.zeros(1000)

    vec = hp.ang2vec(theta, phi)
    # Ra and Dec for qpoint
    ra, dec = hp.vec2ang(vec, lonlat=True)

    plt.plot(theta)
    plt.savefig(opj('img/', 'theta.png'))
    plt.close()

    # pa = np.linspace(0, 360, 1000)
    pa = np.zeros_like(phi)

    q_bore = M.radecpa2quat(ra, dec, pa)

    return q_bore

def scan2(use_dust=True, lmax=300, tag='',
    no_T=False, no_E=False, no_B=False, cmap='RdBu_r'):

    nside_spin  = 128
    mmax        = 4
    mlen        = 10
    sample_rate = 100.0
    nside_out   = 128
    verbose     = 1
    ctime0      = 1510000000

    # fwhm = 19.03863
    fwhm = 60.

    # ell, cls = get_cls(no_B=False)

    np.random.seed(42)
    
    # alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK
    # alm = np.asarray(alm)

    alm = np.zeros((3, int(lmax*(lmax+1)/2 + lmax +1))) + 0j

    idx1 = hp.Alm.getidx(lmax, 1, -1)
    idx2 = hp.Alm.getidx(lmax, 1, 0)
    idx3 = hp.Alm.getidx(lmax, 1, 1)
    idx4 = hp.Alm.getidx(lmax, 2, -2)
    idx5 = hp.Alm.getidx(lmax, 2, -1)
    idx6 = hp.Alm.getidx(lmax, 2, 0)
    idx7 = hp.Alm.getidx(lmax, 2, 1)
    idx8 = hp.Alm.getidx(lmax, 2, 2)

    print('Idx 1: {}'.format(idx1))
    print('Idx 2: {}'.format(idx2))
    print('Idx 3: {}'.format(idx3))    
    print('Idx 4: {}'.format(idx4))
    print('Idx 5: {}'.format(idx5))
    print('Idx 6: {}'.format(idx6))
    print('Idx 7: {}'.format(idx7))
    print('Idx 8: {}'.format(idx8))

    alm[0][:] = 0.0
    alm[0][0] = 1.0
    alm[1][2] = 1.0
    alm[2][2] = 0.0
    alm[1][3:] = 0.0
    alm[2][3:] = 0.0

    # print(np.shape(alm))
    # print(np.shape(alm2))

    # print(alm[0][:])
    # print(alm2[0][:])
    # print(alm[0][0])
    # print(alm2[0][0])


    maps_raw = hp.alm2map(alm, 256)
    
    sym_limits = [250, 0.5, 0.5]
    limits = [[0, 0], [-0.4, 0.0], [-0.4, 0]]


    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_over(cmap(1.0), 1.0)
    cmap.set_under(cmap(0.0), 1.0)


    theta = np.linspace(0, 4 * np.pi / 6., 1000)
    phi = np.linspace(0, np.pi/6., 1000)

    # theta = np.zeros(1000)
    # phi = np.zeros(1000)
    
    nsamp =1000
    theta = get_theta(nsamp)
    phi = get_phi(nsamp)

    vec = hp.ang2vec(theta, phi)
    ra, dec = hp.vec2ang(vec, lonlat=True)

    plt.plot(theta)
    plt.savefig(opj('img/', 'theta.png'))
    plt.close()

    plt.plot(phi)
    plt.savefig(opj('img/', 'phi.png'))
    plt.close()

    plt.plot(ra)
    plt.savefig(opj('img/', 'ra.png'))
    plt.close()

    plt.plot(dec)
    plt.savefig(opj('img/', 'dec.png'))
    plt.close()

    print('Maximum value in map: {:.3f} uK'.format(1e6 * np.max(maps_raw[1].flatten())))

    # plot_tools.plot_iqu(maps_raw, 'img/', 'input_spher'+tag,
    #     plot_func=projview,
    #     sym_limits=[250, 3, 3], cbar=False, #no_limits=True,
    #     # limits=[[-50, 200], [-10, 40], [-7, 7]], cbar=False,
    #     projection='nsper', lon_0=0, lat_0=40, pmin=0.5, pmax=99.5,
    #     cmap=cmap, bgcolor='white', unit='Signal [uK]', cbar_extend='both',
    #     diverging=False, dpi=300, transparent=True)

    print('setting up')
    sat = ScanStrategy(mlen, external_pointing=True, sample_rate=sample_rate,
        location='space', ctime0=ctime0, nside_out=nside_out)

    scan_opts = dict(q_bore_func=short_scan,
        ctime_func=short_ctime,
        q_bore_kwargs=dict(jitter_amp=0.),
        ctime_kwargs=dict(),
        max_spin=mmax,
        nside_spin=nside_spin,
        verbose=verbose,
        binning=False,
        interp=True)

    ############### Gaussian scanning
    sat.create_focal_plane(nrow=1, ncol=1, fov=0,
        fwhm=fwhm, lmax=lmax, mmax=mmax, amplitude=1.,
        no_pairs=True)

    ############### Continuous ideal HWP
    hfreq = 0.1
    varphi = 0.0
    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.create_focal_plane(nrow=1, ncol=1, fov=0,
        fwhm=fwhm, lmax=lmax, mmax=mmax, amplitude=1.,
        no_pairs=True, combine=False)

    print('scanning with Gaussian w/ideal HWP')
    #print(sat.hwp_dict)

    mueller = np.diag([1., 1., -1., -1.])
#    mueller = np.random.uniform(-1,1,size=(4,4))

    # mueller = np.diag([0, 0, 0, 0.])
    for beami in sat.beams:
        print(beami[0])
        print(type(beami[0]))
        beami[0].hwp_mueller = mueller
        beami[1].hwp_mueller = mueller

    sat.scan_instrument_mpi(alm, mu_con_hwp=True, mu_con_spin=True, **scan_opts)
    gaussian_tod_whwp = sat.tod.copy()
    hwp_ang = sat.hwp_ang.copy()
    hwp_ang4 = 4 * 180 * hwp_ang / np.pi
    hwp_ang4 = hwp_ang4 % 360

    hwp_ang2 = 2 * 180 * hwp_ang / np.pi
    hwp_ang2 = hwp_ang2 % 360

    hwp_ang = 180 * hwp_ang / np.pi

    from scipy.signal import argrelextrema
    idx2 = argrelextrema(hwp_ang2, np.less)[0]
    idx4 = argrelextrema(hwp_ang4, np.less)[0]

    plt.plot(hwp_ang)
    plt.plot(hwp_ang2)
    plt.plot(hwp_ang4)
    plt.savefig(opj('img/', 'hwp_ang.png'))
    plt.close()

    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.scan_instrument_mpi(alm, mu_con_hwp=True, mu_con_spin=False, **scan_opts)
    gaussian_tod_whwp_nospin = sat.tod.copy()

    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.scan_instrument_mpi(alm, mu_con_hwp=False, mu_con_spin=False, **scan_opts)
    gaussian_tod_whwp_orig = sat.tod.copy()


    #sat.scan_instrument_mpi(alm_noB, **scan_opts)
    #gaussian_tod_whwp_noB = sat.tod.copy()

    ############## 1BR HWP
    hwp_model='1BR'
    sat.create_focal_plane(nrow=1, ncol=1, fov=0,
        fwhm=fwhm, lmax=lmax, mmax=mmax, amplitude=1.,
        no_pairs=True, combine=False)

    for beami in sat.beams:
        print(beami[0])
        print(type(beami[0]))
        beami[0].set_hwp_mueller(model_name=hwp_model)
        beami[1].set_hwp_mueller(model_name=hwp_model)

    print('scanning with Gaussian')
    #print(sat.hwp_dict)
    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.scan_instrument_mpi(alm, mu_con_hwp=True, mu_con_spin=True, **scan_opts)
    gaussian_tod_w1BR = sat.tod.copy()

    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.scan_instrument_mpi(alm, mu_con_hwp=True, mu_con_spin=False, **scan_opts)
    gaussian_tod_w1BR_nospin = sat.tod.copy()

    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.scan_instrument_mpi(alm, mu_con_hwp=False, mu_con_spin=False, **scan_opts)
    gaussian_tod_w1BR_orig = sat.tod.copy()

    ############## 3BR HWP
    hwp_model='3BR'
    sat.create_focal_plane(nrow=1, ncol=1, fov=0,
        fwhm=fwhm, lmax=lmax, mmax=mmax, amplitude=1.,
        no_pairs=True, combine=False)

    for beami in sat.beams:
        print(beami[0])
        print(type(beami[0]))
        beami[0].set_hwp_mueller(model_name=hwp_model)
        beami[1].set_hwp_mueller(model_name=hwp_model)

    print('scanning with Gaussian')   
    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.scan_instrument_mpi(alm, mu_con_hwp=True, mu_con_spin=True, **scan_opts)
    gaussian_tod_w3BR = sat.tod.copy()    

    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.scan_instrument_mpi(alm, mu_con_hwp=True, mu_con_spin=False, **scan_opts)
    gaussian_tod_w3BR_nospin = sat.tod.copy()    

    sat.set_hwp_mod(mode='continuous', freq=hfreq, varphi=varphi)
    sat.scan_instrument_mpi(alm, mu_con_hwp=False, mu_con_spin=False, **scan_opts)
    gaussian_tod_w3BR_orig = sat.tod.copy()   

    ############## Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

    print('plotting')

    ax1.plot(gaussian_tod_whwp, color=cw[0], label='Gaussian')
    ax1.plot(gaussian_tod_whwp_nospin, color=cw[0], ls='--', label='Gaussian (no spin)')
    ax1.plot(gaussian_tod_whwp_orig, color=cw[0], ls=':', label='Gaussian (orig)')
    ax1.plot(gaussian_tod_w1BR, color=cw[1], label='1BR')
    ax1.plot(gaussian_tod_w1BR_nospin, color=cw[1], ls='--', label='1BR (no spin)')
    ax1.plot(gaussian_tod_w1BR_orig, color=cw[1], ls=':', label='1BR (orig)')
    ax1.plot(gaussian_tod_w3BR, color=cw[2], label='3BR')
    ax1.plot(gaussian_tod_w3BR_nospin, color=cw[2], ls='--', label='3BR (no spin)')
    ax1.plot(gaussian_tod_w3BR_orig, color=cw[2], ls=':', label='3BR (orig)')

    # print('RMS amplitude (Gaussian) {:.6f}'.format(np.std(gaussian_tod)))
    # print('RMS amplitude (Gaussian noB) {:.6f}'.format(np.std(gaussian_tod_noB)))
    # print('RMS amplitude (EG) {:.6f}'.format(np.std(eg_tod)))
    # print('RMS amplitude (PO) {:.6f}'.format(np.std(po_tod)))

    ax2.plot(gaussian_tod_whwp - gaussian_tod_w1BR, color=cw[0], 
        label='Gaussian w/HWP - Gaussian w/HWP 1BR')
    ax2.plot(gaussian_tod_whwp - gaussian_tod_w3BR, color=cw[1], 
        label='Gaussian w/HWP - Gaussian w/HWP 3BR')

    ax3.plot(gaussian_tod_whwp - gaussian_tod_whwp_nospin, color=cw[0], 
        label='Ideal: Both - No spin')
    ax3.plot(gaussian_tod_whwp - gaussian_tod_whwp_orig, color=cw[0], ls='--',
        label='Ideal: Both - Orig')

    ax3.plot(gaussian_tod_w1BR - gaussian_tod_w1BR_nospin, color=cw[1], 
        label='1BR: Both - No spin')
    ax3.plot(gaussian_tod_w1BR - gaussian_tod_w1BR_orig, color=cw[1], ls='--',
        label='1BR: Both - Orig')

    ax3.plot(gaussian_tod_w3BR - gaussian_tod_w3BR_nospin, color=cw[2], 
        label='3BR: Both - No spin')
    ax3.plot(gaussian_tod_w3BR - gaussian_tod_w3BR_orig, color=cw[2], ls='--',
        label='3BR: Both - Orig')

    for idx in idx4:        

        ax3.axvline(x=idx, color='black', alpha=0.3)

    ax1.legend(borderaxespad=0.1, ncol=3, loc=2,
        facecolor='white', edgecolor='None', framealpha=0,
        handlelength=3, fontsize=8)
    ax2.legend(borderaxespad=0.1, ncol=3, loc=2,
        facecolor='white', edgecolor='None', framealpha=0,
        handlelength=3, fontsize=8)
    ax3.legend(borderaxespad=0.1, ncol=3, loc=2,
        facecolor='white', edgecolor='None', framealpha=0,
        handlelength=3, fontsize=8)

    ax1.set_ylabel('Signal [uK]')
    ax2.set_ylabel('Signal [uK]')
    ax3.set_ylabel('Signal [uK]')
    ax2.set_xlabel('Sample number')
    plt.savefig(opj('img/','hwp_tod.png'),
        bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':

    scan2(tag='munich', cmap='RdBu_r')



