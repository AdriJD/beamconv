import os
import pysm
import pickle
from pysm.nominal import models
from beamconv import Beam, ScanStrategy, tools
# pysm.nominal contains all the pre defined models for foregrounds and cmb.
# models [d0,..,d8] correspond to dust polarization, [s0,..s3]
# to synchrotron emission and c1 to CMB.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
opj = os.path.join

dir_out = '../output/'
dir_inp = '../input_maps/'
dir_ideal = '../ideal_case/output/'
blm_dir = '../beams_jon/'

opj=os.path.join

def generate_maps(nside, Freq):

    '''
    Generate sky maps in healpix-format trough pySM and 
    store the in 'fits' format in the directory 'pysm_maps'
    Create the sky configuration as a dictionary of dictionaries, corresponding to
    the components we want to include. 
    One can consider only cmb, only dust, only synchrotron or cmb+dust+synchrotron. 
    For each one components one can choose different models that describe it. 

   ---------
    nside: int
        the nside of the map
    Freq: array-like 
        frequency at which one want to compute the sky

    Returns
    -------
    store the maps on file format '.fits' in directory 'pysm_maps'

    '''

    # dust
    d0_config = models("d0", nside)
    d1_config = models("d1", nside)
    d2_config = models("d2", nside)
    d3_config = models("d3", nside)
    d4_config = models("d4", nside)
    d5_config = models("d5", nside)
    d6_config = models("d6", nside)
    d7_config = models("d7", nside)
    d8_config = models("d8", nside)

    # synchrotron
    s0_config = models("s0", nside)
    s1_config = models("s1", nside)
    s2_config = models("s2", nside)
    s3_config = models("s3", nside)

    # cmb
    c1_config = models("c1", nside)

    # create the sky configuration as a dictionary of dictionaries, corresponding to
    # the components we want to include the 4 cases below to sky maps consisted of:
    # only cmb, only dust, only synchrotron, cmb+dust+synchrotron.
    # example models d5,d3.
    # for CMB we always use c1.

    cmb_config = {'cmb' : c1_config}
    dust_config = {'dust' : d5_config}
    sync_config= {'synchrotron' : s3_config}
    sky_config = {'dust' : d5_config, 'synchrotron' : s3_config, 'cmb': c1_config}
   
    # create the sky object; model the sky signal of the galactic foregrounds.
    sky_cmb = pysm.Sky(cmb_config)
    sky_dust = pysm.Sky(dust_config)
    sky_sync = pysm.Sky(sync_config)
    sky_total = pysm.Sky(sky_config)

    # Then we create and store the maps for the frequency/frequencies.
    filefolder = opj(dir_inp, 'pysm_maps/')
    for k in Freq:
       
        hp.write_map((opj(filefolder, 'CMB_'+str(k)+'GHz.fits')), sky_cmb.signal()(k),
            overwrite=True)
        hp.write_map((opj(filefolder, 'Dust_'+str(k)+'GHz.fits')), sky_dust.signal()(k),
            overwrite=True)
        hp.write_map((opj(filefolder, 'Sync_'+str(k)+'GHz.fits')), sky_sync.signal()(k),
            overwrite=True)
        hp.write_map((opj(filefolder, 'Total_'+str(k)+'GHz.fits')), sky_total.signal()(k),
            overwrite=True)


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

def map2alm(map_file, alm_file, lmax=2000):
    
    '''
    Precalculate alms from input maps and store in
    same directory. Returns alms.
    Arguments
    ---------    
    map_file : path
    alm_file : path
    Keyword arguments
    -----------------
    lmax : int
        Bandlimit for storage (default : 2000)

       Returns
    -------
    alm
    '''

    input_map = hp.read_map(map_file, field=[0, 1, 2])
    print('Calculating alms: {}'.format(alm_file))

    # use a high lmax internally to mimize aliassing, we truncate the
    # alms afterwards


    lmax_big = 2 * hp.get_nside(input_map[0])
    lmax_big = lmax if lmax_big <= lmax else lmax_big

    alm = hp.map2alm(input_map, lmax=lmax_big)
    print('...done calculating alms')
    alm = trunc_alm(alm, lmax_new=lmax)

    np.save(alm_file, alm)

    return alm


def calc_alms(lmax,Freq):

    '''
    Precalculate alms from input maps and store in
    the directory 'alms'.

    Arguments
    lmax: int
        max value of the multipole
    Freq: array-like 
        frequency at which one want to compute the sky
       
    '''
    
    filefolder1 = opj(dir_inp, 'pysm_maps/')
    filefolder2 = opj(dir_out,'alms/')
    for freq in Freq:
        
        Map=hp.read_map(os.path.join(filefolder1,'CMB_'+str(freq)+'GHz.fits'),field=(0,1,2))
        T=hp.map2alm(Map,lmax)
        print(np.shape(T))
        np.save(os.path.join(filefolder2, 'alm_'+str(freq)+'GHz.npy'),T)
         

def Scan_maps(nside,alms, lmax,Freq, Own_Stack = True, ideal_hwp=False):

    '''
    Scanning simulation, store the output map, the blm, the tod, and the 
    condition number in the output directory.
    ---------
    nside: int
        the nside of the map
    alms : array-like
        array of alm arrays that
        share lmax and mmax. For each frequency we have three healpy alm array
    lmax: int
        The bandlmit.
    Freq: array-like 
        frequency at which one want to compute the sky

    Keyword arguments
    -----------------
    ideal_hwp : bool
        If True: it is considered an ideal HWP,
        if Flase: it is considered a real HWP.
        (default : False)
    '''
    
    po_file = opj(blm_dir, 'pix0000_90_hwpproj_v5_f1p6_6p0mm_mfreq_lineard_hdpe.npy')
    #eg_file = opj(blm_dir, 'blm_hp_eg_X1T1R1C8A_800_800.npy')
    beam_file = 'pix0000_90_hwpproj_v5_f1p6_6p0mm_mfreq_lineard_hdpe.pkl'
    
    hwp = Beam().hwp()

    if Own_Stack:
    	thicknesses=np.array([0.427, 4.930, 0.427])
    	indices = np.array([[1.,1.],[1.02,1.02],[1.,1.]])
    	losses = np.array([[0.,0.],[1e-4,1e-4],[0.,0.]])
    	angles = np.array([0.,0.,0.])
    	hwp.stack_builder(thicknesses=thicknesses, indices=indices, losses=losses, angles=angles)
    else:    
    	hwp.choose_HWP_model('SPIDER_95')
        
    beam_opts = dict(az=0,
                     el=0,
                     polang=0.,
                     btype='PO',
                     fwhm=32.2,
                     lmax=lmax,
                     mmax=4,
                     amplitude=1.,
                     po_file=po_file,
                     #eg_file=eg_file,
                     deconv_q=True,  # blm are SH coeff from hp.alm2map
                     normalize=True,
                     hwp=hwp)
    
    with open(beam_file, 'wb') as handle:
        pickle.dump(beam_opts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    beam=Beam(**beam_opts)

    ### SCAN OPTIONS
    # duration = nsamp/sample_rate
    nsamp      = 864000
    fsamp      = 10
    mlen       = nsamp/fsamp
    lmax       = lmax
    mmax       = 4
    ra0        = -10
    dec0       = -57.5
    az_throw   = 50
    scan_speed = 2.8
    rot_period = 4.5*60*60
    nside_spin = nside
    # scan the given sky and return the results as maps;
    
    
   
    filefolder = opj(dir_out, 'Output_maps/')
    for freq, alm in zip(Freq, alms):
        
        ss = ScanStrategy(mlen,sample_rate=fsamp, location='atacama')

        # Add the detectors to the focal plane; 9 detector pairs used in this case
        ss.create_focal_plane(nrow=4, ncol=4, fov=3,**beam_opts)

        # Use a half-wave plate. Other options one can use is to add an elevation
        # pattern or a periodic instrument rotation as, for example: 
        # ss.set_instr_rot(period=rot_period, angles=[68, 113, 248, 293])
        # and ss.set_el_steps(step_period, steps=[-4, -3, -2, -1, 0, 1, 2, 3, 4, 4])
        ss.set_hwp_mod(mode='continuous', freq=1.)
        # scan the given sky and return the results as maps;
        ss.allocate_maps(nside=nside)
        if ideal_hwp:
        	ss.scan_instrument_mpi(alm, verbose=1, ra0=ra0, dec0=dec0,
        		az_throw=az_throw,nside_spin=nside_spin, max_spin=mmax, binning=True, hwp_status='ideal')
        else:
        	ss.scan_instrument_mpi(alm, verbose=1, ra0=ra0, dec0=dec0,
        		az_throw=az_throw,nside_spin=nside_spin, max_spin=mmax, binning=True)

        tod = ss.tod
        maps, cond = ss.solve_for_map(fill = np.nan)
        blm = np.asarray(beam.blm).copy()

        # We need to divide out sqrt(4pi / (2 ell + 1)) to get 
        # correctly normlized spherical harmonic coeffients.
        ell = np.arange(hp.Alm.getlmax(blm[0].size))
        q_ell = np.sqrt(4. * np.pi / (2 * ell + 1))
        blm[0] = hp.almxfl(blm[0], 1 / q_ell)
        blm[1] = hp.almxfl(blm[1], 1 / q_ell)
        blm[2] = hp.almxfl(blm[2], 1 / q_ell)

        # print(np.shape(alm))
        # print(np.shape(alms))
        # print(np.shape(maps))
        # print(np.shape(cond))

        # maps = [maps[0], maps[1], maps[2]]

        hp.write_map(opj(dir_out, 'Output_maps/Bconv_'+str(freq)+'GHz.fits'), maps, overwrite=True)
        hp.write_map(opj(dir_out, 'Output_maps/Cond_Numb_'+str(freq)+'GHz.fits'), cond, overwrite=True)
        np.save(os.path.join(dir_out, 'blms/blm_'+str(freq)+'GHz.npy'), blm)
        np.save(os.path.join(dir_out, 'tods/tod_'+str(freq)+'GHz.npy'), tod)


def view_maps(output_maps,conds):
    
    '''
    Display the output maps and the relative condition number.
    ---------
    output_maps : array-like
        healpy maps array
    conds: array-like
        ondition number relative to the scan 

    '''
    Freq=[90, 95, 100, 105, 110]
    for maps,cond in zip(output_maps,conds):
        cond[cond == np.inf] = hp.UNSEEN
        cart_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]')
        hp.cartview(cond, **cart_opts)
        hp.cartview(maps[0], min=-250, max=250, **cart_opts)
        hp.cartview(maps[1], min=-5, max=5, **cart_opts)
        hp.cartview(maps[2], min=-5, max=5, **cart_opts)
        plt.show()



def residual(nside,alms, lmax, blms, Ideal_comp=True, Conv_comp= False, Smooth_comp=False, View_diffMap = False, Plot = False):

    '''
    Compute the APS of the residual maps which are the differnce maps between
    the output map and the input map convolved with the beam. 
    ---------
    nside: int
        the nside of the map
    alms : array-like
        array of alm arrays that
        share lmax and mmax. For each frequency we have three healpy alm array
    lmax: int
        The bandlmit.
    blms : array-like
        array of alm arrays that share lmax and mmax. Realtive to optical beam.
        For each frequency we have three healpy blm array
        
    Keyword arguments
    -----------------
    View_diffMap : bool
        if True: Display the residual maps
        (default : False)
    Plot : bool
        if True: shows the residual APS and store them in the output folder 
        (default : False)
    '''

    Freq=[90, 95, 100, 105, 110]
    for freq, alm, blm in zip(Freq, alms, blms):
        O_maps = hp.read_map(opj(dir_out, 'Output_maps/Bconv_'+str(freq)+'GHz.fits'), field=(0,1,2), verbose = False)
        In_maps= hp.read_map(opj(dir_inp, 'pysm_maps/CMB_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False)
        
        O_maps[np.isnan(O_maps)] = 0.
        In_maps[np.isnan(In_maps)] = 0.

        # ## COMPARISON wrt IDEAL CASE
        if Ideal_comp:
            Ideal_maps= hp.read_map(opj(dir_ideal, 'Output_maps/Bconv_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False)
            Ideal_maps[np.isnan(Ideal_maps)] = 0.
            cl_in = hp.anafast(Ideal_maps, lmax=lmax-1, mmax=4)
            res_maps_ring = O_maps - Ideal_maps
            res_maps_ring[np.isnan(res_maps_ring)] = 0.

        # ## COMPARISON wrt PYSM MAP CONVOLVED WITH OPTICAL BEAM
        elif Conv_comp:
            blmax= hp.Alm.getlmax(alm[0].size)
            blm = trunc_alm(blm, blmax)
            alm =alm*blm
            In_maps_sm = hp.alm2map(alm, nside= hp.get_nside(O_maps),  pol = True, verbose = False)
            cl_in = hp.anafast(In_maps_sm, lmax=lmax-1, mmax=4)
            hp.write_map(opj(dir_out, 'Smoot_IM/In_map_smoot_'+str(freq)+'GHz.fits'), In_maps_sm, overwrite=True)
            res_maps_ring = O_maps - In_maps_sm
            res_maps_ring[np.isnan(res_maps_ring)] = 0.

        # ## COMPARISON wrt PYSM MAP SMOOTHED WITH GAUSSIAN BEAM
        elif Smooth_comp:
            In_maps_gauss= hp.smoothing(In_maps, fwhm=np.radians(32.2/60), verbose=False)
            cl_in = hp.anafast(In_maps_gauss, lmax=lmax-1, mmax=4)
            hp.write_map(opj(dir_out, 'Gauss_IM/In_map_gauss_'+str(freq)+'GHz.fits'), In_maps_gauss, overwrite=True)
            res_maps_ring = O_maps - In_maps_gauss
            res_maps_ring[np.isnan(res_maps_ring)] = 0.

        hp.write_map(opj(dir_out, 'Res_Maps/res_maps_'+str(freq)+'GHz.fits'), res_maps_ring, overwrite=True)
        
        cl_res = hp.anafast(res_maps_ring, lmax=lmax-1, mmax=4) ## TT,EE,BB,TE,EB,TB 
        np.save(os.path.join(dir_out+'residual_spectra/Cell_'+str(freq)+'GHz.npy'), cl_res)

        ratio_cl_res=np.zeros((6,lmax))
        ratio_cl_res[:,2:] = (cl_res[:,2:lmax]/cl_in[:,2:lmax])*100 ### the first two multipoles in cl_in are zeros, so we will have a numerical problem..
        np.save(os.path.join(dir_out+'residual_spectra/perc_Cell_'+str(freq)+'GHz.npy'), ratio_cl_res)
        
        
        if View_diffMap:
            cart_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]')
            hp.cartview(res_maps_ring[0], min=-250, max=250, **cart_opts)
            hp.cartview(res_maps_ring[1], min=-5, max=5, **cart_opts)
            hp.cartview(res_maps_ring[2], min=-5, max=5, **cart_opts)
            plt.show()

        if Plot:
            plot_APS_residual(ratio_cl_res[0],ratio_cl_res[1],ratio_cl_res[2],dir_out+'residual_spectra/plots/Cell'+str(freq)+'GHz.pdf', r'$\nu = $'+str(freq)+'GHz')
            

def plot_APS_residual(cell1,cell2,cell3, file, name):
    plt.rcParams['axes.labelsize'] = 13
    fig = plt.figure ()
    ax = fig.add_subplot(1,1,1)
    l = len(cell1)
    vett_l = np.arange (0,l)
    ax.plot (vett_l, cell1 , 'b-', label = 'TT')
    ax.plot (vett_l, cell2 , 'r-', label = 'EE')
    #ax.plot (vett_l, cell3 , 'k-', label = 'BB')
    ax.set_ylabel (r'$\frac{C_{\ell ,res}}{C_{\ell ,inp}}(\%)$')
    ax.set_xlabel (r'$\ell$ ')
    ax.set_yscale('log')
    #ax.title(name)
    plt.grid()
    plt.legend()
    plt.savefig(file)
    plt.show()


def run(genmaps=True, calc_alm=True, scan_maps=True, comparison_test=False, Residual = True, See_Maps= False,
        nside=512, lmax = 700, Freq=[90, 95, 100, 105, 110]):

    '''
    It is the 'main'. 
    ---------
    nside: int
        the nside of the map
    lmax: int
        The bandlmit.
    Freq: array-like 
        frequency at which one want to compute the sky
        
    Keyword arguments
    -----------------
    genmaps: bool
        if True: generates the maps
        (default : True)
    calc_alm : bool
        if True: computes the alm from the input maps
        (default : True)
    scan_maps : bool
        if True: simulates a scan
        (default : True)
    comparison_test: bool
        if True: compares the input and the output maps
        (default : False)
    Residual : 
        if True: computes the residuals
        (default : True)
    See_maps : 
        if True: display the output maps and the relative condition number.
        (default : False)
    '''

    if genmaps:

        generate_maps(nside,Freq)
    pysm_maps=[]
    for freq in Freq:

        pysm_maps.append(hp.read_map(opj(dir_inp, 'pysm_maps/CMB_'+str(freq)+'GHz.fits')))

    if calc_alm:

        calc_alms(lmax,Freq)

    alms=[]
    for k in Freq:

        filefolder = opj(dir_out, 'alms/')
        alms.append(np.load(opj(filefolder, 'alm_'+str(k)+'GHz.npy')))
       
    print(alms)
    print(np.shape(alms))
    if scan_maps:

        Scan_maps(nside,alms,lmax,Freq=[90, 95, 100, 105, 110])

    output_maps=[]
    blms=[]
    conds=[]
    tods=[]
    for freq in Freq:

        output_maps.append(hp.read_map(opj(dir_out, 'Output_maps/Bconv_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False))
        conds.append(hp.read_map(opj(dir_out, 'Output_maps/Cond_Numb_'+str(freq)+'GHz.fits'), verbose = False))
        blms.append(np.load(opj(dir_out, 'blms/blm_'+str(k)+'GHz.npy')))
        tods.append(np.load(opj(dir_out, 'tods/tod_'+str(k)+'GHz.npy')))


    print(np.shape(output_maps))
    print(np.shape(blms))

    if See_Maps:
        view_maps(output_maps,conds)

    if Residual:
    	residual(nside,alms,lmax,blms)

    if comparison_test:
        Comparison_Test(pysm_maps,Output_maps)




def main():
    
    # We will expand on this function with argument parsers and more in the future
    run()
if __name__ == '__main__':
    
    main()


     
