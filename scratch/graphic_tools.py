import os
import pysm
import pickle
# from pysm.nominal import models
# from beamconv import Beam, ScanStrategy, tools

# pysm.nominal contains all the pre-defined models for foregrounds and cmb.
# models [d0,..,d8] correspond to dust polarization, [s0,..s3]
# to synchrotron emission and c1 to CMB.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
opj = os.path.join

dir_out = '../test7/output/'
dir_ideal = '../ideal_case/output/'
dir_inp = '../test7/input_maps/'
blm_dir = '../beams_jon/'



def view_maps(Maps, conds, Freq, dir_, file, Mollview=True, Cond=False):
    for maps, freq, cond in zip(Maps,Freq,conds):
    	if Mollview:
    		moll_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]', min=-200, max=+200)
    		hp.mollview(maps[0], **moll_opts)
    		plt.savefig(dir_+'MAPs_VIEW/mollview/'+file+'Tmap'+str(freq)+'GHz.pdf')
    		hp.mollview(maps[1], **moll_opts)
    		plt.savefig(dir_+'MAPs_VIEW/mollview/'+file+'Qmap'+str(freq)+'GHz.pdf')
    		hp.mollview(maps[2], **moll_opts)
    		plt.savefig(dir_+'MAPs_VIEW/mollview/'+file+'Umap'+str(freq)+'GHz.pdf')
    		if Cond:
    			cond[cond == np.inf] = hp.UNSEEN
    			hp.mollview(cond, **moll_opts)
    			plt.savefig(dir_+'MAPs_VIEW/mollview/'+file+'cond_numb'+str(freq)+'GHz.pdf')
    	else:
    		cart_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]',lonra=[60, 180], latra=[-90, -30], min=-200, max=+200)
    		hp.cartview(maps[0], **cart_opts)
    		plt.savefig(dir_+'MAPs_VIEW/cartview/'+file+'Tmap'+str(freq)+'GHz.pdf')
    		hp.cartview(maps[1], **cart_opts)
    		plt.savefig(dir_+'MAPs_VIEW/cartview/'+file+'Qmap'+str(freq)+'GHz.pdf')
    		hp.cartview(maps[2], **cart_opts)
    		plt.savefig(dir_+'MAPs_VIEW/cartview/'+file+'Umap'+str(freq)+'GHz.pdf')
    		if Cond:
    			cond[cond == np.inf] = hp.UNSEEN
    			hp.cartview(cond, **cart_opts)
    			plt.savefig(dir_+'MAPs_VIEW/cartview/'+file+'cond_numb'+str(freq)+'GHz.pdf')
        #plt.show()

def plot_APS_residual(cell1,cell2,cell3, file, name):
    '''
    Display the residual of the diff maps and the relative condition number.
    ---------
    '''
    plt.rcParams['axes.labelsize'] = 14
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
    #plt.show()

def residual(Freq):
    lmax = 700
    for freq in Freq:
        cl_res = np.load(opj(dir_out, 'residual_spectra/Cell_'+str(freq)+'GHz.npy'))
        ratio_cl_res = np.load(opj(dir_out, 'residual_spectra/perc_Cell_'+str(freq)+'GHz.npy'))
        plot_APS_residual(ratio_cl_res[0],ratio_cl_res[1],ratio_cl_res[2],dir_out+'residual_spectra/plots/Cell'+str(freq)+'GHz.pdf', r'$\nu = $'+str(freq)+'GHz')
        #plot_APS_residual(cl_res[0],cl_res[1],cl_res[2],dir_out+'residual_spectra/plots/Cell'+str(freq)+'GHz.pdf', r'$\nu = $'+str(freq)+'GHz')

def tods_plots(tods, Freq):
	for tod, freq in zip(tods, Freq):
		fig = plt.figure ()
		ax = fig.add_subplot(1,1,1)
		ax.plot (tod[:1000] , 'b-', label = 'TOD '+str(freq)+'GHz')
		ax.set_ylabel ('TOD')
		ax.set_xlabel ('t(s)')
		plt.grid()
		plt.legend(loc='upper left')
		plt.savefig(dir_out+'tods/plots/tod_'+str(freq)+'_GHz.pdf')
		#plt.show()



def _main_():
    #Freq=[90, 95, 100, 105, 110]
    Freq=[110]
    pysm_maps = []
    pysm_maps_smoot = []
    pysm_maps_gauss = []
    ideal_maps =[]
    conds_ideal=[]
    output_maps=[]
    conds=[]
    res_maps =[]
    tods=[]
    for freq in Freq:
        #pysm_maps.append(hp.read_map(opj(dir_inp, 'pysm_maps/CMB_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False))
        #pysm_maps_smoot.append(hp.read_map(opj(dir_out, 'Smoot_IM/In_map_smoot_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False))
        #pysm_maps_gauss.append(hp.read_map(opj(dir_out, 'Gauss_IM/In_map_gauss_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False))
        ideal_maps.append(hp.read_map(opj(dir_ideal, 'Output_maps/Bconv_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False))
        conds_ideal.append(hp.read_map(opj(dir_ideal, 'Output_maps/Cond_Numb_'+str(freq)+'GHz.fits'), verbose = False))
        output_maps.append(hp.read_map(opj(dir_out, 'Output_maps/Bconv_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False))
        conds.append(hp.read_map(opj(dir_out, 'Output_maps/Cond_Numb_'+str(freq)+'GHz.fits'), verbose = False))
        res_maps.append(hp.read_map(opj(dir_out, 'Res_Maps/res_maps_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False))
        tods.append(np.load(opj(dir_out, 'tods/tod_'+str(freq)+'GHz.npy')))


    residual(Freq)
    #view_maps(pysm_maps, conds,Freq, dir_out,'pysm_')
    #view_maps(pysm_maps_smoot, conds,Freq,dir_out,'pysm_smooth_')
    #view_maps(pysm_maps_gauss, conds,Freq,dir_out,'pysm_maps_gauss_')
    #view_maps(ideal_maps, conds_ideal,Freq,dir_ideal,'ideal_maps_', Cond = True)
    view_maps(output_maps, conds,Freq,dir_out,'outmap_', Cond=True)
    view_maps(res_maps, conds,Freq,dir_out,'res_map_')
    tods_plots(tods, Freq)

_main_()


######################################################################################################
# def view_maps(Freq):
    
#     '''
#     Display output maps and the relative condition number.
#     ---------
#     '''

#     for freq in Freq:

#         cond = hp.read_map(opj(dir_out, 'Output_maps/Cond_Numb_'+str(freq)+'GHz.fits'), verbose = False)
#         maps = hp.read_map(opj(dir_out, 'Output_maps/Bconv_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False)

#         cond[cond == np.inf] = hp.UNSEEN
#         cart_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]',lonra=[60, 180], latra=[-90, -30])
#         hp.cartview(cond, **cart_opts)
#         plt.savefig(dir_out+'Output_maps/view/cond_numb'+str(freq)+'GHz.pdf')
#         hp.cartview(maps[0], **cart_opts)
#         plt.savefig(dir_out+'Output_maps/view/Tmap'+str(freq)+'GHz.pdf')
#         hp.cartview(maps[1], **cart_opts)
#         plt.savefig(dir_out+'Output_maps/view/Qmap'+str(freq)+'GHz.pdf')
#         hp.cartview(maps[2], **cart_opts)
#         plt.savefig(dir_out+'Output_maps/view/Umap'+str(freq)+'GHz.pdf')
#         #plt.show()

# def view_diff_maps(Freq):
#     '''
#     Display diff maps and the relative condition number.
#     ---------
#     '''
    
#     for freq in Freq:
#         maps = hp.read_map(opj(dir_out, 'Res_Maps/res_maps_'+str(freq)+'GHz.fits'),field=(0,1,2), verbose = False)
#         cart_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]',lonra=[60, 180], latra=[-90, -30])
#         hp.cartview(maps[0], **cart_opts)
#         plt.savefig(dir_out+'Res_Maps/view/Tmap'+str(freq)+'GHz.pdf')
#         hp.cartview(maps[1], **cart_opts)
#         plt.savefig(dir_out+'Res_Maps/view/Qmap'+str(freq)+'GHz.pdf')
#         hp.cartview(maps[2], **cart_opts)
#         plt.savefig(dir_out+'Res_Maps/view/Umap'+str(freq)+'GHz.pdf')
#         #plt.show()