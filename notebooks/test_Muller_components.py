import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import warnings
import numpy as np
import healpy as hp
from beamconv.instrument import ScanStrategy, Beam
from beamconv import tools as beam_tools
from beamconv import plot_tools
import scipy.constants as constants
import emcee
import multiprocessing
from itertools import repeat
opj = os.path.join

c=constants.c
n1 = 3.07
n2 = 3.47

base_dir = '/home/nadia/git_repos/hwp_sims/'
outdir = opj(base_dir, 'hwp_script_out/')


def log_prob(x, ivar):
	return -0.5 * np.sum(ivar * x ** 2)

def compute_best_value(frequency):
	'''
	compute the best value for the sapphire thicknesses for each frequency)

	Arguments:
		frequency : Hz
		n1,n2 : sapphire fast and slow indices
	'''

	bv = c/(2*frequency*np.abs(n1-n2))*10e2
	
	return bv

def set_values():
	'''
	assign the allowed values range for the parameters
	'''

	freq_v = np.arange(30, 465, 15)
	ind_v = np.arange(1.5, 3.1, 0.1)
	thick_d_v = np.arange(0, 2.51, 0.01)
	thick_s_v = np.arange(0, 6, 0.01)
	in_angle_v = np.arange(0, 21, 1)

	return ind_v, thick_d_v

def test_hwp_models(freq, ind, thick_d, thick_s, in_angle, model=False,  
	losses=[1e-04,1e-04], fixed_values=True, check=False, estimator='abs_diff', plot_stuff=False):
	'''
	Test different hwp structures for fixed or fluctuating parameters

	Arguments:
		model : the model of the hwp
		freq : sensitive frequency
		ind : indices of the coatings
		thick_d : array-like, the duroid thicknesses
		thick_s : array-like, sapphire thicknesses (optimized for each frequency through compute_best_val)
		in_angle : angle of incidence

		'''
	if not fixed_values:
		# not ready, needs alterations
		#	ndim, nwalkers = 5, 100
		#	ivar = 1. / np.random.rand(ndim)
		#	p0 = np.random.randn(nwalkers, ndim)
		#	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
		#	sampler.run_mcmc(p0, 10000)
		pass

	if check:
		freq_v, ind_v, thick_d_v, thick_s_v, in_angle_v = set_values()
		if ([freq,ind,thick_d,thick_s,in_angle,layers] not in 
			zip(freq_v, ind_v, thick_d_v, thick_s_v, in_angle_v)):

			raise ValueError('Parameter values are incorrect')

	test_hwp = Beam().hwp()
	if model:
		test_hwp.choose_HWP_model(model)
	else:
		test_hwp.stack_builder(thicknesses=[thick_d,thick_s,thick_d], indices=[[ind,ind],[3.019,3.337],[ind,ind]], 
							losses=np.ones((3,2))*1e-4, angles=np.array([0.0, 0.0, 0.0]))
	T,rho,c,s = test_hwp.compute4params(freq,in_angle)

	if estimator == 'abs_diff':
		F = (T-1)**2 + (rho*T)**2 + (c*T+1)**2 + (T*s)**2
		return F
	

	#if args.system_name == 'Owl':

	#	base_dir = '/mn/stornext/d8/ITA/spider/jon/analysis/'
	#	base_dir_adri = '/mn/stornext/d8/ITA/spider/adri/analysis/'
	#	outdir = opj(base_dir, '20191108_hwpsat_test/')
	#	beam_dirs = opj(base_dir_adri, '20180627_sat/beams')

	#elif args.system_name == 'nadia':

	#base_dir = '/home/nadia/git_repos/hwp_sims/'
	#outdir = opj(base_dir, 'hwp_script_out/')

	#np.save(opj(outdir,'abs_diff.npy'), F)


def main(mult_cores=False, same_length=False, n_cores=4):

	ind_v,thick_d_v = set_values()
	thick_s = compute_best_value(1e11)
	thick_s_v = np.arange(0, 2*thick_s, 2*thick_s/250)
	#print(thick_s,len(ind_v),len(thick_d_v),len(thick_s_v))
	F = np.zeros((len(ind_v),len(thick_d_v),len(thick_s_v)))
	print(ind_v, thick_d_v, thick_s_v)

	if mult_cores and same_length:
		# at some point this will be possible for fluctuating all the parameters at once
		# example of fluctuating one parameter with starmap 
		pool = multiprocessing.Pool(n_cores)
		starmap_args = list(zip(repeat('1layer_HWP'),repeat(30e09),repeat([1.5,1.5]),repeat(0.2e-3),repeat(4e-3),in_angle_v*np.pi/180))
		#starmap_args = list(zip(repeat('1layer_HWP'),freq_v,ind_v,thick_d_v,thick_s_v,in_angle_v))
		results = pool.starmap(test_hwp_models,starmap_args)
		#np.reshape(results, (len(freq_v),len(ind_v),len(thick_d_v),len(thick_s_v),len(in_angle_v)))

	elif mult_cores and not same_length:
		pool = multiprocessing(n_cores)
		for i in range(len(ind_v)):
			for j in range(len(thick_d_v)):
				results = pool.map(test_hwp_models,'1layer_HWP',1e11,i,j,thick_s_v,0)
				F[i,j,:] = results
		time1 = time.ctime()
		print(time1)


	else:
		#stupid one core sims
		# example for freq=100 GHz, incidence_angle=0
		for i in range(len(ind_v)):
			print(i)
			for j in range(len(thick_d_v)):
				#print(j)
				for k in range(len(thick_s_v)):
					F[i,j,k] = test_hwp_models(1e11,ind_v[i],thick_d_v[j],thick_s_v[k],0)
					print(F[i,j,k])
		time2 = time.ctime()
		print(time2)
	np.save(opj(outdir,'abs_diff.npy'), F)



if __name__ == '__main__':
	main()