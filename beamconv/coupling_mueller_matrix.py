import numpy as np
from . import transfer_matrix as tm

from matplotlib.pylab import *
import matplotlib.gridspec as gridspec

####################################################################################################################################
##SET THE SYSTEM

## SET1: ar+hwp+ar
sapphire1 = tm.material( 3.07, 3.41, 2.3e-4, 1.25e-4, 'Sapphire', materialType='uniaxial')
duroid1   = tm.material( 1.715, 1.715, 1.2e-3, 1.2e-3, 'RT Duroid', materialType='isotropic')
angles1   = [0.0, 0.0, 0.0]
materials1   = [duroid1, sapphire1, duroid1]
thicknesses1 = [305e-6, 3.15*tm.mm, 305e-6]
hwp1 = tm.Stack( thicknesses1, materials1, angles1)

## SET2: hwp
sapphire2 = tm.material( 3.07, 3.41, 2.3e-4, 1.25e-4, 'Sapphire', materialType='uniaxial')
angles2   = [0.0]
materials2   = [sapphire2]
thicknesses2= [3.15*tm.mm]
hwp2 = tm.Stack( thicknesses2, materials2, angles2)

## SET3: ar1+ar2+hwp+ar2+ar1 
sapphire3 = tm.material( 3.05, 3.38, 0.2e-4, 0.01e-4, 'Sapphire', materialType='uniaxial')
duroid3a   = tm.material( 1.55, 1.55, 0.5e-4, 0.5e-4, 'RT Duroid', materialType='isotropic')
duroid3b   = tm.material( 2.52, 2.52, 56.6e-4, 56.5e-4, 'RT Duroid', materialType='isotropic')
angles3   = [0.0, 0.0, 0.0, 0.0, 0.0]
materials3   = [duroid3a, duroid3b, sapphire3, duroid3b, duroid3a]
thicknesses3 = [0.38*tm.mm, 0.27*tm.mm, 3.75*tm.mm, 0.27*tm.mm,0.38*tm.mm]
hwp3 = tm.Stack( thicknesses3, materials3, angles3)

## SET4: ar+hwp+ar SPYDER
sapphire4 = tm.material( 3.019, 3.336, 2.3e-4, 1.25e-4, 'Sapphire', materialType='uniaxial')
duroid4   = tm.material( 1.951, 1.951, 1.2e-3, 1.2e-3, 'RT Duroid', materialType='isotropic')
angles4   = [0.0, 0.0, 0.0]
materials4   = [duroid4, sapphire4, duroid4]
thicknesses4 = [0.427*tm.mm, 4.930*tm.mm, 0.427*tm.mm]
hwp4 = tm.Stack( thicknesses4, materials4, angles4)


####################################################################################################################################

def TopRowMuellerMatrix(hwp, alph, thet, freqq, xi, psi):
	eta = 1. ## co-polar quantity (FREEZE) 
	delta = 0. ## cross-polar quantity (FREEZE) 
	gamma = (eta**2-delta**2)/(eta**2+delta**2) ## polarization efficienty (FREEZE)
	H = 0.5*(eta**2+delta**2)
	## Ideal case: 
	# eta = 1.
	# delta = 0. 
	# gamma = 1.
	# H = 1.
	
	## Mueller( stack, frequency, incidenceAngle, rotation, inputIndex=1.0, exitIndex=1.0, reflected=False): 
	Mueller = tm.Mueller(hwp, freqq, alph, 0., reflected=False)
	T = Mueller[0,0]
	rho= Mueller[0,1]/ Mueller[0,0]
	c =  Mueller[2,2]/ Mueller[0,0]
	s =  Mueller[3,2]/ Mueller[0,0]

	MII = H*T*(1+(gamma*rho*np.cos((2*thet)+(2*xi))))
	MIQ = H*T*(rho*np.cos((2*thet)+(2*psi)) + (0.5*(1+c)*gamma*np.cos((2*psi)-(2*xi))) + (0.5*(1-c)*gamma*np.cos((4*thet)+(2*xi)+(2*psi))))
	MIU = H*T*(rho*np.sin((2*thet)+(2*psi)) + (0.5*(1+c)*gamma*np.sin((2*psi)-(2*xi))) + (0.5*(1-c)*gamma*np.sin((4*thet)+(2*xi)+(2*psi))))
	MIV = H*T*(s*gamma*np.sin((2*thet)+(2*xi)))
	Mtr= np.array([MII,MIQ,MIU,MIV])
	return Mtr
	

def coupling_system(hwp, freq, theta, alpha, xi, psi):
	if np.isscalar (alpha) and np.isscalar (theta) and np.isscalar (freq):
		M_tr = TopRowMuellerMatrix(hwp, alpha, theta, freq, xi, psi)
		MII = M_tr[0]
		MIP = 0.5*(M_tr[1]-1j*M_tr[2]) 
		MIP_t = 0.5*(M_tr[1]+1j*M_tr[2]) 
	else:
		if np.isscalar (alpha) and np.isscalar (theta):
			M_tr = np.zeros((4,len(freq)))
			for k in range (len(freq)):
				M_tr[:,k]= TopRowMuellerMatrix(hwp, alpha, theta, freq[k], xi, psi)
			MII = M_tr[0,:]
			MIP = 0.5*(M_tr[1,:]-1j*M_tr[2,:]) 
			MIP_t = 0.5*(M_tr[1,:]+1j*M_tr[2,:]) 
		else: 
			if np.isscalar (alpha):
				M_tr = np.zeros((4,len(theta),len(freq)))
				for j in range(len(theta)):
					for k in range (len(freq)):
						M_tr[:,j,k]= TopRowMuellerMatrix(hwp, alpha, theta[j], freq[k], xi, psi)
				MII = M_tr[0,:,:]
				MIP = 0.5*(M_tr[1,:,:]-1j*M_tr[2,:,:]) 
				MIP_t = 0.5*(M_tr[1,:,:]+1j*M_tr[2,:,:]) 
			else:
				M_tr =np.zeros((4,len(alpha),len(theta),len(freq)))
				for i in range (len(alpha)):
					for j in range(len(theta)):
						for k in range (len(freq)):
							M_tr[:,i,j,k]= TopRowMuellerMatrix(hwp, alpha[i], theta[j], freq[k], xi, psi)
				MII = M_tr[0,:,:,:]
				MIP = 0.5*(M_tr[1,:,:,:]-1j*M_tr[2,:,:,:]) 
				MIP_t = 0.5*(M_tr[1,:,:,:]+1j*M_tr[2,:,:,:]) 
				
	return MII, MIP, MIP_t 

####################################################################################################################################
## SETTING TEST PARAMETERS (all the angles in degrees)

# GHz = 1e9 
# deg = pi/180. #from deg2rad

# freq = linspace(10,300,1000) * GHz ## frequency range
# theta = array([0.,2.,4.,6.,8.,10.])*deg ## HWP angle
# alpha = array([0.0, 2.39, 4.75,  7.06, 9.29, 11.43, 13.48])*deg ## angle of incidence
# #alpha = array([0.])*deg

# eta = 1. ## co-polar quantity (FREEZE) 
# delta = 0. ## cross-polar quantity (FREEZE) 
# gamma = (eta**2-delta**2)/(eta**2+delta**2) ## polarization efficienty (FREEZE)
# ## Ideal case: eta = 1., delta = 0., gamma = 1.

# xi = 0.*deg ## angle of the detector (FREEZE)  
# psi = 0.*deg ## instrument rotation (FREEZE) 

#coupling_system(hwp4, freq, theta, alpha, xi, psi)
