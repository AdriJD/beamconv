import numpy as np
from . import transfer_matrix as tm

from matplotlib.pylab import *
import matplotlib.gridspec as gridspec

from .detector import Beam

class Half_Wave_plate(Beam):

    def __init__(self, alpha=0.0, theta=0.0, freq=1.5e9, xi=0.0, psi=0.0, hwp=None, model_name=None):
        self.alpha = alpha 
        self.theta = theta
        self.freq = freq
        self.xi = xi
        self.psi = psi
        self.hwp = hwp
        self.model_name = model_name
        self.freq = Beam.sensitive_freq

    def _choose_HWP_model(model_name):
        sapphire = tm.material( 3.07, 3.41, 2.3e-4, 1.25e-4, 'Sapphire', materialType='uniaxial')
        duroid_a = tm.material( 1.55, 1.55, 0.5e-4, 0.5e-4, 'RT Duroid', materialType='isotropic')
        duroid_b = tm.material( 1.715, 1.715, 1.2e-3, 1.2e-3, 'RT Duroid', materialType='isotropic')
        duroid_c = tm.material( 2.52, 2.52, 56.6e-4, 56.5e-4, 'RT Duroid', materialType='isotropic')
        duroid_d = tm.material( 1.951, 1.951, 1.2e-3, 1.2e-3, 'RT Duroid', materialType='isotropic')
        if (model_name=='HWP_only'):
            thicknesses = [305e-6, 3.15*tm.mm, 305e-6]
            angles   = [0.0, 0.0, 0.0]
            materials   = [duroidb, sapphire, duroidb]
        
        elif (model_name=='Ar+HWP+Ar'):
            thicknesses = [3.15*tm.mm]
            materials = [sapphire]
            angles = [0.0]

        elif (model_name=='Ar1+Ar2+HWP+Ar2+Ar1'):
            
            thicknesses = [0.38*tm.mm, 0.27*tm.mm, 3.75*tm.mm, 0.27*tm.mm,0.38*tm.mm]
            materials = [duroid_a, duroid_c, sapphire, duroid_c, duroid_a]
            angles = [0.0, 0.0, 0.0, 0.0, 0.0]

        elif (model_name=='SPIDER'):
            
            thicknesses = [0.427*tm.mm, 4.930*tm.mm, 0.427*tm.mm]
            materials = [duroid_d, sapphire, duroid_d]
            angles = [0.0, 0.0, 0.0]

        else:
            raise ValueError('Unknown tyoe of HWP entered')

        hwp = tm.Stack( thicknesses, materials, angles)
        return hwp

    def _topRowMuellerMatrix(self):
        eta = 1. ## co-polar quantity (FREEZE) 
        delta = 0. ## cross-polar quantity (FREEZE) 
        gamma = (eta**2-delta**2)/(eta**2+delta**2) ## polarization efficienty (FREEZE)
        H = 0.5*(eta**2+delta**2)
        ## Ideal case: H = 0.5

        #########################
        ## REAL HWP
        #           Mueller( stack, frequency, incidenceAngle, rotation, inputIndex=1.0, exitIndex=1.0, reflected=False): 
        Mueller = tm.Mueller(self.hwp, self.freq, self.alpha, 0., reflected=False)
        T = Mueller[0,0]
        rho= Mueller[0,1]/ Mueller[0,0]
        c =  Mueller[2,2]/ Mueller[0,0]
        s =  Mueller[3,2]/ Mueller[0,0]
        #########################

        #########################
        # IDEAL HWP 
        #T = 1.
        #c = -1.
        #rho = 0.
        #s = 0.
        #########################

        MII = H*T*(1+(gamma*rho*np.cos((2*theta)+(2*xi))))
        MIQ = H*T*(rho*np.cos((2*theta)+(2*psi)) + (0.5*(1+c)*gamma*np.cos((2*psi)-(2*xi))) + (0.5*(1-c)*gamma*np.cos((4*theta)+(2*xi)+(2*psi))))
        MIU = H*T*(rho*np.sin((2*theta)+(2*psi)) + (0.5*(1+c)*gamma*np.sin((2*psi)-(2*xi))) + (0.5*(1-c)*gamma*np.sin((4*theta)+(2*xi)+(2*psi))))
        MIV = H*T*(s*gamma*np.sin((2*theta)+(2*xi)))
        Mtr= np.array([MII,MIQ,MIU,MIV])
        return Mtr

    def _coupling_system(self):
        if (self.hwp==None and self.model_name!=None):
            self.hwp = self._choose_HWP_model(model_name)
        elif (self.hwp==None and self.model_name==None):
            raise ValueError('Asked to modulate by a half wave plate, but none specified or given')

        if np.isscalar (self.theta):
            M_tr = TopRowMuellerMatrix(self.hwp, self.alpha, self.theta, self.freq, self.xi, -self.psi)
            ### IQUV base
            MII = M_tr[0]
            MIQ = M_tr[1] 
            MIU = M_tr[2]

            ### IPPV base
            MIP = 0.5*(M_tr[1]-1j*M_tr[2]) 
            MIP_t = 0.5*(M_tr[1]+1j*M_tr[2]) 
        else:
            M_tr = np.zeros((4,len(theta)))
            for i in range (len(theta)):
                M_tr[:,i]= TopRowMuellerMatrix(self.hwp, self.alpha, self.theta[i], self.freq, self.xi, -self.psi[i])
            ### IQUV base
            MII = M_tr[0,:]
            MIQ = M_tr[1,:] 
            MIU = M_tr[2,:]

            ### IPPV base
            MIP = 0.5*(M_tr[1,:]-1j*M_tr[2,:]) 
            MIP_t = 0.5*(M_tr[1,:]+1j*M_tr[2,:])

        #return MII, MIQ, MIU 
        return MII, MIP, MIP_t 



#############END CLASS

    

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

## SET4: ar+hwp+ar SPIDER
sapphire4 = tm.material( 3.019, 3.336, 2.3e-4, 1.25e-4, 'Sapphire', materialType='uniaxial')
duroid4   = tm.material( 1.951, 1.951, 1.2e-3, 1.2e-3, 'RT Duroid', materialType='isotropic')
angles4   = [0.0, 0.0, 0.0]
materials4   = [duroid4, sapphire4, duroid4]
thicknesses4 = [0.427*tm.mm, 4.930*tm.mm, 0.427*tm.mm]
hwp4 = tm.Stack( thicknesses4, materials4, angles4)


####################################################################################################################################

'''
## ALTERNATIVE WAY TO COMPUTE THE TOTAL MUELLER MATRIX THROUGH THE MATRIX PRODUCT
def rot_matrix(angle): 
    c = np.cos(2*angle)
    s = np.sin(2*angle)
    return np.array([[1,0,0,0],[0,c,s,0],[0,-s,c,0],[0,0,0,1]])

def TopRowMuellerMatrix(hwp, alph, thet, freqq, xi, psi):
    eta = 1. ## co-polar quantity (FREEZE) 
    delta = 0. ## cross-polar quantity (FREEZE) 
    
    a = (eta**2+delta**2)
    b = (eta**2-delta**2)
    c = 2*eta*delta

    M_pol = 0.5*np.array([[a,b,0,0],[b,a,0,0],[0,0,c,0],[0,0,0,c]])

    #########################
    # REAL HWP
    # Mueller( stack, frequency, incidenceAngle, rotation, inputIndex=1.0, exitIndex=1.0, reflected=False): 
    # Mueller = tm.Mueller(hwp, freqq, alph, 0, reflected=False)
    # T = Mueller[0,0]
    # rho= Mueller[0,1]
    # c =  Mueller[2,2]
    # s =  Mueller[3,2]
    #########################

    
    ########################
    # # IDEAL HWP
    T = 1.
    rho= 0.
    c =  -1.
    s =  0.
    Mueller = np.array([[T,rho,0,0],[rho,T,0,0],[0,0,c,-s],[0,0,s,c]])
    ########################

    
    M_thet=rot_matrix(thet)
    M_mthet=rot_matrix(-thet)
    M_xi= rot_matrix(-xi)
    M_psi= rot_matrix(psi)
    M_hwp = dot(Mueller,M_thet)
    M_hwp = dot(M_mthet,M_hwp)

    M_tot = dot(M_pol, dot(M_xi, dot(M_hwp,M_psi)))

    Mtr= array([M_tot[0,0],M_tot[0,1],M_tot[0,2],M_tot[0,3]])
    return Mtr

'''


def TopRowMuellerMatrix(hwp, alph, thet, freqq, xi, psi):
    eta = 1. ## co-polar quantity (FREEZE) 
    delta = 0. ## cross-polar quantity (FREEZE) 
    gamma = (eta**2-delta**2)/(eta**2+delta**2) ## polarization efficienty (FREEZE)
    H = 0.5*(eta**2+delta**2)
    ## Ideal case: H = 0.5

    #########################
    ## REAL HWP
    # Mueller( stack, frequency, incidenceAngle, rotation, inputIndex=1.0, exitIndex=1.0, reflected=False): 
    Mueller = tm.Mueller(hwp, freqq, alph, 0., reflected=False)
    T = Mueller[0,0]
    rho= Mueller[0,1]/ Mueller[0,0]
    c =  Mueller[2,2]/ Mueller[0,0]
    s =  Mueller[3,2]/ Mueller[0,0]
    #########################

    #########################
    # IDEAL HWP 
    #T = 1.
    #c = -1.
    #rho = 0.
    #s = 0.
    #########################

    MII = H*T*(1+(gamma*rho*np.cos((2*thet)+(2*xi))))
    MIQ = H*T*(rho*np.cos((2*thet)+(2*psi)) + (0.5*(1+c)*gamma*np.cos((2*psi)-(2*xi))) + (0.5*(1-c)*gamma*np.cos((4*thet)+(2*xi)+(2*psi))))
    MIU = H*T*(rho*np.sin((2*thet)+(2*psi)) + (0.5*(1+c)*gamma*np.sin((2*psi)-(2*xi))) + (0.5*(1-c)*gamma*np.sin((4*thet)+(2*xi)+(2*psi))))
    MIV = H*T*(s*gamma*np.sin((2*thet)+(2*xi)))
    Mtr= np.array([MII,MIQ,MIU,MIV])
    return Mtr


### ONLY HWP_ANGLE CAN BE AN ARRAY
def coupling_system(hwp, freq, theta, alpha, xi, psi):
    if np.isscalar (theta):
        M_tr = TopRowMuellerMatrix(hwp, alpha, theta, freq, xi, psi)
        ### IQUV base
        MII = M_tr[0]
        MIQ = M_tr[1] 
        MIU = M_tr[2]

        ### IPPV base
        MIP = 0.5*(M_tr[1]-1j*M_tr[2]) 
        MIP_t = 0.5*(M_tr[1]+1j*M_tr[2]) 
    else:
        M_tr = np.zeros((4,len(theta)))
        for i in range (len(theta)):
            M_tr[:,i]= TopRowMuellerMatrix(hwp, alpha, theta[i], freq, xi, psi[i])
        ### IQUV base
        MII = M_tr[0,:]
        MIQ = M_tr[1,:] 
        MIU = M_tr[2,:]

        ### IPPV base
        MIP = 0.5*(M_tr[1,:]-1j*M_tr[2,:]) 
        MIP_t = 0.5*(M_tr[1,:]+1j*M_tr[2,:])

    #return MII, MIQ, MIU 
    return MII, MIP, MIP_t 

####################################################################################################################################
# SETTING TEST PARAMETERS (all the angles in degrees)

# GHz = 1e9 
# deg = pi/180. #from deg2rad

# freq = linspace(10,300,1000) * GHz ## frequency range
# theta = array([0.,2.,4.,6.,8.,10.])*deg ## HWP angle
# alpha = array([0.0, 2.39, 4.75,  7.06, 9.29, 11.43, 13.48])*deg ## angle of incidence

# xi = 0.*deg ## angle of the detector (FREEZE)  what in sasha's note is called xi
# psi = 10.*deg ## instrument rotation (FREEZE) 


# M_II,M_IP, M_IP_t = coupling_system(hwp4, freq, theta, alpha, xi, psi)
