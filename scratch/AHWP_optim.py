##Comute the penalty function post phase substraction

import numpy as np
import matplotlib.pyplot as plt
from beamconv import Beam
from beamconv.transfer_matrix import MuellerRotation

nu_0 = (95+150)/2.
epsilon = 150/nu_0 - 1.
delta = np.pi/3 - (epsilon*np.pi/2)**2/(2*np.sqrt(3))

nu_1 = (95**2+150**2)/(95+150)
epsilon_1 = 165 /nu_1 - 1.
delta_1 = np.pi/3 - (epsilon_1*np.pi/2)**2/(2*np.sqrt(3))

nu = np.array((80, 85,90,95,100,105,110,135,140,145,150,155,160,165))
theta = np.linspace(0,180,1801)
alpha = np.linspace(0, 20, 21)
Jones_gamma = np.zeros((nu.size,2))
Trhocs = np.zeros((nu.size,4))
nT = np.zeros((nu.size,4,4))
Muel = np.zeros((nu.size,4,4))
s_i =[3.019, 3.336]
s_l =[2.3e-4, 1.25e-4]
s_t = 3.65

ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
penalty = np.zeros(nu.size)
A = np.zeros((nu.size,3,theta.size), dtype=complex)
B = np.zeros((nu.size, 4,4))
for i, freq in enumerate(nu):
    print('Frequency:', freq)
    dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
    dumbeam.set_hwp_mueller(model_name='1AR3BRcent')
    A[i,:,:] = dumbeam.get_mueller_top_row_full(psi=np.zeros(theta.size), xi=np.zeros(theta.size), theta=np.radians(theta))
    phase = np.radians(57.1)#np.radians(theta[np.argmax(np.real(A[i,1,:900]), axis=0)])
    phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
                        (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1)))#Rotateback by twice the phase
    B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
    penalty[i] = np.sum(np.square(B[i]-ideal_muell_mat))
    print('HWP Mueller is:\n', dumbeam.hwp_mueller)
    print('Phase is:\n',np.degrees(phase))
    print('Rotated Mueller is:\n', np.dot(phase_rot,dumbeam.hwp_mueller))
    print('Penalty is: \n',penalty[i])#Sum over all elements of squared differences

print('Total_penalty:\n', np.sum(penalty))