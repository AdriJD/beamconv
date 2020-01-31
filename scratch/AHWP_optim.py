##Comute the penalty function post phase substraction

import numpy as np
import matplotlib.pyplot as plt
from beamconv import Beam
from beamconv import transfer_matrix as tm
from scipy.optimize import minimize


def penalty1(x):
    n1=x[0]; d1=x[1];
    sp_index = np.array([3.019, 3.336])
    sp_thick = 3.86
    angles = np.array([0., 0.,52.5,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                         [1e-3,1e-3]])
    phase = np.radians(57.1)
    thicknesses=np.array([d1, sp_thick, sp_thick, sp_thick, d1])
    indices=np.array([[n1,n1], 
                    sp_index, sp_index, sp_index,
                     [n1,n1]])
    nu = np.array((80,85,90,95,100,105,110,135,140,145,150,155,160,165))

    B = np.zeros((nu.size, 4,4))
    penalty = np.zeros(nu.size)
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
                    (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1)))
    for i, freq in enumerate(nu):

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)

        penalty[i] = np.sum(np.square(B[i]-ideal_muell_mat))
        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty[i] += np.sum(np.square(B[i]-ideal_muell_mat))
    return np.sum(penalty)

def grad_penalty1(x):
    n1=x[0]; d1=x[1];
    grad = np.zeros(2)
    grad[0] = (penalty1([n1+0.0001, d1])-penalty1([n1-0.0001, d1]))/0.0002
    grad[1] = (penalty1([n1, d1+0.0001])-penalty1([n1, d1-0.0001]))/0.0002
    return grad


def penalty2(x):
    n1=x[0]; n2=x[1]; d1=x[2]; d2=x[3]; 
    sp_index = np.array([3.019, 3.336])
    sp_thick = 3.86
    angles = np.array([0.,0., 0.,52.5,0., 0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3], [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [1e-3,1e-3], [1e-3,1e-3]])
    phase = np.radians(57.1)
    thicknesses=np.array([d1, d2, sp_thick, sp_thick, sp_thick, d2, d1])
    indices=np.array([[n1,n1],[n2, n2], 
                    sp_index, sp_index, sp_index,
                    [n2,n2], [n1,n1]])
    nu = np.array((80,85,90,95,100,105,110,135,140,145,150,155,160,165))

    B = np.zeros((nu.size, 4,4))
    penalty = np.zeros(nu.size)
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
                    (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1)))
    for i, freq in enumerate(nu):

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)

        penalty[i] = np.sum(np.square(B[i]-ideal_muell_mat))
        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty[i] += np.sum(np.square(B[i]-ideal_muell_mat))
    return np.sum(penalty)

def grad_penalty2(x):
    n1=x[0]; n2=x[1]; d1=x[2]; d2=x[3]; 
    grad = np.zeros(4)
    grad[0] = (penalty2([n1+0.0001,n2, d1, d2])-penalty2([n1-0.0001, n2, d1,d2]))/0.0002
    grad[1] = (penalty2([n1,n2+0.0001, d1, d2])-penalty2([n1,n2-0.0001, d1, d2]))/0.0002
    grad[2] = (penalty2([n1,n2, d1+0.0001, d2])-penalty2([n1,n2, d1-0.0001, d2]))/0.0002
    grad[3] = (penalty2([n1,n2, d1, d2+0.0001])-penalty2([n1,n2, d1,d2-0.0001 ]))/0.0002
    return grad


def penalty3(x):
    n1=x[0]; n2=x[1]; n3=x[2]; d1=x[3]; d2=x[4]; d3=x[5];
    sp_index = np.array([3.019, 3.336])
    sp_thick = 3.86
    angles = np.array([0.,0.,0., 0.,52.5,0., 0.,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3],[1e-3,1e-3], [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [1e-3,1e-3], [1e-3,1e-3], [1e-3,1e-3]])
    phase = np.radians(57.1)
    thicknesses=np.array([d1, d2, d3, sp_thick, sp_thick, sp_thick, d3, d2, d1])
    indices=np.array([[n1, n1],[n2, n2],[n3, n3], 
                    sp_index, sp_index, sp_index,
                      [n3,n3],[n2,n2], [n1,n1]])
    nu = np.array((80,85,90,95,100,105,110,135,140,145,150,155,160,165))

    B = np.zeros((nu.size, 4,4))
    penalty = np.zeros(nu.size)
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
                    (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1)))
    for i, freq in enumerate(nu):

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)

        penalty[i] = np.sum(np.square(B[i]-ideal_muell_mat))
        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty[i] += np.sum(np.square(B[i]-ideal_muell_mat))
    return np.sum(penalty)

def grad_penalty3(x):
    n1=x[0]; n2=x[1]; n3=x[2]; d1=x[3]; d2=x[4]; d3=x[5];
    grad = np.zeros(6)
    grad[0] = (penalty3([n1+0.0001,n2, n3, d1, d2, d3])-penalty3([n1-0.0001,n2, n3, d1, d2, d3]))/0.0002
    grad[1] = (penalty3([n1,n2+0.0001, n3, d1, d2, d3])-penalty3([n1,n2-0.0001, n3, d1, d2, d3]))/0.0002
    grad[2] = (penalty3([n1,n2, n3+0.0001, d1, d2, d3])-penalty3([n1,n2, n3-0.0001, d1, d2, d3]))/0.0002
    grad[3] = (penalty3([n1,n2, n3, d1+0.0001, d2, d3])-penalty3([n1,n2, n3, d1-0.0001, d2, d3]))/0.0002
    grad[4] = (penalty3([n1,n2, n3, d1, d2+0.0001, d3])-penalty3([n1,n2, n3, d1, d2-0.0001, d3]))/0.0002
    grad[5] = (penalty3([n1,n2, n3, d1, d2, d3+0.0001])-penalty3([n1,n2, n3, d1, d2, d3-0.0001]))/0.0002
    return grad

def penalty5(x):
    n1=x[0]; n2=x[1]; n3=x[2]; d1=x[3]; d2=x[4]; d3=x[5];
    sp_index = np.array([3.019, 3.336])
    sp_thick = 3.86
    angles = np.array([0.,0.,0., 0.,29.,94.5,29.,0., 0.,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3],[1e-3,1e-3], [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [1e-3,1e-3], [1e-3,1e-3], [1e-3,1e-3]])
    phase = np.radians(51)
    thicknesses=np.array([d1, d2, d3,sp_thick,
                        sp_thick, sp_thick, sp_thick, 
                        sp_thick, d3, d2, d1])
    indices=np.array([[n1, n1],[n2, n2],[n3, n3], 
                    sp_index, sp_index, sp_index, sp_index, sp_index,
                      [n3,n3],[n2,n2], [n1,n1]])
    nu = np.array((80,85,90,95,100,105,110,135,140,145,150,155,160,165))

    B = np.zeros((nu.size, 4,4))
    penalty = np.zeros(nu.size)
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
                    (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1)))
    for i, freq in enumerate(nu):

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)

        penalty[i] = np.sum(np.square(B[i]-ideal_muell_mat))
        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty[i] += np.sum(np.square(B[i]-ideal_muell_mat))
    return np.sum(penalty)


print('Optimization for the 5-stack AHWP with 3 AR layers')
x3 = [1.5, 2, 2.5, .3,.3,.3]
res5 = minimize(penalty5, x3, method='BFGS', jac=grad_penalty3, options={'disp':True})
print('Indices, Thicknesses(mm)', res5.x)

print('Optimization for the 3-stack AHWP with 1 AR layer')
x1 = [2.2, 0.4]
res1 = minimize(penalty1, x1, method='BFGS', jac=grad_penalty1, options={'disp':True})
print('Indices, Thicknesses(mm)', res1.x)

print('Optimization for the 3-stack AHWP with 2 AR layer')
x2 = [1.5, 2, .3,.3]
res2 = minimize(penalty2, x2, method='BFGS', jac=grad_penalty2, options={'disp':True})
print('Indices, Thicknesses(mm)', res2.x)

print('Optimization for the 3-stack AHWP with 3 AR layer')
x3 = [1.5, 2, 2.5, .3,.3,.3]
res3 = minimize(penalty3, x3, method='BFGS', jac=grad_penalty3, options={'disp':True})
print('Indices, Thicknesses(mm)', res3.x)





# nu_0 = (95+150)/2.
# epsilon = 150/nu_0 - 1.
# delta = np.pi/3 - (epsilon*np.pi/2)**2/(2*np.sqrt(3))

# nu_1 = (95**2+150**2)/(95+150)
# epsilon_1 = 165 /nu_1 - 1.
# delta_1 = np.pi/3 - (epsilon_1*np.pi/2)**2/(2*np.sqrt(3))

# sp_index = np.array([3.019, 3.336])
# sp_thick = 3.86
# angles = np.array([0.,0.,52.5,0.,0.])*np.pi/180.0
# losses = np.array([[1e-3,1e-3], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],[1e-3,1e-3 ]])
# phase = np.radians(57.1)

# ar_thick = np.linspace(.2, .45, 51)
# ar_indx = np.linspace(1.4, 2.2, 17)

# phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
#                     (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1))) #Rotate back by twice the phase

# nu = np.array((80, 85,90,95,100,105,110,135,140,145,150,155,160,165))
# theta = np.linspace(0,180,1801)

# ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))

# F_array = np.zeros((ar_indx.size,ar_thick.size)) 



# A = np.zeros((nu.size,3,theta.size), dtype=complex)

# for k, arin in enumerate(ar_indx):
#     print(arin)
#     for j, arth in enumerate(ar_thick):
#         B = np.zeros((nu.size, 4,4))
#         penalty = np.zeros(nu.size)
#         for i, freq in enumerate(nu):
               
#             # print('Frequency:', freq)
#             # dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
#             # dumbeam.set_hwp_mueller(model_name='1AR3BRcent')
#             # A[i,:,:] = dumbeam.get_mueller_top_row_full(psi=np.zeros(theta.size), xi=np.zeros(theta.size), theta=np.radians(theta))
#             #np.radians(theta[np.argmax(np.real(A[i,1,:900]), axis=0)])

#             dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
#             dumbeam.set_hwp_mueller(thicknesses=np.array([arth, sp_thick, sp_thick, sp_thick, arth]), 
#                             indices=np.array([[arin, arin], sp_index, sp_index, sp_index, [arin,arin]]), 
#                             losses=losses, angles=angles)
#             B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
#             if (arin==1.8 and arth==0.35):
#                 print('Frequency: ', freq)
#                 print('Unrotated Mueller:\n', dumbeam.hwp_mueller)
#                 print('Rotated Mueller:\n', B[i])
#             penalty[i] = np.sum(np.square(B[i]-ideal_muell_mat))
#             dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
#             dumbeam.set_hwp_mueller(thicknesses=np.array([arth, sp_thick, sp_thick, sp_thick, arth]), 
#                             indices=np.array([[arin, arin], sp_index, sp_index, sp_index, [arin,arin]]), 
#                             losses=losses, angles=angles)
#             B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
#             penalty[i] += np.sum(np.square(B[i]-ideal_muell_mat))

#         F_array[k,j] = np.sum(penalty)
#     # print('HWP Mueller is:\n', dumbeam.hwp_mueller)
#     # print('Phase is:\n',np.degrees(phase))
#     # print('Rotated Mueller is:\n', np.dot(phase_rot,dumbeam.hwp_mueller))
#     # print('Penalty is: \n',penalty[i])#Sum over all elements of squared differences
# # print('Total_penalty:\n', np.sum(penalty))

# cont_f =plt.contourf(ar_thick, ar_indx, F_array, levels=30)
# plt.colorbar(cont_f)
# plt.xlabel('AR thickness (mm)')
# plt.ylabel('AR refraction index')
# plt.title('Penalty function for a 3 layer AHWP with 1AR layer, phase corrected at 95 and 150GHz')
# plt.show()