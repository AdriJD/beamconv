##Comute the penalty function post phase substraction

import numpy as np
import matplotlib.pyplot as plt
from beamconv import Beam
from beamconv import transfer_matrix as tm
from scipy.optimize import minimize

def modBB(nu, beta):
    hkb = 4.799e-11
    T = 15.9
    wien_peak = 9.3475e11
    wien_modi = (wien_peak**beta)*np.exp(-hkb*wien_peak/T) 
    return ((nu*1e9)**beta)*np.exp(-hkb*nu*1e9/T)/wien_modi

def penalty_monoref(x):
    n1=x[0]; n2=x[1]; n3=x[2]; d1=x[3]; d2=x[4]; d3=x[5];
    sp_index = np.array([3.019, 3.336])
    sp_thick = x[6];
    angles = np.array([0.,0.,0.,0., 0.,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3],[1e-3,1e-3], [1e-3,1e-3], 
                    [2.3e-4, 1.25e-4],[1e-3,1e-3], [1e-3,1e-3], [1e-3,1e-3]])
    thicknesses=np.array([d1, d2, d3 , sp_thick, d3, d2, d1])
    indices=np.array([[n1, n1],[n2, n2],[n3, n3], 
                    sp_index,[n3,n3],[n2,n2], [n1,n1]])

    nu = np.array((80,85,90,95,100,105,110,135,140,145,150,155,160,165))

    penalty = np.zeros(nu.size)
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))

    for i, freq in enumerate(nu):

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        penalty[i] = np.sum(np.square(dumbeam.hwp_mueller-ideal_muell_mat))*modBB(freq, 1.59)

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        penalty[i] += np.sum(np.square(dumbeam.hwp_mueller-ideal_muell_mat))*modBB(freq, 1.59)
    return .5*np.sum(penalty)

def grad_penalty_monoref(x):

    grad = np.zeros(x.size)
    for i in range(grad.size):
        xp=np.copy(x)
        xm=np.copy(x)
        xp[i]+=1e-6
        xm[i]-=1e-6
        grad[i]=(penalty_monoref(xp)-penalty_monoref(xm))/2e-6

    return grad

def penalty_mono2(x):
    n1=x[0]; n2=x[1]; d1=x[2]; d2=x[3]; sp_thick=x[4];
    sp_index = np.array([3.019, 3.336])
    angles = np.array([0.,0.,0., 0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3],[1e-3,1e-3], 
                    [2.3e-4, 1.25e-4], [1e-3,1e-3], [1e-3,1e-3]])
    thicknesses=np.array([d1, d2, sp_thick, d2, d1])
    indices=np.array([[n1, n1],[n2, n2], 
                    sp_index,[n2,n2], [n1,n1]])
    nu = np.array((80,85,90,95,100,105,110,135,140,145,150,155,160,165))
    penalty = np.zeros(nu.size)
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    
    for i, freq in enumerate(nu):

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        penalty[i] = np.sum(np.square(dumbeam.hwp_mueller-ideal_muell_mat))

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        penalty[i] += np.sum(np.square(dumbeam.hwp_mueller-ideal_muell_mat))
    return np.sum(penalty)

def grad_penalty_mono2(x):

    grad = np.zeros(x.size)
    for i in range(grad.size):
        xp=np.copy(x)
        xm=np.copy(x)
        xp[i]+=1e-6
        xm[i]-=1e-6
        grad[i]=(penalty_mono2(xp)-penalty_mono2(xm))/2e-6

    return grad

def penalty_mono1(x):
    n1=x[0]; d1=x[1]; sp_thick=x[2];
    sp_index = np.array([3.019, 3.336])
    angles = np.array([0.,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3], [2.3e-4, 1.25e-4], [1e-3,1e-3]])
    thicknesses=np.array([d1, sp_thick, d1])
    indices=np.array([[n1, n1], sp_index, [n1,n1]])
    nu = np.array((80,85,90,95,100,105,110,135,140,145,150,155,160,165))
    penalty = np.zeros(nu.size)
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    
    for i, freq in enumerate(nu):

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        penalty[i] = np.sum(np.square(dumbeam.hwp_mueller-ideal_muell_mat))

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        penalty[i] += np.sum(np.square(dumbeam.hwp_mueller-ideal_muell_mat))
    return np.sum(penalty)

def grad_penalty_mono1(x):

    grad = np.zeros(x.size)
    for i in range(grad.size):
        xp=np.copy(x)
        xm=np.copy(x)
        xp[i]+=1e-6
        xm[i]-=1e-6
        grad[i]=(penalty_mono1(xp)-penalty_mono1(xm))/2e-6

    return grad


def penalty1(x):
    n1=x[0]; d1=x[1];
    sp_index = np.array([3.019, 3.336])
    sp_thick = x[2]
    angles = np.array([0., 0.,52.5,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                         [1e-3,1e-3]])
    phase = np.radians(x[3])
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
    grad[0] = (penalty1([n1+1e-6, d1])-penalty1([n1-1e-6, d1]))/2e-6
    grad[1] = (penalty1([n1, d1+1e-6])-penalty1([n1, d1-1e-6]))/2e-6
    return grad


def penalty2(x):
    n1=x[0]; n2=x[1]; d1=x[2]; d2=x[3]; 
    sp_index = np.array([3.019, 3.336])
    sp_thick = x[4]
    angles = np.array([0.,0., 0.,52.5,0., 0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3], [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [1e-3,1e-3], [1e-3,1e-3]])
    phase = np.radians(x[5])
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
    grad = np.zeros(x.size)
    for i in range(x.size):
        xp=np.copy(x)
        xm=np.copy(x)
        xp[i]+=1e-6
        xm[i]-=1e-6
        grad[i]=(penalty2(xp)-penalty2(xm))/2e-6
    return grad


def penalty3(x):
    n1=x[0]; n2=x[1]; n3=x[2]; d1=x[3]; d2=x[4]; d3=x[5];
    sp_index = np.array([3.019, 3.336])
    sp_thick =x[6]
    angles = np.array([0.,0.,0., 0.,52.5,0., 0.,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3],[1e-3,1e-3], [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [1e-3,1e-3], [1e-3,1e-3], [1e-3,1e-3]])
    phase = np.radians(x[7])
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
        penalty[i] = np.sum(np.square(B[i]-ideal_muell_mat))*modBB(freq, 1.59)

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty[i] += np.sum(np.square(B[i]-ideal_muell_mat))*modBB(freq, 1.59)
    return .5*np.sum(penalty)

def grad_penalty3(x):
    grad = np.zeros(x.size)
    for i in range(grad.size):
        xp=np.copy(x)
        xm=np.copy(x)
        xp[i]+=1e-6
        xm[i]-=1e-6
        grad[i]=(penalty3(xp)-penalty3(xm))/2e-6
    return grad


def penalty5(x):
    n1=x[0]; n2=x[1]; n3=x[2]; d1=x[3]; d2=x[4]; d3=x[5];
    sp_index = np.array([3.019, 3.336])
    sp_thick = x[6];
    angles = np.array([0.,0.,0., 0.,29.,94.5,29.,2., 0.,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3],[1e-3,1e-3], [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [1e-3,1e-3], [1e-3,1e-3], [1e-3,1e-3]])
    phase = np.radians(x[7])
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
        penalty[i] = np.sum(np.square(B[i]-ideal_muell_mat))*modBB(freq, 1.59)

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(thicknesses=thicknesses, 
                        indices=indices, 
                        losses=losses, angles=angles)
        B[i]=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty[i] += np.sum(np.square(B[i]-ideal_muell_mat))*modBB(freq, 1.59)
    return .5*np.sum(penalty)

def grad_penalty5(x):
    grad = np.zeros(x.size)
    for i in range(grad.size):
        xp=np.copy(x)
        xm=np.copy(x)
        xp[i]+=1e-6
        xm[i]-=1e-6
        grad[i]=(penalty5(xp)-penalty5(xm))/2e-6
    return grad

def big_penalty(x):
    n1=x[0]; n2=x[1]; n3=x[2]; d1=x[3]; d2=x[4]; d3=x[5]; 
    s1=x[6];s2=x[7];s3=x[8]; a1=x[9]; a2=x[10]; a3=x[11];
    sp_index = np.array([3.019, 3.336])
    angles = np.array([0.,0.,0., a1,a2,a3, 0.,0.,0.])*np.pi/180.0
    losses = np.array([ [1e-3,1e-3],[1e-3,1e-3], [1e-3,1e-3], 
                        [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4], [2.3e-4, 1.25e-4],
                        [1e-3,1e-3], [1e-3,1e-3], [1e-3,1e-3]])
    phase = np.radians(x[12])
    thicknesses=np.array([d1, d2, d3, s1, s2, s3, d3, d2, d1])
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

def big_gradient(x):
    grad = np.zeros(x.size)
    for i in range(grad.size):
        xp=np.copy(x)
        xm=np.copy(x)
        xp[i]+=1e-6
        xm[i]-=1e-6
        grad[i]=(big_penalty(xp)-big_penalty(xm))/2e-6
    return grad

# print('Optimization for the 1-stack HWP with 1 AR layer')
# x_mono1 = [1.1,0.1,3.]
# res_mono1 = minimize(penalty_mono1, x_mono1, method='BFGS', jac=grad_penalty_mono1, options={'disp':True})
# print('Indices, Thicknesses(mm)', res_mono1.x)

# print('Optimization for the 1-stack HWP with 2 AR layers')
# x_mono2 = [1.1, 2., 0.4, 0.3, 3.78]
# res_mono2 = minimize(penalty_mono2, x_mono2, method='BFGS', jac=grad_penalty_mono2, options={'disp':True})
# print('Indices, Thicknesses(mm)', res_mono2.x)

print('Optimization for the 1-stack HWP with 3 AR layers')
x_mono = [1.1, 2., 2.6, 0.4, 0.3, 0.2, 3.78]
res_mono = minimize(penalty_monoref, x_mono, method='L-BFGS-B', jac=grad_penalty_monoref, options={'disp':True},
    bounds=[(1., 3.1), (1., 3.1), (1., 3.1), (1e-3, 5), (1e-3, 5), (1e-3, 5), (1e-3, 5)])
print('Indices, Thicknesses(mm), angle (°)', res_mono.x)

# print('Optimization for the 3-stack AHWP with 1 AR layer')
# x1 = [2.2, 0.4]
# res1 = minimize(penalty1, x1, method='BFGS', jac=grad_penalty1, options={'disp':True})
# print('Indices, Thicknesses(mm)', res1.x)

# print('Optimization for the 3-stack AHWP with 2 AR layer')
# x2 = [1.5, 2, .3,.3]
# res2 = minimize(penalty2, x2, method='BFGS', jac=grad_penalty2, options={'disp':True})
# print('Indices, Thicknesses(mm)', res2.x)

print('Optimization for the 3-stack AHWP with 3 AR layer')
x3 = [1.5, 2, 2.5, .3, .3, .3, 3.8, 50]
res3 = minimize(penalty3, x3, method='L-BFGS-B', jac=grad_penalty3, options={'disp':True},
    bounds=[(1., 3.1), (1., 3.1), (1., 3.1), (1e-3, 5), (1e-3, 5), (1e-3, 5), (1e-3, 5), (0,90)])
print('Indices, Thicknesses(mm), angle (°)', res3.x)

print('Optimization for the 5-stack AHWP with 3 AR layers')
x3 = [1.5, 2, 2.5, .3, .3, .3, 3.8, 50]
res5 = minimize(penalty5, x3, method='L-BFGS-B', jac=grad_penalty5, options={'disp':True}, 
    bounds=[(1., 3.1), (1., 3.1), (1., 3.1), (1e-3, 5), (1e-3, 5), (1e-3, 5), (1e-3, 5), (0,90)])
print('Indices, Thicknesses(mm)', res5.x)



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