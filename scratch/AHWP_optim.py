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

def quick_penalty(model_name, phase95=0., phase150=0.):
    penalty = 0.
    nu = np.array((80,85,90,95,100,105,110,135,140,145,150,155,160,165))
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))

    for i, freq in enumerate(nu):

        if (freq<130):
            phase = np.radians(phase95)
        elif (freq>130):
            phase = np.radians(phase150)
        phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
                    (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1)))
        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(model_name=model_name)
        B=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty += np.sum(np.square(B-ideal_muell_mat))*modBB(freq, 1.59)

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(model_name=model_name)
        B=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty += np.sum(np.square(B-ideal_muell_mat))*modBB(freq, 1.59)
    return .5*np.sum(penalty)

def penalty95(model_name, phase95=0.):
    penalty = 0.
    nu = np.array((80,85,90,95,100,105,110))
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    phase = np.radians(phase95)
    for i, freq in enumerate(nu):        
        phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
                    (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1)))
        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(model_name=model_name)
        B=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty += np.sum(np.square(B-ideal_muell_mat))*modBB(freq, 1.59)

        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(model_name=model_name)
        B=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty += np.sum(np.square(B-ideal_muell_mat))*modBB(freq, 1.59)
    return .5*np.sum(penalty)

def penalty150(model_name, phase150=0.):
    penalty = 0.
    nu = np.array((135,140,145,150,155,160,165))
    ideal_muell_mat = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    phase = np.radians(phase150)
    for i, freq in enumerate(nu):        
        phase_rot = np.array(((1.,0.,0.,0.), (0.,np.cos(-4*phase),np.sin(-4*phase),0.),
                    (0.,-np.sin(-4*phase),np.cos(-4*phase),0.),(0.,0.,0.,1)))
        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=0.0)
        dumbeam.set_hwp_mueller(model_name=model_name)
        B=np.dot(phase_rot,dumbeam.hwp_mueller)

        penalty += np.sum(np.square(B-ideal_muell_mat))*modBB(freq, 1.59)
        dumbeam = Beam(name='testbeam', sensitive_freq=freq, el=10.0)
        dumbeam.set_hwp_mueller(model_name=model_name)
        B=np.dot(phase_rot,dumbeam.hwp_mueller)
        penalty += np.sum(np.square(B-ideal_muell_mat))*modBB(freq, 1.59)
    return .5*np.sum(penalty)

def phase_through_penalty(model_name):
    phases = np.linspace(50,65, num=151, endpoint=True)
    best_phase9 = 0
    best_phase150 = 0
    bestpenalty95 = 100
    bestpenalty150 = 100
    for i, phi in enumerate(phases):
        if (penalty95(model_name, phi)<bestpenalty95):
            bestpenalty95 = penalty95(model_name, phi)
            best_phase95 = phi

        if (penalty150(model_name, phi)<bestpenalty150):
            bestpenalty150 = penalty150(model_name, phi)
            best_phase150 = phi

    print('95GHz phase: %.1f, 150GHz phase: %.1f, combined penalty score: %f'%
        (best_phase95, best_phase150, bestpenalty95+bestpenalty150))
    return  best_phase95, best_phase150, bestpenalty95+bestpenalty150

# print('Optimization for the 1-stack HWP with 1 AR layer')
# x_mono1 = [1.1,0.1,3.]
# res_mono1 = minimize(penalty_mono1, x_mono1, method='BFGS', jac=grad_penalty_mono1, options={'disp':True})
# print('Indices, Thicknesses(mm)', res_mono1.x)

# print('Optimization for the 1-stack HWP with 2 AR layers')
# x_mono2 = [1.1, 2., 0.4, 0.3, 3.78]
# res_mono2 = minimize(penalty_mono2, x_mono2, method='BFGS', jac=grad_penalty_mono2, options={'disp':True})
# print('Indices, Thicknesses(mm)', res_mono2.x)

# print('Optimization for the 1-stack HWP with 3 AR layers')
# x_mono = [1.1, 2., 2.6, 0.4, 0.3, 0.2, 3.78]
# res_mono = minimize(penalty_monoref, x_mono, method='L-BFGS-B', jac=grad_penalty_monoref, options={'disp':True},
#     bounds=[(1., 3.1), (1., 3.1), (1., 3.1), (1e-3, 5), (1e-3, 5), (1e-3, 5), (1e-3, 5)])
# print('Indices, Thicknesses(mm), angle (°)', res_mono.x)

# print('Optimization for the 3-stack AHWP with 1 AR layer')
# x1 = [2.2, 0.4]
# res1 = minimize(penalty1, x1, method='BFGS', jac=grad_penalty1, options={'disp':True})
# print('Indices, Thicknesses(mm)', res1.x)

# print('Optimization for the 3-stack AHWP with 2 AR layer')
# x2 = [1.5, 2, .3,.3]
# res2 = minimize(penalty2, x2, method='BFGS', jac=grad_penalty2, options={'disp':True})
# print('Indices, Thicknesses(mm)', res2.x)

# print('Optimization for the 3-stack AHWP with 3 AR layer')
# x3 = [1.5, 2, 2.5, .3, .3, .3, 3.8, 50]
# res3 = minimize(penalty3, x3, method='L-BFGS-B', jac=grad_penalty3, options={'disp':True},
#     bounds=[(1., 3.1), (1., 3.1), (1., 3.1), (1e-3, 5), (1e-3, 5), (1e-3, 5), (1e-3, 5), (0,90)])
# print('Indices, Thicknesses(mm), angle (°)', res3.x)

# print('Optimization for the 5-stack AHWP with 3 AR layers')
# x3 = [1.5, 2, 2.5, .3, .3, .3, 3.8, 50]
# res5 = minimize(penalty5, x3, method='L-BFGS-B', jac=grad_penalty5, options={'disp':True}, 
#     bounds=[(1., 3.1), (1., 3.1), (1., 3.1), (1e-3, 5), (1e-3, 5), (1e-3, 5), (1e-3, 5), (0,90)])
# print('Indices, Thicknesses(mm)', res5.x)

print('Current penalty 3BR old, Dust: %f\n Current penalty 3BR new, Dust: %f\n Current penalty 5BR old, Dust: %f\n Current penalty 5BR new, Dust: %f\n' %
    (quick_penalty('3AR3BRcent', phase95=57.1, phase150=57.6), quick_penalty('3AR3BR_new', phase95=59.2, phase150=57.4) ,
    quick_penalty('3AR5BR', phase95=51.3, phase150=51.3) , quick_penalty('3AR5BR_new', phase95=53.5, phase150=53.0)))

print('3BR old, Dust:')
phase_through_penalty('3AR3BRcent')
print('3BR new, Dust:')
phase_through_penalty('3AR3BR_new')
print('5BR old, Dust:')
phase_through_penalty('3AR5BR')
print('5BR new, Dust:')
phase_through_penalty('3AR5BR_new')


