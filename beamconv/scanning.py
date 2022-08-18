import numpy as np
import healpy as hp
import qpoint as qp

Q = qp.QPoint()

pi = np.pi
radeg = (180./pi)

'''
This module is used to generate scan strategies specifically for satellite missions.
Code based on pyScan written Tomotake Matsumura at IPMU.
See: https://github.com/tmatsumu/LB_SYSPL_v4.2
'''

def convert_Julian2Dublin(JD):
    '''
    Calculate the Julian day (JD) to Dublin Julian day (DJD)

    Arguments
    ---------
    JD : array-like

    Returns
    -------
    DJD : array-like

    '''
    DJD = JD - 2415020.
    return np.array(DJD)

def wraparound_2pi(x):
    '''
    Wraps around x so that it falls within the proper range in multiples of 2pi

    Arguments
    ---------
    x : array-like

    Returns
    -------
    arr : array-like

    '''

    if len(x)==1: n = np.int(x/(2.*pi))
    if len(x)>1: n = np.int_(x/(2.*pi))
    arr = x - n * 2.*pi
    return arr

def wraparound_npi(x, n_):
    '''
    Wraps around x so that it falls within the proper range in multiples of 2pi

    Arguments
    ---------
    x : array-like

    Returns
    -------
    arr : array-like

    '''

    n_ = float(n_)
    if len(x)==1:
        n_wrap = np.int(x/(n_*pi))
        if x<0: n_wrap-=1
    if len(x)>1:
        n_wrap = np.int_(x/(n_*pi))
        ind = np.where((x<0))
        if len(ind[0]) != 0: n_wrap[ind[0]]-=1
    return x-n_wrap*n_*pi

def sun_position_quick(DJD):
    '''
    Roughly estimates the phase of the sun on the sky

    Arguments
    ---------
    DJD : array-like

    Returns
    -------
    arr : array-like

    '''

    freq = 1./((365.25)*24.*60.*60.)
    phi = wraparound_2pi(2.*pi*freq*DJD*(8.64*1.e4))
    return phi, np.zeros(len(phi))

def matrix2x2_multi_xy(x,  y, phi):
    '''
    Roughly estimates the phase of the sun on the sky

    Arguments
    ---------
    x : array-like
    y : array-like
    phi : array-like

    Returns
    -------
    arr : array-like

    '''
    cp, sp = np.cos(phi), np.sin(phi)
    return cp*x - sp*y, sp*x + cp*y

def matrix2x2_multi_xz(x,  z, theta):
    ct, st = np.cos(theta), np.sin(theta)
    return ct*x + st*z, -st*x + ct*z

def cosangle(xi, yi, zi, xii, yii, zii):
    return np.arccos((xi*xii+yi*yii+zi*zii)/np.sqrt(xi*xi+yi*yi+zi*zi)\
          /np.sqrt(xii*xii+yii*yii+zii*zii))

def deriv_theta(xi, yi, zi):
    theta = np.arctan(np.sqrt(xi*xi+yi*yi)/zi)+5.*pi # -pi/2 ~ pi/2
    theta = wraparound_npi(theta,1)
    phi = np.arctan2(yi,xi)+10.*pi #  -pi ~ pi
    phi = wraparound_npi(phi,2)
    ct = np.cos(theta)
    return ct * np.cos(phi), ct * np.sin(phi), -np.sin(theta)

def deriv_phi(xi, yi, zi):
    theta = np.arctan(np.sqrt(xi*xi+yi*yi)/zi);
    theta = wraparound_npi(theta,1);
    phi = np.arctan2(yi,xi)+10*pi; # -pi ~ pi
    phi = wraparound_npi(phi,2);
    return -np.sin(phi), np.cos(phi), phi*0.

def LB_rotmatrix_multi2(theta_asp, phi_asp,
        theta_antisun, theta_boresight, omega_pre, omega_spin, time):
    '''
    Rotation matrix multiplication for standarization

    '''

    out = np.empty((4, theta_asp.shape[0]))

    x = np.sin(theta_asp) * np.cos(phi_asp)
    y = np.sin(theta_asp) * np.sin(phi_asp)
    z = np.cos(theta_asp)

    # this is the temporary offset to compare with the Guillaume's code
    phi_prec_init_phase = 3./2.*pi
    phi_spin_init_phase = pi

    rel_phi = np.where(y>=0, np.arccos(x), -np.arccos(x)+2*pi)

    omega_pre_t = omega_pre*time + phi_prec_init_phase
    omega_spin_t = omega_spin*time + phi_spin_init_phase

    x, y = matrix2x2_multi_xy( x, y,  -rel_phi) # 1->2
    x, z = matrix2x2_multi_xz( x, z,  -pi/2.) # 2->3
    x, z = matrix2x2_multi_xz( x, z, theta_boresight) # 3->4A
    x, y = matrix2x2_multi_xy( x, y, omega_spin_t) # 4A->4B

    # pass the variable to calculate the psi angle:
    # psi is defined as the phi direction when the focal plane
    # is rotating about the spin axis
    xii = x.copy()
    yii = y.copy()
    zii = z.copy()

    # complete the rest of the boresight pointing calculation
    x, z = matrix2x2_multi_xz( x, z, theta_antisun) # // 4B->8
    x, y = matrix2x2_multi_xy( x, y, omega_pre_t) # 8->9
    x, z = matrix2x2_multi_xz( x, z, pi/2.) # 9->10
    x, y = matrix2x2_multi_xy( x, y, rel_phi) # 10->11

    out[0,:] = x
    out[1,:] = y
    out[2,:] = z

    # pick up the remaining calculation for psi
    xii, yii, zii = deriv_phi(xii,yii,zii)
    xii, zii = matrix2x2_multi_xz( xii, zii, theta_antisun) # 4B->8
    xii, yii = matrix2x2_multi_xy( xii, yii, omega_pre_t) # 8->9
    xii, zii = matrix2x2_multi_xz( xii, zii, pi/2.) # 9->10
    xii, yii = matrix2x2_multi_xy( xii, yii, rel_phi) # 10->11

    xo, yo,zo = deriv_theta(x,y,z)
    fpout_psi_theta = cosangle(xo,yo,zo,xii,yii,zii)

    xo, yo, zo = deriv_phi(x,y,z)
    fpout_psi_phi = cosangle(xo,yo,zo,xii,yii,zii)

    fpout_psi_theta = np.where(((fpout_psi_theta>0.)
                            & (fpout_psi_theta<=1./2.*pi))
                           & ((fpout_psi_phi>1./2.*pi)
                            & (fpout_psi_phi<=pi)),
                              -fpout_psi_theta,
                               fpout_psi_theta)
    fpout_psi_theta = np.where(((fpout_psi_theta>1./2.*pi)
                            & (fpout_psi_theta<=pi))
                           & ((fpout_psi_phi>1./2.*pi)
                            & (fpout_psi_phi<=pi)),
                              -fpout_psi_theta,
                               fpout_psi_theta)
    out[3,:] = fpout_psi_theta
    return out


def ctime2DJD(ctime):
    '''
    Calculating the Dublin JD

    Arguments
    -------
    ctime : ndarray
        Unix time array. Size = (end - start)

    Returns
    -------
    DJD : array-like


    Note:
    ctime (Unix) time is defined as Jan 1 1970 GMT
    First number in paranthesis is number of days since 12h Jan 1, 4713 BC [JD]
    Second number is to convert to Dublin JD [12h December 31, 1899]
    '''

    return ctime / 86400. + (2440587.5 - 2415020)

def ctime2bore(ctime, theta_antisun=45., theta_boresight=50.,
    freq_antisun=192.348, freq_boresight=0.314):
    '''
    Generate boresight quaternion

    Arguments
    ---------
    ctime : ndarray
        The Unix time vector


    Keyword arguments
    -----------------
    theta_antisun : float
        The theta anti-sun angle in degrees of the scanning strategy
        (default : 45.)
    theta_boresight : float
        The theta boresight angle in degrees of the scanning strategy
        (default : 50.)
    freq_antisun : float
        The rotation frequency of theta anti-sun in units of 1/min
        (default : 192.348)
    freq_boresight : float
        The rotation frequency of theta boresight in units of radians/min
        (default : 0.314)


    '''

    theta_antisun = np.radians(theta_antisun)
    theta_boresight = np.radians(theta_boresight)
    freq_antisun = 1. / (freq_antisun * 60.)
    freq_boresight = freq_boresight / 60.

    # cal sun position
    DJD = convert_Julian2Dublin(ctime2DJD(ctime))
    phi_asp, theta_asp = sun_position_quick(DJD)

    # from ecliptic lat, lon convention to theta, phi convention
    theta_asp = pi/2. - theta_asp

    omega_pre = 2. * pi * freq_antisun
    omega_spin = 2. * pi * freq_boresight

    p_out = LB_rotmatrix_multi2(theta_asp, phi_asp, theta_antisun,
        theta_boresight, omega_pre, omega_spin, ctime)

    theta_out = np.arctan2(np.sqrt(p_out[0,:]**2 + p_out[1,:]**2), p_out[2,:])
    phi_out = np.arctan2(p_out[1,:],p_out[0,:])

    theta_out = wraparound_npi(theta_out, 1.)
    phi_out = wraparound_npi(phi_out, 2.)
    psi_out = wraparound_2pi(p_out[3, :])

    ra = np.degrees(phi_out) - 180.
    dec = np.degrees(theta_out) - 90.
    psi = np.degrees(psi_out)

    q_bore = Q.radecpa2quat(ra, dec, psi)

    return q_bore

def main():

    ####################################################
    # All the scanning parameters
    ####################################################

    import time
    import healpy as hp

    theta_antisun = 45      # degrees
    theta_boresight = 50    # degrees
    freq_antisun = 192.348  # minutes
    freq_boresight = 0.314  # radians per minutes
    siad = 86400.            # seconds in a day
    today_julian = 1        # I just set 1 by hand
    sample_rate = 19.1      # Hz
    ydays = 20              # duration of the mission (in days)
    nside = 256
    runtime_i = time.time()

    theta_antisun = theta_antisun/radeg
    theta_boresight = theta_boresight/radeg
    freq_antisun = 1./(freq_antisun*60.)
    freq_boresight = freq_boresight/60.

    ####################################################
    # This is a minimal version of the gen_scan_c_mod2
    # function of pyScan
    ####################################################

    # we don't really need the next two lines, I just
    # wanted to look at the nhits map
    npix = hp.nside2npix(nside)
    nhits = np.zeros(npix)
    theta_array = np.zeros(int(sample_rate * siad * ydays))
    phi_array = np.zeros(int(sample_rate * siad * ydays))

    for i in range(ydays):
        # define time related variables
        n_time = int(siad * sample_rate)
        sec2date = (1.e-4/8.64)
        time_julian = np.arange(n_time)/(n_time-1.)*siad*sec2date + today_julian # time in julian [day]

        # cal sun position
        DJD = convert_Julian2Dublin(time_julian)
        phi_asp, theta_asp = sun_position_quick(DJD)
        DJD = 0

        time_i = time_julian /sec2date # time in [sec]

        # from ecliptic lat, lon convention to theta, phi convention
        theta_asp = pi/2.-theta_asp

        omega_pre = 2.*pi*freq_antisun
        omega_spin = 2.*pi*freq_boresight

        p_out = LB_rotmatrix_multi2(theta_asp, phi_asp, theta_antisun,
            theta_boresight, omega_pre, omega_spin, time_i)

        theta_out = np.arctan2(np.sqrt(p_out[0,:]**2 + p_out[1,:]**2), p_out[2,:])
        phi_out = np.arctan2(p_out[1,:],p_out[0,:])

        theta_out = wraparound_npi(theta_out, 1.)
        phi_out = wraparound_npi(phi_out, 2.)

        theta_array[i*int(sample_rate*siad):(i+1)*int(sample_rate*siad)] = theta_out
        phi_array[i*int(sample_rate*siad):(i+1)*int(sample_rate*siad)] = phi_out

        ####################################################
        # we don't really need the next lines, I just
        # wanted to look at the nhits map
        nbPix = hp.nside2npix(nside)
        ipix = hp.ang2pix(nside,theta_out,phi_out)
        beta=0; dphivec=0; dpvec=0; dtvec=0; theta_out=0; phi_out=0
        nhits += np.bincount(ipix, minlength=nbPix)
        ipix=0
        runtime_f = time.time()
        print('day ', (i+1), ':', (time.time()-runtime_i), 'sec')
        print()

        today_julian += 1.

    filename_out = str(ydays) + '_days'
    hp.mollview(nhits, title= str(ydays) + ' days', rot=[0.,0.])
    hp.graticule(dpar=10,dmer=10,coord='E')
    py.savefig(filename_out+'.png')

if __name__ == '__main__':
    main()

