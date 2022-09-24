import numpy as np
import healpy as hp
import qpoint as qp

Q = qp.QPoint()

pi = np.pi
radeg = (180./pi)

'''
This module is used to generate scan strategies specifically for satellite missions.
Code based on pyScan written Tomotake Matsumura at IPMU. The below code fragments
have been added here with the authors permission.
See: https://github.com/tmatsumu/LB_SYSPL_v4.2

The below functions are invoked by functions in instrument.py, in particular
a function called l2_scan.

'''

def convert_Julian2Dublin(JD):
    '''
    Convert Julian dates (JD) to Dublin Julian dates (DJD).

    Arguments
    ---------
    JD : array-like
    	Julian dates are expressed as a Julian day number (number of solar days 
    	elapsed since noon Universal Time on Monday, January 1, 4713 BC, proleptic 
    	Julian calendar) with a decimal fraction added (representing the fraction 
    	of solar day since the preceding noon in Universal Time).

    Returns
    -------
    DJD : array-like
    	Dublin Julian dates are defined similarly to Julian dates, but starting
    	the count from noon Universal Time on December 31, 1899. DJD = 0 corresponds
    	to JD = 2415020.

    '''
    
    DJD = JD - 2415020.
    return np.array(DJD)

def wraparound_2pi(x):
    '''
    Wrap input angles to the interval [0, 2pi).

    Arguments
    ---------
    x : array-like
    	angles in radians.

    Returns
    -------
    arr : array-like
    	corresponding angles in radians in the interval [0, 2pi).

    '''

    if len(x)==1: n = np.int(x/(2.*pi))
    if len(x)>1: n = np.int_(x/(2.*pi))
    arr = x - n * 2.*pi
    return arr

def wraparound_npi(x, n_):
    '''
    Wrap input angles to the interval [0, n*pi).

    Arguments
    ---------
    x : array-like
    	angles in radians.

    Returns
    -------
    arr : array-like
    	corresponding angles in radians in the interval [0, n*pi).

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
    Roughly estimate the phase of the Sun on the sky at given times.

    Arguments
    ---------
    DJD : array-like
    	dates in Dublin Julian format.

    Notes
    -----
    It also returns a numpy array of zeros with same length as the input, representing 
    the Sun's polar angle.
    
    '''

    freq = 1./((365.25)*24.*60.*60.)
    phi = wraparound_2pi(2.*pi*freq*DJD*(24*60*60))
    return phi, np.zeros(len(phi))

def matrix2x2_multi_xy(x, y, phi):
    '''
    Given a set of vectors on the plane xy (in terms of their x and y components), 
    it rotates each of them by some angle around the z axis and returns the rotated 
    components.
    
    Arguments
    ---------
    x : array-like
    	each element represents the x component of a vector
    
    y : array-like
    	each element represents the y component of a vector
    
    phi : array-like
    	rotation angles in radians
    
    '''
    
    cp, sp = np.cos(phi), np.sin(phi)
    return cp*x - sp*y, sp*x + cp*y

def matrix2x2_multi_xz(x,  z, theta):
    '''
    Given a set of vectors on the plane xz (in terms of their x and z components), 
    it rotates each of them by some angle around the y axis and returns the rotated 
    components.
    
    Arguments
    ---------
    x : array-like
    	each element represents the x component of a vector
    
    z : array-like
    	each element represents the y component of a vector
    
    theta : array-like
    	rotation angles in radians
    
    '''
    
    ct, st = np.cos(theta), np.sin(theta)
    return ct*x + st*z, -st*x + ct*z

def cosangle(xi, yi, zi, xii, yii, zii):
    '''
    Calculate the angles (in radians) between two sets of 3D vectors, given in 
    terms of their x, y and z components.
    
    '''
    
    return np.arccos((xi*xii+yi*yii+zi*zii)/np.sqrt(xi*xi+yi*yi+zi*zi)\
          /np.sqrt(xii*xii+yii*yii+zii*zii))

def deriv_theta(xi, yi, zi):
    '''
    For each vector (x,y,z), it returns the components of a unit vector orthogonal to 
    (x,y,z). In particular, denoting the initial vector in spherical coordinates as 
    (r,theta,phi), the output vector is (1,theta+pi/2,phi).
    
    '''
    
    theta = np.arctan(np.sqrt(xi*xi+yi*yi)/zi)+5.*pi	# in the range [4.5*pi,5.5*pi]
    theta = wraparound_npi(theta,1)			# in the range [0,pi)
    phi = np.arctan2(yi,xi)+10.*pi	# in the range [9*pi,11*pi]
    phi = wraparound_npi(phi,2)	# in the range [0,2*pi)
    ct = np.cos(theta)
    return ct * np.cos(phi), ct * np.sin(phi), -np.sin(theta)

def deriv_phi(xi, yi, zi):
    '''
    For each vector (x,y,z), it returns the components of a unit vector orthogonal to 
    (x,y,0). In particular, denoting the initial vector in spherical coordinates as 
    (r,theta,phi), the output vector is (1,0,phi+pi/2).
    
    '''
    
    theta = np.arctan(np.sqrt(xi*xi+yi*yi)/zi);
    theta = wraparound_npi(theta,1);
    phi = np.arctan2(yi,xi)+10*pi;	# in the range [9*pi, 11*pi]
    phi = wraparound_npi(phi,2);	# in the range [0,2*pi)
    return -np.sin(phi), np.cos(phi), phi*0.

def LB_rotmatrix_multi2(theta_asp, phi_asp,
        theta_antisun, theta_boresight, omega_pre, omega_spin, time):
    '''
    Calculate the Cartesian coordinates of the boresight unit vector at given times, 
    together with the psi angle. Return a (4,nobs) array, "out", such that out[0,:]=x, 
    out[1,:]=y, out[2,:]=z and out[3,:]=psi.
    
    Arguments
    ---------
    theta_asp : array-like
        polar angle of the anti-Sun position in radians
        
    phi_asp : array-like
        azimuthal angle of the anti-Sun position in radians
    
    theta_antisun: float
        anti-Sun angle of the scanning strategy in radians
        
    theta_boresight : float
        boresight angle of the scanning strategy in radians
    
    omega_pre : float
        precession angular frequency in radians/sec
    
    omega_spin : float
        spin angular frequency in radians/sec
    
    time : array-like
        dates in Unix format
         
    Notes
    -----
    This function is called in the definition of ctime2bore(), where phi_asp is the 
    Sun's phase returned by sun_position_quick() and theta_asp is pi/2.

    '''

    out = np.empty((4, theta_asp.shape[0]))

    # Components of the anti-Sun unit vector
    x = np.sin(theta_asp) * np.cos(phi_asp)
    y = np.sin(theta_asp) * np.sin(phi_asp)
    z = np.cos(theta_asp)
    # NOTE: in ctime2bore(), theta_asp = pi/2 and therefore 
    # x = np.cos(phi_asp)
    # y = np.sin(phi_asp)
    # z = 0

    # Fixed offset values of precession and spin phases
    phi_prec_init_phase = 3./2.*pi
    phi_spin_init_phase = pi

    # Precession and spin angles
    omega_pre_t = omega_pre*time + phi_prec_init_phase
    omega_spin_t = omega_spin*time + phi_spin_init_phase
    
    # Angle between x axis and projection on the xy plane of the anti-Sun direction,
    # it takes values in [0,2pi)
    rel_phi = np.where(y>=0, np.arccos(x), -np.arccos(x)+2*pi)

    # When this function is called within ctime2bore(), these operations return the
    # Cartesian components of the unit vector (1,theta_boresight,omega_spin_t): the
    # boresight unit vector in a coordinate system where the spin axis lies along z
    x, y = matrix2x2_multi_xy( x, y,  -rel_phi)
    x, z = matrix2x2_multi_xz( x, z,  -pi/2.)
    x, z = matrix2x2_multi_xz( x, z, theta_boresight)
    x, y = matrix2x2_multi_xy( x, y, omega_spin_t)

    xii = x.copy()
    yii = y.copy()
    zii = z.copy()

    # When this function is called within ctime2bore(), these operations return the
    # Cartesian components of the boresight unit vector
    x, z = matrix2x2_multi_xz( x, z, theta_antisun)
    x, y = matrix2x2_multi_xy( x, y, omega_pre_t)
    x, z = matrix2x2_multi_xz( x, z, pi/2.)
    x, y = matrix2x2_multi_xy( x, y, rel_phi)

    out[0,:] = x
    out[1,:] = y
    out[2,:] = z

    # When this function is called within ctime2bore(), these operations amount to 
    # calculate the psi angle
    xii, yii, zii = deriv_phi(xii,yii,zii)
    xii, zii = matrix2x2_multi_xz( xii, zii, theta_antisun) # 4B->8
    xii, yii = matrix2x2_multi_xy( xii, yii, omega_pre_t) # 8->9
    xii, zii = matrix2x2_multi_xz( xii, zii, pi/2.) # 9->10
    xii, yii = matrix2x2_multi_xy( xii, yii, rel_phi) # 10->11

    xo, yo,zo = deriv_theta(x,y,z)
    fpout_psi_theta = cosangle(xo,yo,zo,xii,yii,zii)

    xo, yo, zo = deriv_phi(x,y,z)
    fpout_psi_phi = cosangle(xo,yo,zo,xii,yii,zii)

    fpout_psi_theta = np.where(((fpout_psi_theta > 0.) & (fpout_psi_theta <= 1./2.*pi))
                               & ((fpout_psi_phi > 1./2.*pi) & (fpout_psi_phi <= pi)),
                               -fpout_psi_theta, fpout_psi_theta)
    fpout_psi_theta = np.where(((fpout_psi_theta>1./2.*pi) & (fpout_psi_theta<=pi))
                               & ((fpout_psi_phi>1./2.*pi) & (fpout_psi_phi<=pi)),
                               -fpout_psi_theta, fpout_psi_theta)
    out[3,:] = fpout_psi_theta
    return out


def ctime2DJD(ctime):
    '''
    Convert ctime dates to Dublin Julian dates (DJD).

    Arguments
    ---------
    ctime : array-like
    	Seconds elapsed since Jan 1 1970 GMT.

    Returns
    -------
    DJD : array-like
    	Days elapsed since noon Universal Time on December 31, 1899.

    '''
    
    # ctime/86400. : ctime date in days
    # 2440587.5    : ctime's zero in JD format
    # -2415020     : JD's zero in DJD format 
    return ctime / 86400. + (2440587.5 - 2415020)

def ctime2bore(ctime, theta_antisun=45., theta_boresight=50.,
    period_antisun=192.348, rate_boresight=0.05):
    '''
    Generate boresight quaternion at some Unix time, by feeding LiteBIRD-specific 
    arguments to LB_rotmatrix_multi2().     

    Arguments
    ---------
    ctime : ndarray
        Unix time vector

    Keyword arguments
    -----------------
    theta_antisun : float
        theta anti-Sun angle in degrees of the scanning strategy 
        (default : 45.)
    theta_boresight : float
        theta boresight angle in degrees of the scanning strategy
        (default : 50.)
    period_antisun : float
        rotation period of theta anti-Sun in minutes
        (default : 192.348)
    rate_boresight : float
        rotation rate of theta boresight in rpm
        (default : 0.05)

    '''
    
    # Angles in radians and frequencies in 1/sec
    theta_antisun = np.radians(theta_antisun)
    theta_boresight = np.radians(theta_boresight)
    freq_antisun = 1. / (period_antisun * 60.)
    freq_boresight = rate_boresight / 60.

    # Calculate Sun position at input Unix time
    DJD = convert_Julian2Dublin(ctime2DJD(ctime))
    phi_asp, theta_asp = sun_position_quick(DJD)

    # From ecliptic (lat,lon) convention to (theta,phi) convention
    theta_asp = pi/2. - theta_asp

    # Angular frequencies in radians/sec
    omega_pre = 2. * pi * freq_antisun
    omega_spin = 2. * pi * freq_boresight

    # Return the boresight pointings
    p_out = LB_rotmatrix_multi2(theta_asp, phi_asp, theta_antisun,
        theta_boresight, omega_pre, omega_spin, ctime)

    # Calculate theta and phi from Cartesian coordinates
    theta_out = np.arctan2(np.sqrt(p_out[0,:]**2 + p_out[1,:]**2), p_out[2,:])
    phi_out = np.arctan2(p_out[1,:],p_out[0,:])

    theta_out = wraparound_npi(theta_out, 1.)
    phi_out = wraparound_npi(phi_out, 2.)
    psi_out = wraparound_2pi(p_out[3, :])

    # From (theta,phi) to (ra,dec) convention
    # Also, all angles are converted in degrees
    ra = np.degrees(phi_out) - 180.
    dec = np.degrees(theta_out) - 90.
    psi = np.degrees(psi_out)

    # Calculate the quaternion
    q_bore = Q.radecpa2quat(ra, dec, psi)

    return q_bore

