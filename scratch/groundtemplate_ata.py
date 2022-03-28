import healpy as hp
hp.disable_warnings()
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numpy.random import default_rng

h = 6.62e-34
c = 3e8
k_b = 1.38e-23
nside=512
#scope
freq=95e9
deltanu = 30.e9
rng = default_rng(123456789)
sigmaT = 5e6
def tb2b(tb, nu):
    #Convert blackbody temperature to spectral
    x = h*nu/(k_b*tb)
    return 2*h*nu**3/c**2/(np.exp(x) - 1)

def s2tcmb(s_nu, nu):
    #Convert spectral radiance s_nu at frequency nu to t_cmb
    T_cmb = 2.72548 #K from Fixsen, 2009, ApJ 707 (2): 916–920   
    x = h*nu/(k_b*T_cmb)
    slope = 2*k_b*nu**2/c**2*((x/2)/np.sinh(x/2))**2
    return s_nu/slope

def dBdT(tb, nu):
    x = h*nu/(k_b*tb)
    slope = 2*k_b*nu**2/c**2*((x/2)/np.sinh(x/2))**2
    return slope


def band_avg_tcmb(tb, nu_c, frac_bwidth=.2):
    T_cmb = 2.72548 
    power = integrate.quad(lambda nu: tb2b(tb, nu), nu_c*(1-frac_bwidth/2.),
        nu_c*(1+frac_bwidth/2.))
    correction = integrate.quad(lambda nu: dBdT(T_cmb, nu), nu_c*(1-frac_bwidth/2.),
        nu_c*(1+frac_bwidth/2.))
    return power[0]/correction[0]

def boundary(az, el, hwidth, inmap):
    #Convert az, el, width to radians
    raz = np.radians(az)
    rwidth = np.radians(hwidth)
    slope = np.radians(el)/rwidth
    #Get all coordinates
    nside = hp.npix2nside(inmap.size)
    pix_array = np.arange(inmap.size)
    theta, phi = hp.pix2ang(nside, pix_array)
    #Get the rings at constant theta over the horizon
    unique_theta, unique_theta_idx = np.unique(theta, return_index=True)
    unique_theta_idx = unique_theta_idx[np.abs(unique_theta-np.radians(82.5))<=np.radians(7.5)]
    unique_theta = unique_theta[np.abs(unique_theta-np.radians(82.5))<=np.radians(7.5)]
    idx_intervals = np.array([unique_theta_idx[:-1], unique_theta_idx[1:]])
    boundary = []
    for i in range(el.size):#Model each terrain feature
        az0 = raz[i]-rwidth[i] 
        az1 = raz[i]+rwidth[i] 
        condition = ((slope[i]*(phi-az0)>np.pi/2.-theta)*
            (slope[i]*(az1-phi)>np.pi/2.-theta)*(theta<np.pi/2.)).astype(bool)
        #Get the intersection between the ring and the feature
        for j in range(unique_theta.size-1):
            ring = condition[idx_intervals[0,j]:idx_intervals[1,j]+1].astype(int)
            mstretch = np.nonzero(ring)
            if mstretch[0].size>0:
                boundary.append(unique_theta_idx[j]+mstretch[0][0])
                boundary.append(unique_theta_idx[j]+mstretch[0][-1])
    return boundary

#Peaks
az = np.array([40,65,90,120,165., 290., 325, 345])
el = np.array([15,12,5,5,3,-3,2,2])
hwidth = np.array([40.,30.,30.,15,30.,40.,10.,10.])
raz = np.radians(az)
rwidth = np.radians(hwidth)
slope = np.radians(el)/rwidth
#Map
npix = hp.nside2npix(nside)
pixarea= hp.nside2pixarea(nside)
gtemplate = np.empty(npix)
theta, phi = hp.pix2ang(nside, np.arange(npix))
#Temperature templates
t_cmb_cerro = band_avg_tcmb(276, freq, .2)*1e6#uK
dcerro = 400./np.tan(np.radians(15))
t_cmb_ground = band_avg_tcmb(280, freq, .2)*1e6#uK
#ground temperature
npix_under_horizon = len(gtemplate[theta>=np.pi/2.])
gtemplate[:] = np.exp(- 0.5 * (theta - np.pi / 2) ** 2 / np.radians(0.3) ** 2) * t_cmb_ground#Adri fix
gtemplate[theta>=np.pi/2.] = t_cmb_ground
#gtemplate[theta>=np.pi/2.]=rng.normal(loc=t_cmb_ground, scale=sigmaT, 
#    size= npix_under_horizon)

#Define mountains and temperatures
for i in range(el.size): 
    az0 = raz[i]-rwidth[i] 
    az1 = raz[i]+rwidth[i] 
    condition = ((slope[i]*(phi-az0)>np.pi/2.-theta)*
        (slope[i]*(az1-phi)>np.pi/2.-theta)*(theta<np.pi/2.)).astype(bool) 
    gtemplate[condition] = t_cmb_ground+(t_cmb_cerro-t_cmb_ground)/400.*dcerro*np.tan(np.pi/2.-theta[condition])
    """
    gtemplate[condition]= rng.normal(
        loc = (t_cmb_ground +(t_cmb_cerro-t_cmb_ground)/400.
                                *dcerro*np.tan(np.pi/2.-theta[condition])),
        scale = sigmaT)
    """

#Apply the edge treatment to the boundary of the mountains
boundary = np.array(boundary(az, el, hwidth, gtemplate))
for pix in boundary:
    gtemplate = np.maximum(gtemplate, (np.exp(-0.5*((theta - theta[pix])**2+(phi-phi[pix])**2) / np.radians(0.3)**2) 
        * (t_cmb_ground+(t_cmb_cerro-t_cmb_ground)/400.*dcerro*np.tan(np.pi/2.-theta[pix]))))

hp.mollview(gtemplate, cmap='plasma', flip='geo', rot=(180,0,0), 
    min=3e8, max=3.51e8, unit=r"T $(\mu K_{CMB})$", bgcolor='#FFFFFF')#
#plt.title("Blackbody ground template at 95GHz, 20% bandwidth")
hp.graticule(dpar=15, dmer=30)
#plt.savefig("ata_95.png", transparent=True)
plt.show()
#hp.write_map('ground_ata_95_variance5.fits', gtemplate, overwrite=True)
hp.write_map('ground_ata95Adri.fits', gtemplate, overwrite=True)