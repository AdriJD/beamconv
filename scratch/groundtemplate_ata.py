import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
az = np.array([40,65,90,120,165., 290., 325, 345])
el = np.array([15,12,5,5,3,-3,2,2])
hwidth = np.array([40.,30.,30.,15,30.,40.,10.,10.])
nside=512
npix = hp.nside2npix(nside)
gtemplate = np.empty(npix)

theta, phi = hp.pix2ang(nside, np.arange(npix))
raz = np.radians(az)
rwidth = np.radians(hwidth)
slope = (np.radians(el))/rwidth


gtemplate[theta>np.pi/2.]=290
for i in range(el.size): 
    az0 = raz[i]-rwidth[i] 
    az1 = raz[i]+rwidth[i] 
    condition = ((slope[i]*(phi-az0)>np.pi/2.-theta)*
        (slope[i]*(az1-phi)>np.pi/2.-theta)*(phi>az0)*(phi<az1)).astype(bool) 
    gtemplate[condition]=290*(theta/np.pi) 
hp.mollview(gtemplate)
plt.show()