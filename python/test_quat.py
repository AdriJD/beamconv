import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import qpoint as qp
import healpy as hp
import quaternion as qt
import time

Q = qp.QMap(fast_pix=True)
nside = 64
q_off = Q.det_offset(0, 0, 0)
print q_off
ctime = time.time()
roll = 89

az_off = 0
el_off = 70

q_bore = Q.azel2bore(0, 0, None, None, 0, -60, ctime)
pix, sin2psi, cos2psi = Q.bore2pix(q_off, np.array([ctime]), q_bore, nside=nside, pol=True)

q_off = Q.det_offset(az_off, el_off, 0)
q = qt.as_quat_array(q_off)
q2 = qt.as_quat_array(Q.det_offset(0, 0, roll))

q3 = q2 * q #* np.conj(q2)
q_off = qt.as_float_array(q3)
q_off /= np.sqrt(np.sum(q_off**2))
pix2, _, _ = Q.bore2pix(q_off, np.array([ctime]), q_bore, nside=nside, pol=True)


az_off2 = np.cos(np.radians(roll)) * az_off - np.sin(np.radians(roll)) * el_off
el_off2 = np.sin(np.radians(roll)) * az_off + np.cos(np.radians(roll)) * el_off
q_off2 = Q.det_offset(az_off2, el_off2, 0)
pix3, _, _ = Q.bore2pix(q_off2, np.array([ctime]), q_bore, nside=nside, pol=True)

testmap = np.zeros(12 * nside**2)
testmap[pix] = 1.
testmap[pix2] = 2.
testmap[pix3] = -3.


plt.figure()
hp.mollview(testmap, coord='G')
plt.savefig('test.png')
plt.close()

Q.init_detarr(q_off)
