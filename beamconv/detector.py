import os
import numpy as np
import healpy as hp
from beamconv import tools
from . import transfer_matrix as tm

class HWP(object):
    '''
    A class representing the half wave plate, as a transfermatrix stack on which operations are done
    '''
    def __init__(self, stack=None):
        #needs a fix before deployment.
        self.stack = stack

    def __call__(self):
        return self

    def _stack_builder(self, thicknesses, indices, losses, angles):

        """
        Creates a stack of materials, as defined in transfer_matrix.py
        Inputs are:
        thicknesses   - (float) thicknesses in mm
        indices       - ordinary and extraordinary index
        losses        - ratios of the imaginary part of the dielectric constant to the real part.
        angles        - (float) radian angle between (extraordinary) axis and stack axis. 
        """

        if (thicknesses.size != angles.size or 2*thicknesses.size!=indices.size or 2*thicknesses.size!=losses.size):
            raise ValueError('There is a mismatch in the sizes of the inputs for the HWP stack')

        #Make a list of materials, with a name that corresponds to their position in the stack
        material_stack=[]
        for i in range(thicknesses.size):
            
            if (indices[i,0]==indices[i,1] and losses[i,0]==losses[i,1]):
                isotro_str = 'isotropic'
            else:
                isotro_str = 'uniaxial'

            material_stack.append(tm.material(indices[i,0], indices[i,1], 
                losses[i,0], losses[i,1], str(i), materialType=isotro_str))

        self.stack = tm.Stack( thicknesses*tm.mm, material_stack, angles)

    def _choose_HWP_model(self, model_name):
        '''
        Set a particlar stack from a few predefined models
        '''
        sapphire = tm.material( 3.07, 3.41, 2.3e-4, 1.25e-4, 'Sapphire', materialType='uniaxial')
        duroid_a = tm.material( 1.55, 1.55, 0.5e-4, 0.5e-4, 'RT Duroid', materialType='isotropic')
        duroid_b = tm.material( 1.715, 1.715, 1.2e-3, 1.2e-3, 'RT Duroid', materialType='isotropic')
        duroid_c = tm.material( 2.52, 2.52, 56.6e-4, 56.5e-4, 'RT Duroid', materialType='isotropic')
        duroid_d = tm.material( 1.951, 1.951, 1.2e-3, 1.2e-3, 'RT Duroid', materialType='isotropic')
        if (model_name=='Ar+HWP+Ar'):
            thicknesses = [0.305*tm.mm, 3.15*tm.mm, 0.305*tm.mm]
            angles   = [0.0, 0.0, 0.0]
            materials   = [duroid_b, sapphire, duroid_b]
        
        elif (model_name=='HWP_only'):
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
            raise ValueError('Unknown type of HWP entered')

        self.stack = tm.Stack( thicknesses, materials, angles)

    def _compute4params(self, freq, alpha):
        '''
        Compute the parameters for the unrotated Mueller Matrix
        '''
        Mueller = tm.Mueller(self.stack, freq, alpha, 0., reflected=False)
        T = Mueller[0,0]
        rho= Mueller[0,1]/ Mueller[0,0]
        c =  Mueller[2,2]/ Mueller[0,0]
        s =  Mueller[3,2]/ Mueller[0,0]

        return np.array([T,rho,c,s])

    def _topRowMuellerMatrix(self, psi=0.0, xi=0.0, theta=0.0, 
                             hwp_params=None):
        '''
        Compute the top row of the full HWP+polang+boresight Mueller Matrix
        '''
        eta = 1. ## co-polar quantity (FREEZE) 
        delta = 0. ## cross-polar quantity (FREEZE) 
        gamma = (eta**2-delta**2)/(eta**2+delta**2) ## polarization efficienty (FREEZE)
        H = 0.5*(eta**2+delta**2)
        ## Ideal case: H = 0.5

        T = hwp_params[0]
        rho = hwp_params[1]
        c = hwp_params[2]
        s = hwp_params[3]
        #print T, rho, c, s
        MII = H*T*(1+(gamma*rho*np.cos(2*(theta+xi))))
        MIQ = H*T*(rho*np.cos(2*(theta+psi)) + (0.5*(1+c)*gamma*np.cos(2*(psi-xi))) 
            + (0.5*(1-c)*gamma*np.cos(2*(2*theta+xi+psi))))
        MIU = H*T*(rho*np.sin(2*(theta+psi)) + (0.5*(1+c)*gamma*np.sin(2*(psi-xi))) 
            + (0.5*(1-c)*gamma*np.sin(2*(2*theta+xi+psi))))
        MIV = H*T*(s*gamma*np.sin(2*(theta+xi)))

        # IPPV base
        MIP = 0.5*(MIQ-1j*MIU) 
        MIP_t = 0.5*(MIQ+1j*MIU) 

        #return MII, MIQ, MIU 
        return MII, MIP, MIP_t 


class Beam(object):
    '''
    A class representing detector and beam properties.
    '''
    def __init__(self, az=0., el=0., polang=0., name=None,
                 pol='A', btype='Gaussian', fwhm=None, lmax=700, mmax=None, sensitive_freq = 150e9,
                 dead=False, ghost=False, amplitude=1., po_file=None,
                 eg_file=None, cross_pol=True, deconv_q=True,
                 normalize=True, polang_error=0., idx=None,
                 symmetric=False, hwp=HWP(), hwp_precomp_mueller=None):
        '''
        Initialize a detector beam.

        Keyword arguments
        -----------------
        az : float
            Azimuthal location of detector relative to boresight
            in degrees (default : 0.)
        el : float
            Elevation location of detector relative to boresight
            in degrees (default : 0.)
        polang : float
            The polarization orientation of the beam/detector [deg]
            (default: 0.)
        name : str
            Optional callsign of this particular beam (default: None)
        pol : str
            The polarization callsign of the beam (A or B)
            (default: A)
        dead : bool
            True if the beam is dead (not functioning) (default: False)
        btype : str
            Type of detector spatial response model. Options:
                Gaussian : A symmetric Gaussian beam, definied by FWHM
                EG       : An elliptical Gaussian
                PO       : A physical optics beam
            (default: Gaussian)
        fwhm : float
            Detector beam FWHM in arcmin (used for Guassian beam)
            (default : 43)
        lmax : int
            Bandlimit beam. If None, use 1.4*2*pi/fwhm. (default : None)
        mmax : int
            Azimuthal band-limit beam. If None, use lmax (default : None)
        ghost : bool
            Whether the beam is a ghost or not (default : False)
        amplitude : scalar
            Total throughput of beam, i.e. integral of beam over the sphere.
            ( \int d\omega B(\omega) Y_00(\omega) \equiv amplitude ). This
            means that b00 = amplitude / sqrt(4 pi) (default : 1.)
        po_file : str, None
            Absolute or relative path to .npy file with blm array for the
            physical optics beam (default : None)
        eg_file : str, None
            Absolute or relative path to .npy file with blm array for the
            elliptical Gaussian beam (default : None)
        cross_pol : bool
            Whether to use the cross-polar response of the beam (requires
            blm .npy file to be of shape (3,), containing blm, blmm2 and blmp2
            (default : True)
        deconv_q : bool
            Multiply loaded blm's by sqrt(4 pi / (2 ell + 1)) before
            computing spin harmonic coefficients. Needed when using
            blm that are true SH coeff. (default : True)
        normalize : bool
            Normalize loaded up blm's such that 00 component is 1.
            Done after deconv_q operation if that option is set.
        polang_error : float
            Angle offset for polarization angle in deg. Scanning is
            done with `polang_truth` = `polang` + `polang_error`, binning
            can then be done with just `polang`.
        idx : int, None
            Identifier of beam. (default : None)
        symmetric : bool
            If set, beam is assumed azimuthally symmetric.
        '''

        self.az = az
        self.el = el
        self.polang = polang
        self.name = name
        self.pol = pol
        self.btype = btype
        self.dead = dead
        self.amplitude = amplitude
        self.po_file = po_file
        self.eg_file = eg_file
        self.cross_pol = cross_pol
        self.lmax = lmax
        self.mmax = mmax
        self.sensitive_freq = sensitive_freq
        self.fwhm = fwhm
        self.deconv_q = deconv_q
        self.normalize = normalize
        self.polang_error = polang_error
        self._idx = idx
        self.symmetric = symmetric
        self.hwp = hwp
        self.hwp_precomp_mueller=hwp_precomp_mueller

        self.__ghost = ghost
        # Ghosts are not allowed to have ghosts
        if not self.ghost:
            self.__ghosts = []
            self.ghost_count = 0

    @property
    def idx(self):
        return self._idx

    @property
    def ghost(self):
        return self.__ghost

    @property
    def ghosts(self):
        '''Return list of ghost beams.'''
        return self.__ghosts

    @property
    def ghost_count(self):
        return self.__ghost_count

    @ghost_count.setter
    def ghost_count(self, count):
        if not self.ghost:
            self.__ghost_count = count
        else:
            raise ValueError("ghost cannot have ghost_count")

    @property
    def ghost_idx(self):
        '''If two ghosts share ghost_idx, they share blm.'''
        return self.__ghost_idx

    @ghost_idx.setter
    def ghost_idx(self, val):
        if self.ghost:
            self.__ghost_idx = val
        else:
            raise ValueError("main beam cannot have ghost_idx")

    @property
    def dead(self):
        return self.__dead

    @dead.setter
    def dead(self, val):
        '''Make sure ghosts are also declared dead when main beam is.'''
        self.__dead = val
        try:
            for ghost in self.ghosts:
                ghost.dead = val
        except AttributeError:
            # Instance is ghost
            pass

    @property
    def lmax(self):
        return self.__lmax

    @lmax.setter
    def lmax(self, val):
        '''Make sure lmax is >= 0 and defaults to something sensible'''
        if val is None and self.fwhm is not None:
            # Going up to 1.4 naive Nyquist frequency set by beam scale
            self.__lmax = int(2 * np.pi / np.radians(self.fwhm/60.) * 1.4)
        else:
            self.__lmax = max(val, 0)

    @property
    def fwhm(self):
        return self.__fwhm

    @fwhm.setter
    def fwhm(self, val):
        '''
        Set beam fwhm. Returns absolute value of
        input and returns 1.4 * 2 * pi / lmax if
        fwhm is None.
        '''
        if val is None and self.lmax:
            val = (1.4 * 2. * np.pi) / float(self.lmax)
            self.__fwhm = np.degrees(val) * 60
        else:
            self.__fwhm = np.abs(val)

    @property
    def mmax(self):
        return self.__mmax

    @mmax.setter
    def mmax(self, mmax):
        '''Set mmax to lmax if not set.'''
        self.__mmax = min(i for i in [mmax, self.lmax] \
                              if i is not None)

    @property
    def blm(self):
        '''
        Get blm arrays by either creating them or
        loading them (depending on `btype` attr.

        Notes
        -----
        If blm attribute is already initialized and
        btype is changes, blm will not be updated,
        first delete blm attribute in that case.
        '''
        try:
            return self.__blm
        except AttributeError:

            if self.btype == 'Gaussian':
                self.gen_gaussian_blm()
                return self.__blm

            else:
                # NOTE, if blm's are direct map2alm resuls, use deconv_q.

                if self.btype == 'PO':
                    self.load_blm(self.po_file, deconv_q=self.deconv_q,
                                  normalize=self.normalize)
                    return self.__blm

                elif self.btype == 'EG':
                    self.load_blm(self.eg_file, deconv_q=self.deconv_q,
                                  normalize=self.normalize)
                    return self.__blm

                else:
                    raise ValueError("btype = {} not recognized".format(self.btype))

    @blm.setter
    def blm(self, val):
        self.__blm = val

    @blm.deleter
    def blm(self):
        del self.__blm

    @property
    def polang_truth(self):
        return self.polang + self.polang_error

    def __str__(self):

        return "name    : {} \nbtype   : {} \nalive   : {} \nFWHM"\
            "    : {} arcmin \naz      : {} deg \nel      : {} deg "\
            "\npolang  : {} deg\npo_file : {} \n".\
            format(self.name, self.btype,
                str(not self.dead), self.fwhm, self.az, self.el,
                self.polang_truth, self.po_file)

    def gen_gaussian_blm(self):
        '''
        Generate symmetric Gaussian beam coefficients
        (I and pol) using FWHM and lmax.

        Notes
        -----
        Harmonic coefficients are multiplied by factor
        sqrt(4 pi / (2 ell + 1)) and scaled by
        `amplitude` attribute (see `Beam.__init__()`).
        '''

        blm = tools.gauss_blm(self.fwhm, self.lmax, pol=False)
        if self.amplitude != 1:
            blm *= self.amplitude
        blm = tools.get_copol_blm(blm, c2_fwhm=self.fwhm)

        self.btype = 'Gaussian'
        self.blm = blm

    def load_blm(self, filename, **kwargs):
        '''
        Load file containing with blm array(s),
        and use array(s) to populate `blm` attribute.

        May update mmax attribute.

        Arguments
        ---------
        filename : str
            Absolute or relative path to .npy or .fits file

        Keyword arguments
        -----------------
        kwargs : {tools.get_copol_blm_opts}

        Notes
        -----
        Loaded blm are automatically scaled by given the `amplitude`
        attribute.

        blm file can be rank 1 or 2. If rank is 1: array is blm and
        blmm2 and blmp2 are created assuming only the co-polar response
        If rank is 2, shape has to be (3,), with blm, blmm2 and blmp2

        .npy files are assumed to be healpy alm arrays written using
        `numpy.save`. .fits files are assumed in l**2+l+m+1 order (i.e.
        written by healpy.write_alm. mmax may be smaller than lmax.
        '''

        pname, ext = os.path.splitext(filename)
        try:
            if not ext:
                # Assume .npy extension
                ext = '.npy'
            blm = np.load(os.path.join(pname+ext), allow_pickle=True)

        except IOError:
            if not ext:
                # Assume .fits file instead
                ext = '.fits'
            blm_file = os.path.join(pname+ext)

            hdulist = hp.fitsfunc.pf.open(blm_file)
            npol = len(hdulist)-1
            hdulist.close()

            blm_read, mmax = hp.read_alm(blm_file, hdu=1, return_mmax=True)

            # For now, expand blm to full size.
            lmax = hp.Alm.getlmax(blm_read.size, mmax=mmax)
            blm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
            blm[:blm_read.size] = blm_read

            if npol == 3:
                blmm2_read, mmaxm2 = hp.read_alm(blm_file, hdu=2, return_mmax=True)
                blmp2_read, mmaxp2 = hp.read_alm(blm_file, hdu=3, return_mmax=True)

                if not mmax == mmaxm2 == mmaxp2:
                    raise ValueError("mmax does not match between s=0,-2,2")

                # Expand.
                blmm2 = np.zeros_like(blm)
                blmp2 = np.zeros_like(blm)

                blmm2[:blmm2_read.size] = blmm2_read
                blmp2[:blmp2_read.size] = blmp2_read

                blm = (blm, blmm2, blmp2)

            # Update mmax if needed.
            if mmax is None:
                self.mmax = mmax
            else:
                if mmax < self.mmax:
                    self.mmax = mmax

        # If tuple turn to (3, ) or (1, ..) array.
        blm = np.atleast_2d(blm)

        if blm.shape[0] == 3 and self.cross_pol:
            cross_pol = True
        else:
            cross_pol = False
            blm = blm[0]

        if cross_pol:
            # Assume co- and cross-polar beams are provided
            # c2_fwhm has no meaning if cross-pol is known
            kwargs.pop('c2_fwhm', None)
            blm = tools.scale_blm(blm, **kwargs)

            if self.amplitude != 1:
                # Scale beam if needed
                blm *= self.amplitude

            self.blm = blm[0], blm[1], blm[2]

        else:
            # Assume co-polarized beam
            if self.amplitude != 1:
                # scale beam if needed
                blm *= self.amplitude

            # Create spin \pm 2 components
            self.blm = tools.get_copol_blm(blm, **kwargs)

    def create_ghost(self, tag='ghost', **kwargs):
        '''
        Append a ghost Beam object to the `ghosts` attribute.
        This method will raise an error when called from a
        ghost Beam object.

        Keyword Arguments
        -----------------
        tag : str
            Identifier string appended like <name>_<tag>
            where <name> is parent beam's name. If empty string,
            or None, just use parent Beam name. (default : ghost)
        kwargs : {beam_opts}

        Notes
        ----
        Valid Keyword arguments are those accepted by
        `Beam.__init__()` with the exception of `name`,
        which is ignored and `ghost`, which is always set.
        Unspecified kwargs are copied from parent beam.
        '''

        if self.ghost:
            raise RuntimeError('Ghost cannot have ghosts')

        parent_name = self.name
        kwargs.pop('name', None)
        if tag:
            if parent_name:
                name = parent_name + ('_' + tag)
            else:
                name = tag
        else:
            name = parent_name

        # mostly default to parent kwargs
        ghost_opts = dict(az=self.az,
                          el=self.el,
                          polang=self.polang,
                          name=name,
                          pol=self.pol,
                          btype=self.btype,
                          fwhm=self.fwhm,
                          dead=self.dead,
                          lmax=self.lmax,
                          mmax=self.mmax,
                          amplitude=self.amplitude,
                          po_file=self.po_file,
                          eg_file=self.eg_file,
                          cross_pol=self.cross_pol,
                          deconv_q=self.deconv_q,
                          normalize=self.normalize,
                          polang_error=self.polang_error)

        # Note, amplitude is applied after normalization
        # update options with specified kwargs
        ghost_opts.update(kwargs)
        ghost_opts.update(dict(ghost=True))
        ghost = Beam(**ghost_opts)

        # set ghost_idx
        ghost.ghost_idx = self.ghost_count
        self.ghost_count += 1

        self.ghosts.append(ghost)

    def reuse_blm(self, partner):
        '''
        Copy pointers to already initialized beam by
        another Beam instance. If both beams are
        ghosts, beam takes partner's `ghost_idx`.

        Arguments
        ---------
        partner : Beam object
        '''

        if not isinstance(partner, Beam):
            raise TypeError('partner must be Beam object')

        if partner.ghost and self.ghost:
            self.ghost_idx = partner.ghost_idx

        self.blm = partner.blm
        self.btype = partner.btype
        self.lmax = partner.lmax
        self.mmax = partner.mmax
        self.amplitude = partner.amplitude

    def delete_blm(self, del_ghosts_blm=True):
        '''
        Remove the `blm` attribute of the object. Does the same
        for ghosts, if specified.

        Keyword arguments
        -----------------
        del_ghost_blm : bool
            If True, also remove blm attributes of all ghosts
            (default : True)
        '''

        try:
            del(self.blm)
        except AttributeError:
            # no blm attribute to begin with
            pass

        if any(self.ghosts) and del_ghosts_blm:
            for ghost in self.ghosts:
                try:
                    del(ghost.blm)
                except AttributeError:
                    pass

    def get_offsets(self):
        '''
        Return (unrotated) detector offsets.

        Returns
        -------
        az : float
            Azimuth of offset in degrees.
        el : float
            Elevation of offset in degrees.
        polang : float
            Polarization angle in degrees (with
            polang_error included).

        Notes
        -----
        Detector offsets are defined
        as the sequence Rz(polang), Ry(el), Rx(az). Rz is defined
        as the rotation around the boresight by angle `polang`
        which is measured relative to the southern side of
        the local meridian in a clockwise manner when looking
        towards the sky (Rh rot.), (i.e. the `Healpix convention`).
        Followed by Ry and Rx, which are rotations in elevation
        and azimuth with respect to the local horizon and meridian.
        '''

        return self.az, self.el, self.polang_truth

    def _set_HWP_values(self, model_name=None, thicknesses=None, indices=None, losses=None, angles=None):
        if(model_name is None and ((thicknesses is None) or (indices is None) or (losses is None) or (angles is None))):
            raise ValueError('You must give either a model or parameters for a stack !')
        if (model_name !=None):
            self.hwp._choose_HWP_model(model_name=model_name)
        else:
            self.hwp._stack_builder(self, thicknesses=thicknesses, 
                indices=indices, losses=losses, angles=angles)

        self.hwp_precomp_mueller = self.hwp._compute4params(freq=self.sensitive_freq,
                alpha=np.radians(self.el))

    def _get_Mueller_top_row(self, xi, psi, theta):
        if (self.hwp_precomp_mueller is None):
            self.hwp_precomp_mueller = self.hwp._compute4params(freq=self.sensitive_freq,
                alpha=np.radians(self.el))
        return self.hwp._topRowMuellerMatrix(xi = xi, psi=psi, theta=theta, hwp_params = self.hwp_precomp_mueller)


