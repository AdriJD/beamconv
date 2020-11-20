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

    def stack_builder(self, thicknesses, indices, losses, angles):

        '''
        Creates a stack of materials, as defined in transfer_matrix.py
        Arguments:
        ------------
        thicknesses : (N,1) array of floats 
            thickness of each HWP layer in mm
        indices     : (N,2) array of floats 
            ordinary and extraordinary indices for each layer
        losses      : (N,2) array of floats 
            loss tangents in each layer.
        angles      : (N,1) array of floats 
            angle between (extraordinary) axis and stack axis for each layer, in radians
        '''

        if (thicknesses.size != angles.size or 2 * thicknesses.size != indices.size
            or 2*thicknesses.size != losses.size):
            raise ValueError('There is a mismatch in the sizes of the inputs for the HWP stack')

        # Make a list of materials, with a name that corresponds to their position in the stack
        material_stack=[]
        for i in range(thicknesses.size):

            if (indices[i,0]==indices[i,1] and losses[i,0]==losses[i,1]):
                isotro_str = 'isotropic'
            else:
                isotro_str = 'uniaxial'

            material_stack.append(tm.material(indices[i,0], indices[i,1],
                losses[i,0], losses[i,1], str(i), materialType=isotro_str))

        self.stack = tm.Stack( thicknesses*tm.mm, material_stack, angles)

    def choose_HWP_model(self, model_name):
        '''
        Set a particlar stack from a few predefined models
        Argument 
        ---------------
        model_name  : (string) 
            Name of one of the predefined HWP models
        '''

        spider_sapphire = tm.material(3.019, 3.336, 2.3e-4, 1.25e-4, 'Sapphire at 4K', materialType='uniaxial')
        #Spider coatings
        quartz = tm.material(1.951, 1.951, 1.2e-3, 1.2e-3, 'Quartz', materialType='isotropic')
        circlex = tm.material(1.935, 1.935, 1.2e-3, 1.2e-3, 'Circlex', materialType='isotropic')
        hdpe = tm.material(1.51, 1.51, 1e-3, 1e-3, 'HDPE', materialType='isotropic' )
        #Optimization results coatings
        art_ar1_mono = tm.material(3.439, 3.439, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        art_ar2_mono = tm.material(2.644, 2.644, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        art_ar3_mono = tm.material(1.524, 1.524, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        #New 3AR1BR & 3AR3BR & 3AR5BR AR coatings
        equal_ar1 = tm.material(2.855, 2.855, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        equal_ar2 = tm.material(1.979, 1.979, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        equal_ar3 = tm.material(1.268, 1.268, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        #From optimization
        a3b3_ar1 = tm.material(2.350, 2.350, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        a3b3_ar2 = tm.material(1.542, 1.542, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        a3b3_ar3 = tm.material(1.344, 1.344, 1.2e-3, 1.2e-3, 'fiducial AR', materialType='isotropic')
        a3b5_ar1 = tm.material(2.511, 2.511, 1.2e-3, 1.2e-3, 'composite AR', materialType='isotropic')
        a3b5_ar2 = tm.material(1.782, 1.782, 1.2e-3, 1.2e-3, 'composite AR', materialType='isotropic')
        a3b5_ar3 = tm.material(1.279, 1.279, 1.2e-3, 1.2e-3, 'composite AR', materialType='isotropic')

        if (model_name=='SPIDER_95'):

            thicknesses = [0.427*tm.mm, 4.930*tm.mm, 0.427*tm.mm]
            materials = [quartz, spider_sapphire, quartz]
            angles = [0.0, 0.0, 0.0]

        elif(model_name=='SPIDER_150'):

            thicknesses = [0.254*tm.mm, 0.006*tm.mm, 3.16*tm.mm, 0.006*tm.mm, 0.254*tm.mm]
            materials = [circlex, hdpe, spider_sapphire, hdpe, circlex]
            angles = [0.0, 0.0, 0.0, 0.0, 0.0]

        elif (model_name == '3AR1BRopti'):

            thicknesses = [0.358*tm.mm,1.849*tm.mm, 1.978*tm.mm, 3.6*tm.mm, 1.978*tm.mm, 1.849*tm.mm, 0.358*tm.mm]
            materials = [art_ar3_mono, art_ar2_mono ,art_ar1_mono, spider_sapphire, art_ar1_mono, art_ar2_mono, art_ar3_mono]
            angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        elif (model_name == '1BR'):#New 1AR3BR- same thicknesses for New 3AR3BR_new, 3AR5BR_new

            thicknesses = [0.5*tm.mm,0.31*tm.mm, 0.257*tm.mm, 3.75*tm.mm, 0.257*tm.mm, 0.31*tm.mm, 0.5*tm.mm]
            materials = [equal_ar3, equal_ar2, equal_ar1, spider_sapphire, equal_ar1, equal_ar2, equal_ar3]
            angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        elif (model_name == '3AR3BRopti'):#NEW: A AHWP with the best 3 layer of AR centered at 122.5GHz

            thicknesses = [0.338*tm.mm, 0.132*tm.mm, 0.240*tm.mm, 3.86*tm.mm, 3.86*tm.mm, 3.86*tm.mm,
                           0.240*tm.mm, 0.132*tm.mm, 0.338*tm.mm]
            materials = [a3b3_ar3, a3b3_ar2, a3b3_ar1, spider_sapphire, spider_sapphire, spider_sapphire,
                        a3b3_ar1, a3b3_ar2, a3b3_ar3]
            angles = np.array([0.,0. ,0.,0.,52.5,0.,0.,0., 0.])*np.pi/180.

        elif (model_name == '3BR'):#New 3AR3BR- same thicknesses for New 3AR1BR_new, 3AR5BR_new

            thicknesses = [0.5*tm.mm,0.31*tm.mm, 0.257*tm.mm, 3.75*tm.mm, 3.75*tm.mm, 3.75*tm.mm, 0.257*tm.mm, 0.31*tm.mm, 0.5*tm.mm]
            materials = [equal_ar3, equal_ar2, equal_ar1, spider_sapphire, spider_sapphire, spider_sapphire,
                         equal_ar1, equal_ar2, equal_ar3]
            angles = np.array([0.,0. ,0.,0.,54,0.,0.,0., 0.])*np.pi/180.

        elif (model_name=='3AR5BRopti'):#angles from Matsumura, AR from our optimization

            thicknesses = [0.356*tm.mm, 0.225*tm.mm, 0.188*tm.mm,
                            3.86*tm.mm,3.86*tm.mm,3.86*tm.mm,3.86*tm.mm,3.86*tm.mm,
                            0.188*tm.mm, 0.225*tm.mm, 0.356*tm.mm]
            materials = [a3b5_ar3, a3b5_ar2, a3b5_ar1, spider_sapphire,
                         spider_sapphire, spider_sapphire, spider_sapphire,
                         spider_sapphire, a3b5_ar1, a3b5_ar2, a3b5_ar3]
            angles = np.array([0.,0.,0., 0.,29.,94.5,29.,2., 0.,0.,0.])*np.pi/180.0


        elif (model_name == '3AR5BRstd'): #New 3AR5BR- same thicknesses for New 3AR1BR_new, 3AR3BR_new

            thicknesses = [0.5*tm.mm,0.31*tm.mm, 0.257*tm.mm, 3.75*tm.mm, 3.75*tm.mm, 3.75*tm.mm,
                           3.75*tm.mm, 3.75*tm.mm, 0.257*tm.mm, 0.31*tm.mm, 0.5*tm.mm]
            materials = [equal_ar3, equal_ar2, equal_ar1, spider_sapphire,
                         spider_sapphire, spider_sapphire, spider_sapphire,
                         spider_sapphire, equal_ar1, equal_ar2, equal_ar3]
            angles = np.array([0.,0.,0., 0.,26.5,94.8,28.1,-2.6 ,0.,0.,0.])*np.pi/180.0

        elif (model_name == '5BR'): #A 5BR layereed HWP with varphi = 0 over our two frequency bands

            thicknesses = [0.5*tm.mm,0.31*tm.mm, 0.257*tm.mm, 3.75*tm.mm, 3.75*tm.mm, 3.75*tm.mm,
                           3.75*tm.mm, 3.75*tm.mm, 0.257*tm.mm, 0.31*tm.mm, 0.5*tm.mm]
            materials = [equal_ar3, equal_ar2, equal_ar1, spider_sapphire,
                         spider_sapphire, spider_sapphire, spider_sapphire,
                         spider_sapphire, equal_ar1, equal_ar2, equal_ar3]
            angles = np.array([0.,0.,0.,22.9, -50,0,  50,  -22.9 ,0.,0.,0.])*np.pi/180.0
        else:
            raise ValueError('Unknown type of HWP entered')

        self.stack = tm.Stack( thicknesses, materials, angles)

    def compute_mueller(self, freq, vartheta):
        '''
        Compute the unrotated Mueller Matrix

        Arguments
        -------
        freq : float
            Frequency in GHz
        vartheta : float
            Incidence angle on HWP in radians
        '''
        return(tm.Mueller(self.stack, frequency=1.0e9*freq, incidenceAngle=vartheta,
            rotation=0., reflected=False))


class Beam(object):
    '''
    A class representing detector and beam properties.
    '''
    def __init__(self, az=0., el=0., polang=0., name=None,
                 pol='A', btype='Gaussian', fwhm=None, lmax=700, mmax=None, sensitive_freq = 150,
                 dead=False, ghost=False, amplitude=1., po_file=None,
                 eg_file=None, cross_pol=True, deconv_q=True,
                 normalize=True, polang_error=0., idx=None,
                 symmetric=False, hwp=HWP(), 
                 hwp_mueller=None):
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
            Detector beam FWHM in arcmin (used for Gaussian beam)
            (default : 43)
        sensitive_freq : float
            Detector beam frequency in GHz
        lmax : int
            Bandlimit beam. If None, use 1.4*2*pi/fwhm. (default : None)
        mmax : int
            Azimuthal band-limit beam. If None, use lmax (default : None)
        ghost : bool
            Whether the beam is a ghost or not (default : False)
        amplitude : scalar
            Total throughput of beam, i.e. integral of beam over the sphere.
            (int d omega B(omega) Y_00(omega) equiv amplitude ). This
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
        hwp : HWP class, Empty constructor
            An empty HWP with no characteristics, that are to be set afterwards
            by the setters
        hwp_mueller : (4,4) array, None
            Full unrotated Mueller matrix of the stack for a given incidence angle
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
        self.hwp_mueller = hwp_mueller

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
        or (4,), with blm, blmm2, blmp2, blmv

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

            if npol >= 3:
                blmm2_read, mmaxm2 = hp.read_alm(blm_file, hdu=2, return_mmax=True)
                blmp2_read, mmaxp2 = hp.read_alm(blm_file, hdu=3, return_mmax=True)

                if not mmax == mmaxm2 == mmaxp2:
                    raise ValueError("mmax does not match between s=0,-2,2")

                # Expand.
                blmm2 = np.zeros_like(blm)
                blmp2 = np.zeros_like(blm)

                blmm2[:blmm2_read.size] = blmm2_read
                blmp2[:blmp2_read.size] = blmp2_read

                if npol == 4:

                    blmv_read, mvmax = hp.read_alm(blm_file, hdu=4, return_mmax=True)
                    blmv = np.zeros_like(blm)
                    blmv[:blmv_read.size] = blmv_read
                    blm = (blm, blmm2, blmp2, blmv)

                else:    
                    blm = (blm, blmm2, blmp2)

            # Update mmax if needed.
            if mmax is None:
                self.mmax = mmax
            else:
                if mmax < self.mmax:
                    self.mmax = mmax

        # If tuple turn to (3, ) or (1, ..) array.
        blm = np.atleast_2d(blm)

        if blm.shape[0] == 3 or blm.shape[0] == 4 and self.cross_pol:
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

            if np.shape(blm)[0] == 4: 
                self.blm = blm[0], blm[1], blm[2], blm[3]
            else:    
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


    def set_hwp_mueller(self, model_name=None, thicknesses=None, indices=None, losses=None, angles=None):
        '''
        Set HWP mueller matrix for the beam given a stack
        Keyword arguments
        -----------------
        model_name  : string (None)
            A preset hwp model: see HWP.choose_HWP_model() for details
        thicknesses : (N,1) array of floats (None) 
            thickness of each HWP layer in mm
        indices     : (N,2) array of floats (None)
            ordinary and extraordinary indices for each layer
        losses      : (N,2) array of floats (None)
            loss tangents in each layer.
        angles      : (N,1) array of floats (None) 
            angle between (extraordinary) axis and stack axis for each layer, in radians
        '''

        if(model_name is None and any(elem is None for elem in 
            [thicknesses, indices, losses, angles])):
            raise ValueError('You must give either a model or parameters for a stack !')
        if (model_name !=None):
            self.hwp.choose_HWP_model(model_name=model_name)
        else:

            self.hwp.stack_builder(thicknesses=thicknesses,
                indices=indices, losses=losses, angles=angles)

        self.hwp_mueller = self.hwp.compute_mueller(freq=self.sensitive_freq,
                vartheta=np.radians(self.el))




