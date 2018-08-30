import os
import sys
import time
import copy
from warnings import warn, catch_warnings, simplefilter

import numpy as np
import qpoint as qp
import healpy as hp

import tools
from detector import Beam

class MPIBase(object):
    '''
    Parent class for MPI related stuff
    '''

    def __init__(self, mpi=True, comm=None, **kwargs):
        '''
        Check if MPI is working by checking common
        MPI environment variables and set MPI atrributes.

        Keyword arguments
        ---------
        mpi : bool
            If False, do not use MPI regardless of MPI env.
            otherwise, let code decide based on env. vars
            (default : True)
        comm : MPI.comm object, None
            External communicator. If left None, create
            communicator. (default : None)
        '''

        super(MPIBase, self).__init__(**kwargs)

        # Check whether MPI is working
        # add your own environment variable if needed
        # Open MPI environment variable
        ompi_size = os.getenv('OMPI_COMM_WORLD_SIZE')
        # intel and/or mpich environment variable
        pmi_size = os.getenv('PMI_SIZE')

        if not (ompi_size or pmi_size) or not mpi:
            self.mpi = False

        else:
            try:
                from mpi4py import MPI

                self.mpi = True
                self._mpi_double = MPI.DOUBLE
                self._mpi_sum = MPI.SUM
                if comm:
                    self._comm = comm
                else:
                    self._comm = MPI.COMM_WORLD

            except ImportError:
                warn("Failed to import mpi4py, continuing without MPI",
                     RuntimeWarning)

                self.mpi = False

    @property
    def mpi_rank(self):
        if self.mpi:
            return self._comm.Get_rank()
        else:
            return 0

    @property
    def mpi_size(self):
        if self.mpi:
            return self._comm.Get_size()
        else:
            return 1

    def barrier(self):
        '''
        MPI barrier that does nothing if not MPI
        '''
        if not self.mpi:
            return
        self._comm.Barrier()

    def scatter_list(self, list_tot, root=0):
        '''
        Scatter python list from `root` even if
        list is not evenly divisible among ranks.

        Arguments
        ---------
        list_tot : array-like
            List or array to be scattered (in 0-axis).
            Not-None on rank specified by `root`.

        Keyword arguments
        -----------------
        root : int
            Root rank (default : 0)
        '''

        if not self.mpi:
            return list_tot

        if self.mpi_rank == root:
            arr = np.asarray(list_tot)
            arrs = np.array_split(arr, self.mpi_size)

        else:
            arrs = None

        arrs = self._comm.scatter(arrs, root=root)

        return arrs.tolist()

    def broadcast(self, obj):
        '''
        Broadcast a python object that is non-None on root
        to all other ranks. Can be None on other ranks, but
        should exist in scope.

        Arguments
        ---------
        obj : object

        Returns
        -------
        bcast_obj : object
            Input obj, but now on all ranks
        '''

        if not self.mpi:
            return obj

        obj = self._comm.bcast(obj, root=0)
        return obj

    def broadcast_array(self, arr):
        '''
        Broadcast array from root process to all other ranks.

        Arguments
        ---------
        arr : array-like or None
            Array to be broadcasted. Not-None on root
            process, can be None on other ranks.

        Returns
        -------
        bcast_arr : array-like
            input array (arr) on all ranks.
        '''

        if not self.mpi:
            return arr

        # Broadcast meta info first
        if self.mpi_rank == 0:
            shape = arr.shape
            dtype = arr.dtype
        else:
            shape = None
            dtype = None
        shape, dtype = self._comm.bcast((shape, dtype), root=0)

        if self.mpi_rank == 0:
            bcast_arr = arr
        else:
            bcast_arr = np.empty(shape, dtype=dtype)

        self._comm.Bcast(bcast_arr, root=0)

        return bcast_arr

    def reduce_array(self, arr_loc):
        '''
        Sum arrays on all ranks elementwise into an
        array living in the root process.

        Arguments
        ---------
        arr_loc : array-like
            Local numpy array on each rank to be reduced.
            Need to be of same shape and dtype on each rank.

        Returns
        -------
        arr : array-like or None
            Reduced numpy array with same shape and dtype as
            arr_loc on root process, None for other ranks
            (arr_loc if not using MPI)
        '''

        if not self.mpi:
            return arr_loc

        if self.mpi_rank == 0:
            arr = np.empty_like(arr_loc)
        else:
            arr = None

        self._comm.Reduce(arr_loc, arr, op=self._mpi_sum, root=0)

        return arr

    def distribute_array(self, arr):
        '''
        If MPI is enabled, give every rank a proportionate
        view of the total array (or list).

        Arguments
        ---------
        arr : array-like
            Full-sized array present on every rank

        Returns
        -------
        arr_loc : array-like
            View of array unique to every rank.
        '''

        if self.mpi:

            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = np.divmod(len(arr), self.mpi_size)
            sub_size += quot

            if remainder:
                # give first ranks extra element
                sub_size[:int(remainder)] += 1

            start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            end = start + sub_size[self.mpi_rank]

            arr_loc = arr[start:end]

        else:
            arr_loc = arr

        return arr_loc

class Instrument(MPIBase):
    '''
    Initialize a telescope and specify its properties.
    '''

    def __init__(self, location='spole', lat=None, lon=None,
                 **kwargs):
        '''
        Set location of telescope and initialize empty focal 
        plane.

        Keyword arguments
        -----------------
        location : str
            Predefined locations. Current options:
                spole    : (lat=-89.9, lon=169.15)
                atacama  : (lat=-22.96, lon=-67.79)
                space    : (Fixed lat, time-variable lon)
            (default : spole)
        lon : float, None
            Longitude in degrees. (default : None)
        lat : float, None
            Latitude in degrees (default : None)
        kwargs : {mpi_opts}

        Notes
        -----
        The Instrument holds an `beams` attribute that 
        consists of a nested list reprenting pairs of
        beamconv.Beam objects (detectors). `beams` is
        equal on all MPI ranks.
        '''

        if location == 'spole':
            self.lat = -89.9
            self.lon = 169.15

        elif location == 'atacama':
            self.lat = -22.96
            self.lon = -67.79

        elif location == 'space':
            self.lat = None
            self.lon = None

        if lat:
            self.lat = lat
        if lon:
            self.lon = lon

        if (location != 'space') and (not self.lat or not self.lon):
            raise ValueError('Specify location of telescope')

        self.beams = []
        self.ndet = 0

        super(Instrument, self).__init__(**kwargs)
        
    def add_to_focal_plane(self, beams, combine=True):
        '''
        Add beam(s) or beam pair(s) to total list of beam pairs.

        Arguments
        ---------
        beams : (list of) Beam instances
            If nested list: assumed to be list of beam pairs.

        Keyword arguments
        -----------------
        combine : bool
            If some beams already exist, combine these new
            beams with them.
            (default : True)
        '''

        # Check whether single beam, list of beams or list of pairs.
        try:
            beams[0]
            isseq = True
        except TypeError:
            isseq = False

        if isseq:                    
            try:
                beams[0][0]
                isnestseq = True
            except TypeError:
                isnestseq = False
        else:
            isnestseq = False

        # If not pairs, we add None as partner.
        ndet2add = 0
        if isseq is False:
            beams2add = [[beams, None]]
            ndet2add += 1

        if isseq:
            beams2add = []
            for pair in beams:
                if isnestseq is False:
                    pair = [pair, None]
                    ndet2add += 1
                else:
                    ndet2add += 2

                beams2add.append(pair)

        if not hasattr(self, 'beams') or not combine:
            self.beams = beams2add
        else:
            self.beams += beams2add

        if not hasattr(self, 'ndet') or not combine:
            self.ndet = ndet2add
        else:
            self.ndet += ndet2add

        return
            
    def create_focal_plane(self, nrow=1, ncol=1, fov=10.,
                           no_pairs=False, combine=True,
                           scatter=False, **kwargs):
        '''
        Create Beam objects for orthogonally polarized
        detector pairs with pointing offsets lying on a
        rectangular az-el grid on the sky.

        Keyword arguments
        ---------
        nrow : int, optional
            Number of detectors per row (default: 1)
        ncol : int, optional
            Number of detectors per column (default: 1)
        fov : float, optional
            Angular size of side of square focal plane on
            sky in degrees (default: 10.)
        no_pairs : bool
            Do not create detector pairs, i.e. only create
            A detector and let B detector be dead
            (default : False)
        combine : bool
            If some beams already exist, combine these new
            beams with them
            (default : True)
        scatter : bool
            Scatter created pairs over ranks (default : False)
        kwargs : {beam_opts}

        Notes
        -----
        "B"-detector's polarization angle is A's angle + 90.

        If a `beams` attribute already exists, this method
        will append the beams to that list.

        Any keywords accepted by the `Beam` class will be
        assumed to hold for all beams created, with the
        exception of (`az`, `el`, `pol`, `name`, `ghost`),
        which are ignored. `polang` is used for A-detectors,
        B-detectors get polang + 90.
        '''

        # Ignore these kwargs and warn user.
        for key in ['az', 'el', 'pol', 'name', 'ghost']:
            arg = kwargs.pop(key, None)
            if arg:
                warn('{}={} option to `Beam.__init__()` is ignored'
                     .format(key, arg))

        # Some kwargs need to be dealt with seperately.
        polang = kwargs.pop('polang', 0.)
        dead = kwargs.pop('dead', False)

        if not hasattr(self, 'ndet') or not combine:
            self.ndet = 2 * nrow * ncol # A and B detectors
        else:
            self.ndet += 2 * nrow * ncol # A and B detectors

        azs = np.linspace(-fov/2., fov/2., ncol)
        els = np.linspace(-fov/2., fov/2., nrow)

        beams = []

        for az_idx in xrange(azs.size):
            for el_idx in xrange(els.size):

                det_str = 'r{:03d}c{:03d}'.format(el_idx, az_idx)

                beam_a = Beam(az=azs[az_idx], el=els[el_idx],
                              name=det_str+'A', polang=polang,
                              dead=dead, pol='A', **kwargs)

                beam_b = Beam(az=azs[az_idx], el=els[el_idx],
                              name=det_str+'B', polang=polang+90.,
                              dead=dead or no_pairs, pol='B',
                              **kwargs)

                beams.append([beam_a, beam_b])

        # If MPI, distribute beams over ranks
        if scatter:
            beams = self.distribute_array(beams)

        # Check for existing beams
        if not hasattr(self, 'beams') or not combine:
            self.beams = beams
        else:
            self.beams += beams

    def load_focal_plane(self, bdir, tag=None, no_pairs=False,
                         combine=True, scatter=False, polang_A=0.,
                         polang_B=90., print_list=False, **kwargs):
        '''
        Create focal plane by loading up a collection
        of beam properties stored in pickle files.

        Arguments
        ---------
        bdir : str
            The absolute or relative path to the directory
            with .pkl files containing (a list of two)
            <detector.Beam> options in a dictionary.

        Keyword arguments
        -----------------
        tag : str, None
            If set to string, only load files that contain <tag>
            (default : None)
        no_pairs : bool
            Do not create detector pairs, i.e. only create
            A detector and let B detector be dead
            (default : False)
        combine : bool
            If some beams already exist, combine these new
            beams with them
            (default : True)
        scatter : bool
            Scatter loaded pairs over ranks (default : False)
        polang_A : float
            Polarization angle of A detector in pair [deg]. 
            Added to existing or provided polang. (default: 0.)
        polang_B : float
            Polarization angle of b detector in pair [deg].
            Added to existing or provided polang. (default: 90.)
        print_list : bool
            Print list of beam files loaded up (for debugging)
            (default : False)
        kwargs : {beam_opts}

        Notes
        -----
        Raises a RuntimeError if no files are found.

        If loaded files contain a single dictionary, those
        options are assumed to hold for the A detector.
        It is assumed that the B-detectors share the
        A-detectors' beams (up to a shift in polang
        (see A/B-polang). Other properties of
        B-detectors are shared with A.

        Appends "A" or "B" to beam names if provided,
        depending on polarization of detector (only in
        single dictionary case).

        Any keywords accepted by the `Beam` class will be
        assumed to hold for all beams created. with the
        exception of (`pol`, `ghost`), which are
        ignored. 
        '''

        import glob
        import pickle

        # do all I/O on root
        if self.mpi_rank == 0:

            beams = []

            # Ignore these kwargs and warn user
            for key in ['pol', 'ghost']:
                arg = kwargs.pop(key, None)
                if arg:
                    warn('{}={} option to `Beam.__init__()` is ignored'
                         .format(key, arg))

            opj = os.path.join
            tag = '' if tag is None else tag

            file_list = glob.glob(opj(bdir, '*'+tag+'*.pkl'))

            if not file_list:
                raise RuntimeError(
                    'No files matching <*{}*.pkl> found in {}'.format(
                                                             tag, bdir))
            file_list.sort()

            if print_list:
                print('Load focal plane beam file_list:')
                for ii, fname in enumerate(file_list):
                    print('{}/{}: {}'.format(
                        ii+1, len(file_list), os.path.split(fname)[-1]))

                print('tag = {}'.format(tag))

            for bfile in file_list:

                pkl_file = open(bfile, 'rb')
                beam_opts = pickle.load(pkl_file)
                pkl_file.close()

                if isinstance(beam_opts, dict):                    
                    # Single dict of opts: assume A, create A and B.

                    beam_opts_a = beam_opts
                    beam_opts_b = copy.deepcopy(beam_opts)

                    if beam_opts.get('name'):
                        beam_opts_a['name'] += 'A'
                        beam_opts_b['name'] += 'B'

                elif isinstance(beam_opts, (list, tuple, np.ndarray)):
                    # Assume list of dicts for A and B.

                    if len(beam_opts) != 2:
                        raise ValueError('Need two elements: A and B')

                    beam_opts_a = beam_opts[0]
                    beam_opts_b = beam_opts[1]

                # Overrule options with given kwargs
                beam_opts_a.update(kwargs)
                beam_opts_b.update(kwargs)

                # Add polang A and B (perhaps on top of provided or esisting
                # polang).
                polang_Ai = beam_opts_a.setdefault('polang', 0)
                polang_Bi = beam_opts_b.setdefault('polang', 0)
                beam_opts_a['polang'] = polang_Ai + polang_A
                beam_opts_b['polang'] = polang_Bi + polang_B

                if no_pairs:
                    beam_opts_b['dead'] = True

                beam_opts_a.pop('pol', None)
                beam_opts_b.pop('pol', None)

                beam_a = Beam(pol='A', **beam_opts_a)
                beam_b = Beam(pol='B', **beam_opts_b)

                beams.append([beam_a, beam_b])

            ndet = len(beams) * 2
        else:
            beams = None
            ndet = None

        if scatter:
            # If MPI scatter to ranks
            beams = self.scatter_list(beams, root=0)
        else:
            # Broadcast otherwise because root did all I/O
            beams = self.broadcast(beams)
            
        # All ranks need to know total number of detectors
        ndet = self.broadcast(ndet)

        # Check for existing beams
        if not hasattr(self, 'beams') or not combine:
            self.beams = beams
        else:
            self.beams += beams

        if not hasattr(self, 'ndet') or not combine:
            self.ndet = ndet
        else:
            self.ndet += ndet

    def create_reflected_ghosts(self, beams=None, ghost_tag='refl_ghost',
                                rand_stdev=0., **kwargs):
        '''
        Create reflected ghosts based on detector
        offset (azimuth and elevation are multiplied by -1).
        Polarization angles stay the same.
        Ghosts are appended to `ghosts` attribute of beams.

        Keyword arguments
        -----------------
        beams : Beam object, array-like
            Single Beam object or array-like of Beam
            objects. If None, use beams attribute.
            (default : None)
        ghost_tag : str
            Tag to append to parents beam name, see
            `Beam.create_ghost()` (default : refl_ghost)
        rand_stdev : float
            Standard deviation of Gaussian random variable
            added to each ghost (default : 0.)
        kwargs : {create_ghost_opts, beam_opts}

        Notes
        -----
        Any keywords mentioned above accepted by the
        `Beam.create_ghost` method will be set for
        all created ghosts. E.g. set ghost level with
        the `amplitude` keyword (see `Beam.__init__()`)

        `az` and `el` kwargs are ignored.
        '''

        if not beams:
            beams = self.beams

        # tag overrules ghost_tag
        kwargs.setdefault('tag', ghost_tag)

        beams = np.atleast_2d(beams) #2D: we have pairs
        for pair in beams:
            for beam in pair:
                if not beam:
                    continue
                # Note, in python integers are immutable
                # so ghost offset is not updated when
                # beam offset is updated.
                refl_ghost_opts = dict(az=-beam.az,
                                       el=-beam.el)
                kwargs.update(refl_ghost_opts)

                if rand_stdev:
                    # add perturbation to ghost amplitude
                    amplitude = kwargs.get('amplitude', 1.)
                    amplitude += np.random.normal(scale=rand_stdev)
                    kwargs.update(dict(amplitude=amplitude))

                beam.create_ghost(**kwargs)

    def kill_channels(self, killfrac=0.2, pairs=False, rnd_state=None):
        '''
        Randomly identifies detectors in the beams list
        and sets their 'dead' attribute to True.

        Keyword arguments
        ---------
        killfrac : 0 < float < 1  (default: 0.2)
            The fraction of detectors to kill
        pairs : bool
            If True, kill pairs of detectors
            (default : False)
        rnd_state : numpy.random.RandomState
            Numpy random state instance. If None, use
            global instance. (default : None)
        '''

        if pairs:
            ndet = self.ndet / 2
        else:
            ndet = self.ndet

        # Calculate the kill indices on root and broadcast
        # to ensure all ranks share dead detectors.
        if self.mpi_rank == 0:
            if rnd_state:
                kill_indices = rnd_state.choice(ndet, int(ndet*killfrac),
                                            replace=False)
            else:
                kill_indices = np.random.choice(ndet, int(ndet*killfrac),
                                                replace=False)
        else:
            kill_indices = None

        kill_indices = self.broadcast(kill_indices)

        for kidx in kill_indices:
            if pairs:
                self.beams[kidx][0].dead = True
                self.beams[kidx][1].dead = True
            else:
                # If even kill A, else kill B.
                quot, rem = divmod(kidx, 2)
                self.beams[quot][rem].dead = True

    def set_global_prop(self, prop, incl_ghosts=True):
        '''
        Set a property for all beams on the focal plane.

        Arguments
        ---------
        prop : dict
            Dict with attribute(s) and values for beams

        Keyword arguments
        -----------------
        incl_ghosts : bool
            If set, also update attributes of ghosts.
            (default : True)

        Examples
        --------
        >>> S = Instrument()
        >>> S.create_focal_plane(nrow=10, ncol=10)
        >>> set_global_prop(dict(btype='PO'))        

        Notes
        -----
        Ghosts share random deviation with main beam.
        '''
        
        beams = np.atleast_2d(self.beams) #2D: we have pairs
        
        for pair in beams:            
            for beam in pair:

                if not beam:                
                    continue
                
                for key in prop:
                    val = prop[key]
                    setattr(beam, key, val)

                if incl_ghosts:
                    for ghost in beam.ghosts:
                        for key in prop:
                            val = prop[key]
                            setattr(ghost, key, val)

    def add_to_prop(self, prop, incl_ghosts=True,
                    rand_stdev=0., per_pair=False,
                    no_B=False, no_A=False, rnd_state=None):
        '''
        Add a value to a property for all beams on the focal plane.

        Arguments
        ---------
        prop : dict
            Dict with attribute and value for beams

        Keyword arguments
        -----------------
        incl_ghosts : bool
            If set, also update attributes of ghosts.
            (default : True)
        rand_stdev : float
            Standard deviation of Gaussian random variable
            added to each beam's property (default : 0.)
        per_pair : bool
            If set, add same random number to both partners 
            in pair. (default : False)
        no_B : bool
            Do not add value to B beams.
        no_A
            Do not add value to A beams.  
        rnd_state : numpy.random.RandomState
            Numpy random state instance. If None, use
            global instance. (default : None)
          
        Examples
        --------
        >>> S = Instrument()
        >>> S.create_focal_plane(nrow=10, ncol=10)
        >>> add_to_prop(dict(polang=0), rand_stdev=1)

        Notes
        -----
        Ghosts share random deviation with main beam.
        '''

        beams = np.atleast_2d(self.beams) #2D: we have pairs

        if len(prop) != 1:
            raise ValueError("Only update one property at a time.")
        
        for pair in beams:
            
            if rand_stdev != 0 and per_pair:
                if rnd_state:
                    rndvar = rnd_state.normal(scale=rand_stdev)
                else:
                    rndvar = np.random.normal(scale=rand_stdev)

            for bidx, beam in enumerate(pair):

                if no_B and bidx == 1:
                    continue

                if no_A and bidx == 0:
                    continue

                if rand_stdev != 0 and not per_pair:
                    if rnd_state:
                        rndvar = rnd_state.normal(scale=rand_stdev)
                    else:
                        rndvar = np.random.normal(scale=rand_stdev)

                if not beam:                
                    continue
                
                for key in prop: # Loops over single entry.
                    val = prop[key]
                    val = val + rndvar if rand_stdev != 0 else val
                    val0 = getattr(beam, key, 0) # Add to existing val.
                    setattr(beam, key, val0 + val)

                if incl_ghosts:
                    for ghost in beam.ghosts:
                        for key in prop:
                            val = prop[key]
                            val = val + rndvar if rand_stdev != 0 else val
                            val0 = getattr(ghost, key, 0)
                            setattr(ghost, key, val)
                            
    def set_btypes(self, btype='Gaussian'):
        '''
        Set btype for all main beams

        Keyword arguments
        -----------------
        btype : str
            Type of detector spatial response model. Can be one of three
            "Gaussian", "EG", "PO". (default : "Gaussian")
        '''

        beams = np.atleast_2d(self.beams) #2D: we have pairs
        for pair in beams:
            for beam in pair:
                if not beam:
                    continue

                beam.btype = btype

class ScanStrategy(Instrument, qp.QMap):
    '''
    Given an instrument, create a scanning pattern and
    scan the sky.
    '''

    _qp_version = (1, 10, 0)

    def __init__(self, duration=0, external_pointing=False,
                 ctime0=None, sample_rate=30, **kwargs):
        '''
        Initialize scan parameters

        Keyword arguments
        -----------------
        duration : float
            Mission duration in seconds (default : 0)
        external_pointing : bool
            If set, `constant_el_scan` loads up boresight pointing
            and time timestreams instead of calculating them.
            (default : False)
        ctime0 : float
            Start time in unix time. If None, use
            current time. Ignored if `external_pointing` is set
            (default : None)
        sample_rate : float
             Sample rate in Hz. If `external_pointing` is set,
             make sure this matches the sample rate of the
             external pointing. (default : 30)
        kwargs : {mpi_opts, instr_opts, qmap_opts}
        '''

        self.__fsamp = float(sample_rate)

        self.ctime0 = ctime0
        self.__mlen = duration
        self.__nsamp = int(self.mlen * self.fsamp)

        self.ext_point = external_pointing

        self.rot_dict = {}
        self.hwp_dict = {}
        self.step_dict = {}
        self.set_instr_rot()
        self.set_hwp_mod()

      # Checking qpoint version
        if qp.version() < self._qp_version:
            raise RuntimeError(
                 'qpoint version {} required, found version {}'.format(
                     self._qp_version, qp.version()))

        # Set some useful qpoint/qmap options as default
        qmap_opts = dict(pol=True,
                         fast_math=False,
                         mean_aber=True,
                         accuracy='low',
                         fast_pix=True)

        for key in qmap_opts:
            kwargs.setdefault(key, qmap_opts[key])

        super(ScanStrategy, self).__init__(**kwargs)

    def __del__(self):
        '''
        Call QPoint destructor explicitely to make sure
        the c code frees up memory before exiting.
        '''
        self.__del__

    @property
    def ctime0(self):
        return self.__ctime0

    @ctime0.setter
    def ctime0(self, val):
        if val:
            self.__ctime0 = val
        else:
            self.__ctime0 = time.time()

    @property
    def fsamp(self):
        return self.__fsamp

    @property
    def nsamp(self):
        return self.__nsamp

    @property
    def mlen(self):
        return self.__mlen

    def set_instr_rot(self, period=None, start_ang=0.,
                      angles=None):
        '''
        Set options that allow instrument to periodically
        rotate around the boresight.

        Keyword arguments
        ---------
        period : float
            Rotation period in seconds. If left None,
            keep instrument unrotated.
        start_ang : float, optional
            Starting angle of the instrument in deg.
            Default = 0 deg.
        angles : array-like, optional
            Set of rotation angles. If left None, cycle
            through 45 degree steps. If set, ignores
            start_ang
        '''

        self.rot_dict['period'] = period
        if period:
            self.rot_dict['rot_chunk_size'] = int(period * self.fsamp)
        self.rot_dict['angle'] = start_ang
        self.rot_dict['start_ang'] = start_ang
        self.rot_dict['remainder'] = 0

        if angles is None:
            angles = np.arange(start_ang, 360+start_ang, 45)
            np.mod(angles, 360, out=angles)

        self.rot_dict['angles'] = angles

        # init rotation generators
        self.rot_angle_gen = tools.angle_gen(angles)

    def reset_instr_rot(self):
        '''
        "Reset" the instrument rotation generator
        '''
        self.rot_angle_gen = tools.angle_gen(
            self.rot_dict['angles'])
        self.rot_dict['angle'] = self.rot_dict['start_ang']
        self.rot_dict['remainder'] = 0

    def rotate_instr(self):
        '''
        Advance boresight rotation options by one
        rotation
        '''
        if self.rot_dict['period']:
            self.rot_dict['angle'] = self.rot_angle_gen.next()

    def set_hwp_mod(self, mode=None,
                    freq=None, start_ang=0.,
                    angles=None):
        '''
        Set options for modulating the polarized sky signal
        using a (stepped or continuously rotating) half-wave
        plate.

        Keyword arguments
        ---------
        mode : str, None
            Either "stepped" or "continuous". If None,
            or False, do not rotate hwp (default : None)
        freq : float, None
            Rotation or step frequency in Hz
        start_ang : float
            Starting angle for the HWP in deg
        angles : array-like, None
            Rotation angles for stepped HWP. If not set,
            use 22.5 degree steps. If set, ignores
            start_ang.
        '''

        self.hwp_dict['mode'] = mode
        self.hwp_dict['freq'] = freq
        if mode == 'stepped':
            self.hwp_dict['step_size'] = int(self.fsamp / float(freq))
        self.hwp_dict['angle'] = start_ang
        self.hwp_dict['start_ang'] = start_ang
        self.hwp_dict['remainder'] = 0 # num. samp. from last step


        if angles is None:
            angles = np.arange(start_ang, 360+start_ang, 22.5)
            np.mod(angles, 360, out=angles)

        self.hwp_dict['angles'] = angles

        # init hwp ang generator
        self.hwp_angle_gen = tools.angle_gen(angles)

    def reset_hwp_mod(self):
        '''
        "Reset" the hwp modulation generator
        '''
        self.hwp_angle_gen = tools.angle_gen(
            self.hwp_dict['angles'])
        self.hwp_dict['angle'] = self.hwp_dict['start_ang']
        self.hwp_dict['remainder'] = 0

    def set_el_steps(self, period, steps=None):
        '''
        Set options for stepping the boresight up
        or down in elevation.

        Arguments
        ---------
        period : scalar
            The period between steps in seconds

        Keyword arguments
        ---------
        steps : array-like, None
            Steps in elevation that are cycled through (in deg.).
            If not set, use 5, 1 degree steps up and down.
            (default : None)
        '''

        self.step_dict['period'] = period
        self.step_dict['remainder'] = 0

        if steps is None:
            steps = np.array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0])
            np.mod(steps, 360, out=steps)

        self.step_dict['steps'] = steps
        self.step_dict['angle'] = steps[0]
        self.step_dict['step_size'] = int(self.fsamp * period)

        # init el step generator
        self.el_step_gen = tools.angle_gen(steps)

    def reset_el_steps(self):
        '''
        "Reset" the el step generator
        '''

        if not 'steps' in self.step_dict:
            return

        self.el_step_gen = tools.angle_gen(
            self.step_dict['steps'])
        self.step_dict['step'] = self.step_dict['steps'][0]
        self.step_dict['remainder'] = 0

    def partition_mission(self, chunksize=None):
        '''
        Divide up the mission in equal-sized chunks
        of nsample / chunksize (final chunk can be
        smaller).

        Keyword arguments
        ---------
        chunksize : int
            Chunk size in samples. If left None, use
            full mission length (default : None)

        Returns
        -------
        chunks : list of dicts
            Dictionary with start, end indices and index of each
            chunk.
        '''

        nsamp = self.nsamp

        if not chunksize or chunksize >= nsamp:
            chunksize = int(nsamp)

        chunksize = int(chunksize)
        nchunks = int(np.ceil(nsamp / float(chunksize)))
        chunks = []
        start = 0

        for cidx, chunk in enumerate(xrange(nchunks)):
            end = start + chunksize
            end = nsamp if end >= (nsamp) else end
            chunks.append(dict(start=start, end=end, cidx=cidx))
            start += chunksize

        self.chunks = chunks
        return chunks

    def subpart_chunk(self, chunk):
        '''
        Split a chunk up into smaller chunks based on the
        options set for boresight rotation.

        Arguments
        ---------
        chunk : dict
            Dictionary containing start, end indices (and optionally
            cidx) keys

        Returns
        -------
        subchunks : list of dicts
            Subchunks for each rotation of the boresight. If input
            chunk had a `cidx` key, every subchunk inherits this value.
        '''

        period = self.rot_dict['period']

        if not period:
        # No instrument rotation, so no need for subchunks
            return [chunk]

        rot_chunk_size = self.rot_dict['rot_chunk_size']
        chunk_size = chunk['end'] - chunk['start']

        subchunks = []
        start = chunk['start']

        nchunks = (chunk_size - self.rot_dict['remainder'])
        nchunks /= float(rot_chunk_size)
        nchunks = int(np.ceil(nchunks))
        nchunks = 1 if nchunks < 1 else nchunks

        if nchunks == 1:
            # rot period is larger or equal to chunksize
            if self.rot_dict['remainder'] >= chunk_size:

                subchunks.append(dict(start=start, end=chunk['end']))
                self.rot_dict['remainder'] -= chunk_size

            else:
                # one subchunk that is just the remainder if there is one
                end = self.rot_dict['remainder'] + start

                if self.rot_dict['remainder']:

                    # in this rotation chunk, no rotation should be made
                    subchunks.append(dict(start=start, end=end, norot=True))

                # another subchunk that is the rest of the chunk
                subchunks.append(dict(start=end, end=chunk['end']))

                self.rot_dict['remainder'] = rot_chunk_size - (chunk['end'] - end)

        elif nchunks > 1:
            # you can fit at most nstep - 1 full steps in chunk
            # remainder is at most stepsize
            if self.rot_dict['remainder']:

                end = self.rot_dict['remainder'] + start

                # again, no rotation should be done
                subchunks.append(dict(start=start, end=end, norot=True))

                start = end

            # loop over full-sized rotation chunks
            for step in xrange(nchunks-1):

                end = start + rot_chunk_size
                subchunks.append(dict(start=start, end=end))
                start = end

            # fill last part and determine remainder
            subchunks.append(dict(start=start, end=chunk['end']))
            self.rot_dict['remainder'] = rot_chunk_size - (chunk['end'] - start)

        # subchunks get same cidx as parent chunk
        if 'cidx' in chunk:
            cidx = chunk['cidx']
            for subchunk in subchunks:
                subchunk['cidx'] = cidx

        return subchunks

    def allocate_maps(self, nside=256):
        '''
        Allocate space in memory for healpy map-related numpy arrays

        Keyword arguments
        -----------------
        nside : int
            Nside of output (default : 256)
        '''

        self.vec = np.zeros((3, 12*nside**2), dtype=float)
        self.proj = np.zeros((6, 12*nside**2), dtype=float)
        self.nside_out = nside

    def scan_instrument(self, verbose=True, mapmaking=True,
        **kwargs):
        '''
        Cycles through chunks, scans and calculates
        detector tods for all detectors serially.
        Optionally: also bin tods into maps.

        Keyword arguments
        ---------
        verbose : bool [default True]
            Prints status reports
        mapmaking : bool, optional
            If True, bin tods into vec and proj.
        kwargs : {ces_opts}
            Extra kwargs are assumed input to
            `constant_el_scan()`
        '''

        if verbose:
            print('  Scanning with {:d} detectors'.format(
                self.ndet))

        for cidx, chunk in enumerate(self.chunks):

            if verbose:
                print('  Working on chunk {:03}: samples {:d}-{:d}'.format(cidx,
                    chunk['start'], chunk['end']))

            # Make the boresight move
            ces_opts = kwargs.copy()
            ces_opts.update(chunk)
            self.constant_el_scan(**ces_opts)

            # if required, loop over boresight rotations
            for subchunk in self.subpart_chunk(chunk):

                # rotate instrument if needed
                if self.rot_dict['period']:
                    self.rot_dict['angle'] = self.rot_angle_gen.next()

                # Cycling through detectors and scanning
                for chnidx in xrange(self.ndet):

                    az_off = self.azs[chnidx]
                    el_off = self.els[chnidx]
                    polang = self.polangs[chnidx]

                    self.scan(az_off=az_off, el_off=el_off,
                              polang=polang, **subchunk)

                    if mapmaking:
                        self.bin_tod(add_to_global=False)

                        # Adding to global maps
                        self.vec += self.depo['vec']
                        self.proj += self.depo['proj']

    def init_detpair(self, alm, beam_a, beam_b=None, **kwargs):
        '''
        Initialize the internal structure (the spinmaps)
        for a detector pair and all its ghosts.

        Arguments
        ---------
        alm : tuple
            Tuple containing (almI, almE, almB) as
            Healpix-formatted complex numpy arrays
        beam_a : <detector.Beam> object
            The A detector.

        Keyword arguments
        -----------------
        beam_b : <detector.Beam> object
            The B detector. (default : None)
        kwargs : {spinmaps_opts}
            Extra kwargs are assumed input to
            `init_spinmaps()`

        Notes
        -----
        Both detectors are assumed to share the same beam
        (up to a rotation). Therefore, only the A main beam
        needs to specified.
        '''

        if beam_b is not None:
            b_exists = True
        else:
            b_exists = False
        
        # We give the ghosts identical Gaussian beams if they
        # have no blm.
        if beam_a.ghosts:

            for gidx in xrange(beam_a.ghost_count):
                ghost_a = beam_a.ghosts[gidx]

                if gidx == 0:
                    if not hasattr(ghost_a, 'blm'):
                        ghost_a.gen_gaussian_blm()

                else:
                    if not hasattr(ghost_a, 'blm'):
                        ghost_a.reuse_blm(beam_a.ghosts[0])
        if b_exists:
            if beam_b.ghosts:

                for gidx in xrange(beam_b.ghost_count):
                    ghost_b = beam_b.ghosts[gidx]

                    if gidx == 0:
                        if not hasattr(ghost_b, 'blm'):
                            ghost_b.reuse_blm(beam_a.ghosts[0])
                    else:
                        if not hasattr(ghost_b, 'blm'):

                            beam_b.ghosts[gidx].reuse_blm(
                                beam_a.ghosts[0])


        self.init_spinmaps(alm, beam_obj=beam_a, **kwargs)

        # free blm attributes
        beam_a.delete_blm(del_ghosts_blm=True)
        if b_exists:
            beam_b.delete_blm(del_ghosts_blm=True)

    def scan_instrument_mpi(self, alm, verbose=1, binning=True,
            create_memmap=False, scatter=True, reuse_spinmaps=False,
            interp=False, **kwargs):
        '''
        Loop over beam pairs, calculates boresight pointing
        in parallel, rotates or modulates instrument if
        needed, calculates beam-convolved tods, and,
        optionally, bins tods.

        Arguments
        ---------
        alm : tuple
            Tuple containing (almI, almE, almB) as
            Healpix-formatted complex numpy arrays

        Keyword arguments
        ---------
        verbose : int
            Prints status reports (0 : nothing, 1: some,
            2: all) (defaul: 1)
        binning : bool, optional
            If True, bin tods into `vec` and `proj`
            attributes
        create_memmap : bool
            If True, store boresight quaternion (q_bore)
            in memory-mapped file on disk and read in
            q_bore from file on subsequent passes. If
            False, recalculate q_bore for each detector
            pair. (default : False)
        scatter : bool
            Scatter beam pairs over ranks (default : True)
        reuse_spinmaps : bool
            Do not calculate spinmaps when spinmaps attribute
            exists. Useful when iterating single beam per core.
            (default : False)
        interp : bool
            If set, use bi-linear interpolation to obtain TOD.
            (default : False)
        kwargs : {ces_opts, spinmaps_opts}
            Extra kwargs are assumed input to
            `constant_el_scan()` or `init_spinmaps()`
        '''

        if verbose and self.mpi_rank == 0:
            print('Scanning with {:d} detectors'.format(
                self.ndet))

            sys.stdout.flush()
        self.barrier() # just to have summary print statement on top

        # init memmap on root
        if create_memmap and not self.ext_point:
            if self.mpi_rank == 0:
                self.mmap = np.memmap('q_bore.npy', dtype=float,
                                      mode='w+', shape=(self.nsamp, 4),
                                      order='C')
            else:
                self.mmap = None

        # Scatter beams if needed, self.beams stays broadcasted
        if scatter:
            beams = self.scatter_list(self.beams, root=0)
        else:
            beams = self.beams

        # let every core loop over max number of beams per core
        # this makes sure that cores still participate in
        # calculating boresight quaternion
        nmax = int(np.ceil(self.ndet/float(self.mpi_size)/2.))

        for bidx in xrange(nmax):

            if bidx > 0:
                # reset instrument
                self.reset_instr_rot()
                self.reset_hwp_mod()
                self.reset_el_steps()

            try:
                beampair = beams[bidx]
            except IndexError:
                beampair = [None, None]

            beam_a = beampair[0]
            beam_b = beampair[1]

            if verbose == 2:
                print('\n[rank {:03d}]: working on: \n{} \n{}'.format(
                        self.mpi_rank, str(beam_a), str(beam_b)))
            if verbose == 1 and beam_a and beam_b:
                print('[rank {:03d}]: working on: {}, {}'.format(
                        self.mpi_rank, beam_a.name, beam_b.name))

            # Pop init_spinmaps kwargs and initialize spinmaps.
            max_spin = kwargs.pop('max_spin', 5)
            nside_spin = kwargs.pop('nside_spin', 256)
            spinmaps_opts = dict(max_spin=max_spin,
                                 nside_spin=nside_spin,
                                 verbose=(verbose==2))

            # Skip creating spinmaps when no beams or spinmaps
            # already initialized with reuse_spinmaps.
            if not beam_a and not beam_b:
                pass
            elif hasattr(self, 'spinmaps') and reuse_spinmaps:
                pass
            else:
                self.init_detpair(alm, beam_a, beam_b=beam_b,
                                  **spinmaps_opts)

            if not hasattr(self, 'chunks'):
                # Assume no chunking is needed and use full mission length.
                self.partition_mission()

            for cidx, chunk in enumerate(self.chunks):

                if verbose:
                    print(('[rank {:03d}]:\tWorking on chunk {:03}:'
                           ' samples {:d}-{:d}').format(self.mpi_rank,
                            cidx, chunk['start'], chunk['end']))

                # Make the boresight move.
                ces_opts = kwargs.copy()
                ces_opts.update(chunk)

                # Use precomputed pointing on subsequent passes.
                # Note, not used if memmap is not initialized.
                if bidx > 0:
                    ces_opts.update(dict(use_precomputed=True))
                self.constant_el_scan(**ces_opts)

                # If required, loop over boresight rotations.
                subchunks = self.subpart_chunk(chunk)
                for subchunk in subchunks:

                    if verbose == 2:
                        print(('[rank {:03d}]:\t\t...'
                               ' samples {:d}-{:d}, norot={}').format(self.mpi_rank,
                                                 subchunk['start'], subchunk['end'],
                                                       subchunk.get('norot', False)))

                    # rotate instrument and hwp if needed
                    if not subchunk.get('norot', False):
                        self.rotate_instr()

                    self.rotate_hwp(**subchunk)

                    # scan and bin
                    if beam_a and not beam_a.dead:
                        self.scan(beam_a, interp=interp,
                                  **subchunk)

                        # add ghost signal if present
                        if any(beam_a.ghosts):
                            for ghost in beam_a.ghosts:
                                self.scan(ghost,
                                          add_to_tod=True,
                                          interp=interp,
                                          **subchunk)

                        if binning:
                            self.bin_tod(add_to_global=True)

                    if beam_b and not beam_b.dead:
                        self.scan(beam_b, interp=interp,
                                  **subchunk)

                        if any(beam_b.ghosts):
                            for ghost in beam_b.ghosts:
                                self.scan(ghost,
                                          add_to_tod=True,
                                          interp=interp,
                                          **subchunk)

                        if binning:
                            self.bin_tod(add_to_global=True)

    def step_array(self, arr, step_dict, step_gen):
        '''
        Step array based on the properties in
        the `step_dict` atrribute. Performs
        in-place calculations and modifications of `step_dict`

        Arguments
        ---------
        arr : array-like
            Array with coordinates to be stepped
        step_dict : dict
            Dictionary containing "step_size", "remainder",
            and "angle" keys.
        step_gen : iterable
            Python iterable that can cycle through steps,
            see `tools.angle_gen` for example.

        Returns
        -------
        arr_stepped : array-like
            Stepped input array
        '''

        chunk_size = arr.size

        step_size = step_dict['step_size']
        nsteps = int(np.ceil((chunk_size - step_dict['remainder']) / float(step_size)))
        nsteps = 1 if nsteps < 1 else nsteps

        if nsteps == 1:
            # step period is larger or equal to chunksize
            if step_dict['remainder'] >= chunk_size:

                arr[:] += step_dict['angle']
                step_dict['remainder'] -= chunk_size
            else:
                arr[:step_dict['remainder']] += step_dict['angle']

                step_dict['angle'] = step_gen.next()
                arr[step_dict['remainder']:] += step_dict['angle']

                step_dict['remainder'] = step_size - arr[step_dict['remainder']:].size

            return arr

        elif nsteps > 1:
            # You can fit at most nstep - 1 full steps in chunk.
            # Remainder is at most stepsize.
            if step_dict['remainder']:
                arr[:step_dict['remainder']] += step_dict['angle']

            startidx = step_dict['remainder']
            # Loop over full steps.
            for step in xrange(nsteps-1):
                endidx = startidx + step_size

                step_dict['angle'] = step_gen.next()
                arr[startidx:endidx] += step_dict['angle']

                startidx = endidx

            # Fill last part and determine remainder.
            step_dict['angle'] = step_gen.next()
            arr[endidx:] += step_dict['angle']
            step_dict['remainder'] = step_size - arr[endidx:].size

            return arr

    def constant_el_scan(self, ra0=-10, dec0=-57.5, az_throw=90,
        scan_speed=1, vel_prf='triangle',
        check_interval=600, el_min=45, cut_el_min=False,
        use_precomputed=False, q_bore_func=None,
        q_bore_kwargs=None, ctime_func=None,
        ctime_kwargs=None, **kwargs):
        '''
        Populates scanning quaternions.
        Let boresight scan back and forth in azimuth, starting
        centered at ra0, dec0, while keeping elevation constant.

        Keyword Arguments
        ---------
        ra0 : float, array-like
            Ra coordinate of centre of scan in degrees.
            If array, consider items as scan
            centres ordered by preference.
        dec0 : float, array-like
            Dec coordinate of centre of scan in degrees.
            If array, same shape as ra0.
        az_throw : float
            Scan width in azimuth (in degrees)
        scan_speed : float
            Max scan speed in degrees per second
        vel_prf : str
            Velocity profile. Current options:
                triangle : (default) triangle wave with total
                           width=az_throw
        check_interval : float
            Check whether elevation is not below `el_min`
            at this rate in seconds (default : 600)
        el_min : float
            Lower elevation limit in degrees (default : 45)
        cut_el_min: bool
            If True, excludes timelines where el would be less than el_min
        use_precomputed : bool
            Load up precomputed boresight quaternion if
            memory-map is present (default : False)
        q_bore_func : callable, None
            A user-defined function that takes `start` and `end` (kw)args and
            outputs a (unit) quaternion array of shape=(nsamp, 4) on all ranks.
            Here, nsamp = end-start. Used when `external_pointing` is set in
            `ScanStrategy.__init__`. (default : None)
        q_bore_kwargs : dict, None
            Keyword arguments to q_bore_func (default: None)
        ctime_func : callable, None
            A user-defined function that takes `start` and `end` (kw)args and
            outputs a ctime array of shape=(nsamp) on all ranks.
            Here, nsamp = end-start. Used when `external_pointing` is set in
            `ScanStrategy.__init__`. (default : None)
        ctime_kwargs : dict, None
            Keyword arguments to ctime_func (default: None)
        start : int
            Start index
        end : int
            End index

        Notes
        -----
        When using `external_pointing` is set, this method just loads
        the external pointing using the provided functions and ignores
        every other kwarg except `start` and `end`.
        '''

        start = kwargs.pop('start')
        end = kwargs.pop('end')

        if self.ext_point:
            # Use external pointing, so skip rest of function.
            self.ctime = ctime_func(start=start, end=end, **ctime_kwargs)
            self.q_bore = q_bore_func(start=start, end=end, **q_bore_kwargs)

            return

        ctime = np.arange(start, end, dtype=float)
        ctime /= float(self.fsamp)
        ctime += self.ctime0
        self.ctime = ctime

        # Read q_bore from disk if needed (and skip rest).
        if use_precomputed and hasattr(self, 'mmap'):
            if self.mpi_rank == 0:
                self.q_bore = self.mmap[start:end]
            else:
                self.q_bore = None

            self.q_bore = self.broadcast_array(self.q_bore)

            return

        chunk_size = end - start
        check_len = int(check_interval * self.fsamp) # min_el checks

        nchecks = int(np.ceil(chunk_size / float(check_len)))
        p_len = check_len * nchecks # longer than chunk for nicer slicing

        ra0 = np.atleast_1d(ra0)
        dec0 = np.atleast_1d(dec0)
        npatches = dec0.shape[0]

        az0, el0, _ = self.radec2azel(ra0[0], dec0[0], 0,
            self.lon, self.lat, ctime[::check_len])

        flag0 = np.zeros(el0.size, dtype=bool)

        # check and fix cases where boresight el < el_min
        n = 1
        while np.any(el0 < el_min):

            if n < npatches:
                # run check again with other ra0, dec0 options
                azn, eln, _ = self.radec2azel(ra0[n], dec0[n], 0,
                                      self.lon, self.lat, ctime[::check_len])

                elidx = (el0<el_min)
                el0[elidx] = eln[elidx]
                az0[elidx] = azn[elidx]

            else:
                elidx = (el0<el_min)
                el0[elidx] = el_min

                if cut_el_min:
                    # not scanning this elevation check chunk
                    flag0[elidx] = True
                    warn('Cutting el min', RuntimeWarning)

                else:
                    # scanning at fixed elevation
                    warn('Keeping el0 at {:.1f} for part of scan'.format(el_min),
                        RuntimeWarning)
            n += 1

        # Scan boresight, note that it will slowly drift away from az0, el0
        if vel_prf is 'triangle':
            if scan_speed == 0:
                # Replace with small number to simulate staring.
                scan_speed = 1e-12
            scan_period = 2 * az_throw / float(scan_speed)
            if scan_period == 0.:
                az = np.zeros(chunk_size)
            else:
                az = np.arange(p_len, dtype=float)
                az *= (2 * np.pi / scan_period / float(self.fsamp))
                np.sin(az, out=az)
                np.arcsin(az, out=az)
                az *= (az_throw / np.pi)

        # slightly complicated way to multiply az with az0
        # while avoiding expanding az0 to p_len
        az = az.reshape(nchecks, check_len)
        az += az0[:, np.newaxis]
        az = az.ravel()
        az = az[:chunk_size] # discard extra entries

        el = np.zeros((nchecks, check_len), dtype=float)
        el += el0[:, np.newaxis]
        el = el.ravel()
        el = el[:chunk_size]

        flag = np.zeros((nchecks, check_len), dtype=bool)
        flag += flag0[:, np.newaxis]
        flag = flag.ravel().astype(bool)
        flag = flag[:chunk_size]

        self.flag = flag

        # do elevation stepping if necessary
        if self.step_dict.get('period', None):
            el = self.step_array(el, self.step_dict, self.el_step_gen)

        # Transform from horizontal frame to celestial, i.e. az, el -> ra, dec
        if self.mpi:
            # Calculate boresight quaternion in parallel

            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = np.divmod(chunk_size,
                                        self.mpi_size)
            sub_size += quot

            if remainder:
                # give first ranks one extra quaternion
                sub_size[:int(remainder)] += 1

            sub_start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            sub_end = sub_start + sub_size[self.mpi_rank]

            q_bore = np.empty(chunk_size * 4, dtype=float)

            # calculate section of q_bore
            q_boresub = self.azel2bore(az[sub_start:sub_end],
                                    el[sub_start:sub_end],
                                    None, None, self.lon, self.lat,
                                    ctime[sub_start:sub_end])
            q_boresub = q_boresub.ravel()

            sub_size *= 4 # for the flattened quat array

            offsets = np.zeros(self.mpi_size)
            offsets[1:] = np.cumsum(sub_size)[:-1] # start * 4

            # combine all sections on all ranks
            self._comm.Allgatherv(q_boresub,
                            [q_bore, sub_size, offsets, self._mpi_double])
            self.q_bore = q_bore.reshape(chunk_size, 4)

        else:
            self.q_bore = self.azel2bore(az, el, None, None, self.lon,
                                         self.lat, ctime)

        # store boresight quat in memmap if needed
        if hasattr(self, 'mmap'):
            if self.mpi_rank == 0:
                self.mmap[start:end] = self.q_bore
            # wait for I/O
            if self.mpi:
                self._comm.barrier()

    def satellite_ctime(self, **kwargs):
        '''
        Insert text
        '''

        start = kwargs.pop('start')
        end = kwargs.pop('end')

        ctime = np.arange(start, end, dtype=float)
        ctime /= float(self.fsamp)
        ctime += self.ctime0

        return ctime

    def satellite_scan(self, alpha=50., beta=50.,
        alpha_period=5400., beta_period=600., jitter_amp=0.0, return_all=False,
        **kwargs):
        '''
        A function to simulate satellite scanning strategy.

        Keyword arguments
        -----------------
        alpha : float
            Angle between spin axis and precession axis in degree.
            (default : 50.)
        beta : float
            Angle between spin axis and boresight in degree.
            (default : 50.)
        alpha_period : float
            Time period for precession in seconds. (default : 5400)
        beta_period : float
            Spin period in seconds. (default : 600.)
        jitter_amp : float
            Std of iid Gaussian noise added to elevation coords.
            (default : 0.0)
        return_all : bool
            Also return az, el, lon, lat. (default : False)
        start : int
            Start index
        end : int
            End index

        Returns
        -------
        (az, el, lon, lat,) q_bore : array-like
            Depending on return_all

        Notes
        -----
        See Wallis et al., 2017, MNRAS, 466, 425.
        '''

        deg_per_day = 360.9863
        dt = 1 / self.fsamp
        nsamp = self.ctime.size # ctime determines chunk size
        ndays = float(nsamp) / self.fsamp / (24 * 3600)

        az = np.mod(np.arange(nsamp)*dt*360/beta_period, 360)

        if jitter_amp != 0.:
            jitter = jitter_amp * np.random.randn(int(nsamp))
            el = beta * np.ones_like(az) + jitter
        else:
            el = beta * np.ones_like(az)

        # Continue from position left chunk finished
        if self.lon:
            lon_0 = self.lon
        else:
            # no position yet. So start at zero
            lon_0 = 0.

        if self.lat:
            lat_0 = self.lat
        else:
            lat_0 = 0.

        # Anti sun at all times
        lon = np.mod(-np.linspace(lon_0, ndays * deg_per_day + lon_0, nsamp),
                     360.)
        # the starting argument of the sin. Zero if lat_0 = 0.
        t_start = np.arcsin(lat_0 / float(alpha))

        lat = np.sin(np.linspace(
            t_start, 2*np.pi*dt*nsamp/alpha_period + t_start,
            num=nsamp, endpoint=False))
        lat *= alpha

        # Store last lon, lat coord for start next chunk
        self.lon = lon[-1]
        self.lat = lat[-1]

        if self.mpi:
            # Calculate boresight quaternion in parallel
            chunk_size = nsamp
            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = np.divmod(chunk_size,
                                        self.mpi_size)
            sub_size += quot

            if remainder:
                # give first ranks one extra quaternion
                sub_size[:int(remainder)] += 1

            sub_start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            sub_end = sub_start + sub_size[self.mpi_rank]

            q_bore = np.empty(chunk_size * 4, dtype=float)

            # calculate section of q_bore
            q_boresub = self.azel2bore(az[sub_start:sub_end],
                                    el[sub_start:sub_end],
                                    None, None,
                                    lon[sub_start:sub_end],
                                    lat[sub_start:sub_end],
                                    self.ctime[sub_start:sub_end])
            q_boresub = q_boresub.ravel()

            sub_size *= 4 # for the flattened quat array

            offsets = np.zeros(self.mpi_size)
            offsets[1:] = np.cumsum(sub_size)[:-1] # start * 4

            # combine all sections on all ranks
            self._comm.Allgatherv(q_boresub,
                            [q_bore, sub_size, offsets, self._mpi_double])
            q_bore = q_bore.reshape(chunk_size, 4)

        else:
            q_bore = self.azel2bore(az, el, None, None, lon,
                                         lat, self.ctime)
        self.flag = np.zeros_like(az, dtype=bool)

        if return_all:
            return az, el, lon, lat, q_bore
        else:
            return q_bore

    def rotate_hwp(self, **kwargs):
        '''
        Evolve the HWP forward by a number of samples
        given by `chunk_size` (given the current HWP
        parameters). See `set_hwp_mod()`. Stores array
        with HWP angle per sample if HWP frequency is
        set. Otherwise stores current HWP angle.
        Stored HWP angle should be in radians.

        Keyword arguments
        ---------
        hwpang : float, None
            Default HWP angle used when no HWP rotation is specified
            (see `set_hwp_mod()`). If not given, use current angle.
            (in degrees)
        start : int
            Start index
        end : int
            End index
        '''

        start = kwargs.get('start')
        end = kwargs.get('end')

        chunk_size = int(end - start) # size in samples

        # If HWP does not move, just return current angle
        if not self.hwp_dict['freq'] or not self.hwp_dict['mode']:

            if kwargs.get('hwpang'):
                self.hwp_ang = np.radians(kwargs.get('hwpang'))
            else:
                self.hwp_ang = np.radians(self.hwp_dict['angle'])

            return

        # if needed, compute hwp angle array.
        freq = self.hwp_dict['freq'] # cycles per sec for cont. rot hwp
        start_ang = np.radians(self.hwp_dict['angle'])

        if self.hwp_dict['mode'] == 'continuous':

            self.hwp_ang = np.linspace(start_ang,
                   start_ang + 2 * np.pi * chunk_size / (self.fsamp / float(freq)),
                   num=chunk_size, endpoint=False, dtype=float) # radians (w = 2 pi freq)

            # update mod 2pi start angle for next chunk
            self.hwp_dict['angle'] = np.degrees(np.mod(self.hwp_ang[-1], 2*np.pi))
            return

        if self.hwp_dict['mode'] == 'stepped':

            hwp_ang = np.zeros(chunk_size, dtype=float)
            hwp_ang = self.step_array(hwp_ang, self.hwp_dict, self.hwp_angle_gen)
            np.radians(hwp_ang, out=hwp_ang)

            # update mod 2pi start angle for next chunk
            self.hwp_dict['angle'] = np.degrees(np.mod(hwp_ang[-1], 2*np.pi))
            self.hwp_ang = hwp_ang

    def scan(self, beam_obj, add_to_tod=False, interp=False,
             **kwargs):
        '''
        Update boresight pointing with detector offset, and
        use it to bin spinmaps into a tod.

        Arguments
        ---------
        beam_obj : <detector.Beam> object

        Kewword arguments
        ---------
        add_to_tod : bool
            Add resulting TOD to existing tod attribute and do not
            internally store the detector offset pointing.
            (default: False)
        cidx : int, None
            Index to `chunks` attribute. Only needed when
            scanning a subset of a chunk (default : None)
        interp : bool
            If set, use bi-linear interpolation to obtain TOD.
            (default : False)
        start : int
            Start index
        end : int
            End index
        '''

        if beam_obj.dead:
            warn('scan() called with dead beam')
            return

        start = kwargs.get('start')
        end = kwargs.get('end')

        az_off = beam_obj.az
        el_off = beam_obj.el
        polang = beam_obj.polang_truth # True polang for scanning.

        if not beam_obj.ghost:
            func, func_c = self.spinmaps['main_beam']
        else:
            ghost_idx = beam_obj.ghost_idx
            func, func_c = self.spinmaps['ghosts'][ghost_idx]

        # Extract N (= max_spin + 1) and nside of spin maps
        N, npix = func.shape
        nside_spin = hp.npix2nside(npix)

        # Paranoid android
        N2, npix2 = func_c.shape
        assert 2*N-1 == N2, "func and func_c have different max_spin"
        assert npix == npix2, "func and func_c have different npix"

        # We use a offset quaternion without polang.
        # We apply polang at the beam level later.
        q_off = self.det_offset(az_off, el_off, 0)

        # Expose pointing offset for mapmaking
        if not add_to_tod:
            self.q_off = q_off
            self.polang = beam_obj.polang # Possibly offset polang for binning.
        
        # Rotate offset given rot_dict. We rotate the centroid
        # around the boresight. It's q_bore * q_rot * q_off
        ang = np.radians(self.rot_dict['angle'])
        q_rot = np.asarray([np.cos(ang/2.), 0., 0., np.sin(ang/2.)])
        q_off = tools.quat_left_mult(q_rot, q_off)

        tod_size = end - start  # size in samples

        tod_c = np.zeros(tod_size, dtype=np.complex128)

        # Find the indices to the pointing and ctime arrays
        if 'cidx' in kwargs:
            cidx = kwargs['cidx']
            qidx_start = start - self.chunks[cidx]['start']
            # qidx_end = qidx_start + end - start + 1
            # qidx_end = start - chunks['start'] + end - start + 1
            qidx_end = end - self.chunks[cidx]['start']
        else:
            qidx_start = 0
            qidx_end = end - start

        self.qidx_start = qidx_start
        self.qidx_end = qidx_end

        # more efficient if you do bore2pix, i.e. skip
        # the allocation of ra, dec, pa etc. But you need pa....
        # Perhaps ask Sasha if she can make bore2pix output pix
        # and pa (instead of sin2pa, cos2pa)
        ra = np.empty(tod_size, dtype=np.float64)
        dec = np.empty(tod_size, dtype=np.float64)
        pa = np.empty(tod_size, dtype=np.float64)

        self.bore2radec(q_off,
                       self.ctime[self.qidx_start:self.qidx_end],
                       self.q_bore[qidx_start:qidx_end],
                       q_hwp=None, sindec=False, return_pa=True,
                       ra=ra, dec=dec, pa=pa)

        np.radians(pa, out=pa)
        
        if interp:
            # Convert qpoint output to input healpy (in-place).
            tools.radec2colatlong(ra, dec)
        else:
            pix = tools.radec2ind_hp(ra, dec, nside_spin)

            # Expose pixel indices for test centroid.
            self.pix = pix

        # Fill complex array, i.e. the linearly polarized part
        # Init arrays used for recursion: exp i n pa = (exp i pa) ** n
        # to avoid doing triginometry at each n.
        expipa = np.exp(1j * pa) # used for recursion
        expipan = np.exp(1j * pa * (-N + 1)) # starting point (n = -N+1)

        for nidx, n in enumerate(xrange(-N+1, N)):

            if nidx != 0: # expipan is already initialized for nidx=0
                expipan *= expipa

            if interp:
                tod_c += hp.get_interp_val(func_c[nidx], dec, ra) \
                         * expipan
            else:
                tod_c += func_c[nidx,pix] * expipan

        # check for HWP angle array
        if self.hwp_ang is None:
            hwp_ang = 0

            if self.hwp_dict:
                # HWP options set, but not executed
                warn('call rotate_hwp() to have HWP modulation',
                     RuntimeWarning)
        else:
            hwp_ang = self.hwp_ang

        # Modulate by hwp angle and polarization angle
        expm2 = np.exp(1j * (4 * hwp_ang + 2 * np.radians(polang)))
        tod_c[:] = np.real(tod_c * expm2 + np.conj(tod_c * expm2)) / 2.
        tod = np.real(tod_c) # shares memory with tod_c

        # Add unpolarized tod.
        # Reset starting point recursion.
        expipan[:] = 1.
        for n in xrange(N):

            # Equal to 0.5 ( func exp(n pa) + c.c) (pa: position angle).
            if n == 0:
                if interp:
                    tod += np.real(hp.get_interp_val(func[n], dec, ra))
                else:
                    tod += np.real(func[n,pix])

            if n > 0:
                expipan *= expipa

                if interp:
                    tod += np.real(hp.get_interp_val(func[n], dec, ra) \
                                   * expipan)
                else:
                    tod += np.real(func[n,pix] * expipan)

        if add_to_tod and hasattr(self, 'tod'):
            self.tod += tod
        else:
            self.tod = tod

    def init_spinmaps(self, alm, blm=None, max_spin=5, nside_spin=256,
                     verbose=True, beam_obj=None):
        '''
        Compute convolution of map with different spin modes
        of the beam. Computed per spin, so creates spinmmap
        for every s<= 0 for T and for every s for pol.

        Arguments
        ---------
        alm : tuple of array-like
            Tuple of (alm, almE, almB)
        blm : tuple of array-like
            Tuple of (blmI, blmm2, blmp2)

        Keyword arguments
        -----------------
        max_spin : int, optional
            Maximum azimuthal mode describing the beam
            (default : 5)
        nside_spin : int
            Nside of spin maps (default : 256)
        beam_obj : <detector.Beam> object
            If provided, create spinmaps for main beam and
            all ghosts (if present). `ghost_idx` attribute
            decides whether ghosts have distinct beams. See
            `Beam.__init__()`

        Notes
        -----
        Uses minimum lmax value of beam and sky SWSH coefficients.
        Uses min value of max_spin and the mmax of beam object as azimuthal
        band-limit.
        '''

        # NOTE it would be nice to have a symmetric beam option
        # that only makes it run over n=0, -2 and 2.

        # this block iterates the function to create spinmaps for
        # the main beam and unique ghost beams (if present)
        # spinmaps are stored internally in self.spinmaps dict.
        if beam_obj:

            self.spinmaps = {'main_beam' : [],
                             'ghosts': []}

            max_s = min(beam_obj.mmax, max_spin)
            blm = beam_obj.blm # main beam

            # calculate spinmaps for main beam
            func, func_c = self.init_spinmaps(alm, blm=blm, max_spin=max_s,
                                              nside_spin=nside_spin,
                                              verbose=verbose)
            self.spinmaps['main_beam'][:] = func, func_c

            if beam_obj.ghosts:

                # find unique ghost beams
                assert len(beam_obj.ghosts) == beam_obj.ghost_count

                g_indices = np.empty(beam_obj.ghost_count, dtype=int) # ghost_indices

                for gidx, ghost in enumerate(beam_obj.ghosts):
                    g_indices[gidx] = ghost.ghost_idx

                unique, u_indices = np.unique(g_indices, return_index=True)

                # calculate spinmaps for unique ghost beams
                for uidx, u in enumerate(unique):

                    self.spinmaps['ghosts'].append([])

                    # use the blms from the first occurrence of unique
                    # ghost_idx
                    ghost = beam_obj.ghosts[u_indices[uidx]]

                    blm = ghost.blm
                    max_s = min(ghost.mmax, max_spin)

                    func, func_c = self.init_spinmaps(alm, blm=blm, max_spin=max_s,
                                                      nside_spin=nside_spin,
                                                      verbose=verbose)

                    self.spinmaps['ghosts'][u][:] = func, func_c
            return

        # Check for nans in alms. E.g from when there was a nan in original maps.
        # If so, crash. Otherwise healpy will just create nan alms.
        crash_a = False
        crash_b = False
        i = 0
        while i < 3 and not (crash_a or crash_b):
            crash_a = ~np.isfinite(np.sum(alm[i][:]))
            crash_b = ~np.isfinite(np.sum(blm[i][:]))
            i += 1
        if crash_a or crash_b:
            name = 'alm' if crash_a else 'blm'
            raise ValueError('{}[{}] contains nan/inf.'.format(name, i-1))

        N = max_spin + 1
        lmax = hp.Alm.getlmax(alm[0].size)

        # Match up bandlimits beam and sky.
        lmax_beam = hp.Alm.getlmax(blm[0].size)

        if lmax > lmax_beam:
            alm = tools.trunc_alm(alm, lmax_beam)
            lmax = lmax_beam
        elif lmax_beam > lmax:
            blm = tools.trunc_alm(blm, lmax)

        # Unpolarized sky and beam first.
        func = np.zeros((N, 12*nside_spin**2),
                             dtype=np.complex128) # only s >=0 spheres

        start = 0
        for s in xrange(N): # s are the azimuthal modes in bls.
            end = lmax + 1 - s
            if s == 0: # Scalar transform.

                flms = hp.almxfl(alm[0], blm[0][start:start+end], inplace=False)
                func[s,:] += hp.alm2map(flms, nside_spin, verbose=False)

            else: # Spin transforms.

                bell = np.zeros(lmax+1, dtype=np.complex128)
                # s beam modes.
                bell[s:] = blm[0][start:start+end]

                flms = hp.almxfl(alm[0], bell, inplace=False)
                flmms = hp.almxfl(alm[0], np.conj(bell), inplace=False)

                # Turn into plus and minus (think E and B) modes for healpy's
                # alm2map_spin.
                flmsp = - (flms + flmms) / 2.
                flmsm = 1j * (flms - flmms) / 2.
                spinmaps = hp.alm2map_spin([flmsp, flmsm], nside_spin, s, lmax,
                                           lmax)
                func[s,:] = spinmaps[0] + 1j * spinmaps[1]

            start += end

        # Pol (all s spheres).
        func_c = np.zeros((2*N-1, 12*nside_spin**2), dtype=np.complex128)

        almp2 = -1 * (alm[1] + 1j * alm[2])
        almm2 = -1 * (alm[1] - 1j * alm[2])

        blmm2 = blm[1]
        blmp2 = blm[2]

        start = 0
        for s in xrange(N):
            end = lmax + 1 - s

            bellp2 = np.zeros(lmax+1, dtype=np.complex128)
            bellm2 = bellp2.copy()

            bellp2[np.abs(s):] = blmp2[start:start+end]
            bellm2[np.abs(s):] = blmm2[start:start+end]

            ps_flm_p = hp.almxfl(almp2, bellm2, inplace=False) + \
                hp.almxfl(almm2, np.conj(bellm2), inplace=False)
            ps_flm_p /= -2.

            ps_flm_m = hp.almxfl(almp2, bellm2, inplace=False) - \
                hp.almxfl(almm2, np.conj(bellm2), inplace=False)
            ps_flm_m *= 1j / 2.

            ms_flm_p = hp.almxfl(almm2, bellp2, inplace=False) + \
                hp.almxfl(almp2, np.conj(bellp2), inplace=False)
            ms_flm_p /= -2.

            ms_flm_m = hp.almxfl(almm2, bellp2, inplace=False) - \
                hp.almxfl(almp2, np.conj(bellp2), inplace=False)
            ms_flm_m *= 1j / 2.

            if s == 0:
                spinmaps = [hp.alm2map(-ps_flm_p, nside_spin, verbose=False),
                            hp.alm2map(-ms_flm_m, nside_spin, verbose=False)]

                func_c[N-s-1,:] = spinmaps[0] - 1j * spinmaps[1]

            else:
                # Positive s.
                spinmaps = hp.alm2map_spin([ps_flm_p, ps_flm_m],
                                           nside_spin, s, lmax, lmax)
                func_c[N+s-1,:] = spinmaps[0] + 1j * spinmaps[1]

                # Negative s.
                spinmaps = hp.alm2map_spin([ms_flm_p, ms_flm_m],
                                           nside_spin, s, lmax, lmax)
                func_c[N-s-1,:] = spinmaps[0] - 1j * spinmaps[1]

            start += end

        return func, func_c

    def bin_tod(self, az_off=None, el_off=None, polang=None,
                init=True, add_to_global=True):
        '''
        Take internally stored tod and boresight
        pointing, combine with detector offset,
        and bin into map and projection matrices.

        Keyword arguments
        -----------------
        az_off : float
            The detector azimuthal offset relative to boresight in deg
            (default : None)
        el_off : float
            The detector elevation offset relative to boresight in deg
            (default : None)
        polang : float
            Polarization angle in deg (default : None)
        init : bool
            Call `init_dest()` before binning (default : True)
        add_to_global : bool
            Add local maps to maps allocated by `allocate_maps`
            (default : True)
        '''

        q_hwp = self.hwp_quat(np.degrees(self.hwp_ang)) #from qpoint

        self.init_point(q_bore=self.q_bore[self.qidx_start:self.qidx_end],
                        ctime=self.ctime[self.qidx_start:self.qidx_end],
                        q_hwp=q_hwp)

        # use q_off quat with polang (and instr. ang) included.
        q_off = self.q_off
        
        # Note minus sign. Also not true polang when polang_error != 0.
        polang = -np.radians(self.polang) 
        q_polang = np.asarray([np.cos(polang/2.), 0., 0., np.sin(polang/2.)])
        q_off = tools.quat_left_mult(q_off, q_polang)

        if init:
            self.init_dest(nside=self.nside_out, pol=True, reset=True)

        q_off = q_off[np.newaxis]
        tod = self.tod[np.newaxis]
        flag = self.flag[self.qidx_start:self.qidx_end]
        flag = flag[np.newaxis]

        self.from_tod(q_off, tod=tod, flag=flag)

        if add_to_global:
            # Add local maps to global maps.
            self.vec += self.depo['vec']
            self.proj += self.depo['proj']

    def solve_for_map(self, fill=hp.UNSEEN, return_proj=False):
        '''
        Solve for the output map given the stored
        vec map and proj matrix.
        If MPI, reduce maps to root and solve there.

        Keyword arguments
        -----------------
        fill : scalar
            Fill value for unobserved pixels
            (default : hp.UNSEEN)
        return_proj : bool
            Also return proj matrix (default : False)

        Returns
        -------
        maps : array-like
            Solved I, Q and U maps in shape (3, npix)
        cond : array-like
            Condition number map
        proj : array-like
            (Only when `return_proj` is set) Projection
            matrix (proj)
        '''

        if self.mpi:
            # Collect the binned maps on the root process.
            vec = self.reduce_array(self.vec)
            proj = self.reduce_array(self.proj)
        else:
            vec = self.vec
            proj = self.proj

        # Solve map on root process.
        if self.mpi_rank == 0:
            # Suppress 1/0 warnings from numpy linalg.
            with catch_warnings(RuntimeWarning):
                simplefilter("ignore")

                maps = self.solve_map(vec=vec, proj=proj,
                                      copy=True, fill=fill)
            cond = self.proj_cond(proj=proj)
            cond[cond == np.inf] = fill
        else:
            maps = None
            cond = None

        if return_proj:
            return maps, cond, proj

        return maps, cond

