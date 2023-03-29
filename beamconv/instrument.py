import os
import sys
import time
import copy
from warnings import warn, catch_warnings, filterwarnings
import glob
import pickle

import numpy as np
import qpoint as qp
import healpy as hp

from . import scanning
from . import tools
from .detector import Beam

class MPIBase(object):
    '''
    Parent class for MPI related stuff
    '''

    def __init__(self, mpi=True, comm=None, **kwargs):
        '''
        Check if MPI is working by checking common
        MPI environment variables and set MPI attributes.

        Keyword arguments
        -----------------
        mpi : bool
            If False, do not use MPI regardless of MPI env.
            otherwise, let code decide based on env. vars
            (default : True)
        comm : MPI.comm object, None
            External communicator. If left None, create
            communicator. (default : None)
        '''

        super(MPIBase, self).__init__(**kwargs)

        # Check whether MPI is working.
        # Add your own environment variable if needed.
        # Open MPI environment variable.
        ompi_size = os.getenv('OMPI_COMM_WORLD_SIZE')
        # intel and/or mpich environment variable.
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

        # Broadcast meta info first.
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
        arr : array-like, list
            Full-sized array present on every rank

        Returns
        -------
        arr_loc : array-like
            View of array unique to every rank.
        '''

        if self.mpi:

            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = divmod(len(arr), self.mpi_size)
            sub_size += quot

            if remainder:
                # Give first ranks extra element.
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
        lon : float, vec, None
            Longitude in degrees. (default : None)
        lat : float, vec, None
            Latitude in degrees (default : None)
        kwargs : {mpi_opts}

        Notes
        -----
        The Instrument holds a `beams` attribute that
        consists of a nested list reprenting pairs of
        beamconv.Beam objects (detectors). `beams` is
        equal on all MPI ranks.
        '''

        if location == 'spole':
            self.lat = -89.9
            self.lon = 169.15

        elif location == 'atacama':
            self.lat = -22.958
            self.lon = -67.786

        elif location == 'space':
            self.lat = None
            self.lon = None

        if lat is not None:
            self.lat = lat
        if lon is not None:
            self.lon = lon

        if (location != 'space') and (not hasattr(self, 'lat') or not hasattr(self, 'lon')):
            raise ValueError('Specify location of telescope')

        self.beams = []
        self.ndet = 0

        super(Instrument, self).__init__(**kwargs)

    def beams_idxs(self):
        '''
        Return indices of beams currently on focal plane.

        Returns
        -------
        idxs : array-like
            Array of beam indices.
        '''

        beam_idx = []
        for pair in self.beams:
            for beam in pair:
                if beam is None:
                    continue
                beam_idx.append(beam.idx)

        return np.asarray(beam_idx, dtype=int)

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

        Notes
        -----
        This method will overwrite the idx parameter of the
        beams.
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

        # Determine beam indices. If needed start counting from
        # highest existing index.
        if combine:
            idxs = self.beams_idxs()
            try:
                idx= idxs.max() + 1
            except ValueError:
                # Empty list.
                idx = 0
        else:
            idx = 0

        # If not pairs, we add None as partner.
        ndet2add = 0
        if isseq is False:
            beams2add = [[beams, None]]
            ndet2add += 1
            beams._idx = idx

        if isseq:
            beams2add = []
            for pair in beams:
                if isnestseq is False:
                    pair._idx = idx
                    pair = [pair, None]
                    ndet2add += 1
                    idx += 1
                else:
                    ndet2add += 2
                    pair[0]._idx = idx
                    pair[1]._idx = idx + 1
                    idx += 2

                beams2add.append(pair)

        if not combine:
            self.beams = beams2add
        else:
            self.beams += beams2add

        if not combine:
            self.ndet = ndet2add
        else:
            self.ndet += ndet2add

    def remove_from_focal_plane(self, bad_beams):
        '''
        Remove beam(s) from focal plane.

        Arguments
        ---------
        bad_beams : (list of) Beam instance(s)
            If nested list: assumed to be list of beam pairs.
        '''

        # Check whether single beam, list of beams or list of pairs.
        try:
            bad_beams[0]
        except TypeError:
            bad_beams = [bad_beams]

        try:
            bad_beams[0][0]
            isnestseq = True
        except TypeError:
            isnestseq = False

        # Flatten input.
        if isnestseq:
            bad_beams = [i for sublist in bad_beams for i in sublist]

        # Set bad beams to None.
        for pidx, pair in enumerate(self.beams):
            for bidx, beam in enumerate(pair):

                if beam in bad_beams:
                    self.beams[pidx][bidx] = None
                    self.ndet -= 1

        # Remove pairs where both beams are None.
        new_beams = []
        for pidx, pair in enumerate(self.beams):

            if pair != [None, None]:
                new_beams.append(pair)

        self.beams = new_beams

    def create_focal_plane(self, nrow=1, ncol=1, fov=10.,
                           no_pairs=False, ab_diff=90., 
                           combine=True, scatter=False, **kwargs):
        '''
        Create Beam objects for orthogonally polarized
        detector pairs with pointing offsets lying on a
        rectangular az-el grid on the sky.

        Keyword arguments
        -----------------
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
        ab_diff : float, optional
            Difference between A and B detectors in degrees
            e.g. if detector B is -Q to A the difference is 
            90 degrees (default : 90.)
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
        for key in ['az', 'el', 'pol', 'name', 'ghost', 'idx']:
            arg = kwargs.pop(key, None)
            if arg:
                warn('{}={} option to `Beam.__init__()` is ignored'
                     .format(key, arg))

        # Some kwargs need to be dealt with seperately.
        polang = kwargs.pop('polang', 0.)
        dead = kwargs.pop('dead', False)

        if combine:
            self.ndet += 2 * nrow * ncol
            idxs = self.beams_idxs()
            try:
                idx= idxs.max() + 1
            except ValueError:
                # Empty list.
                idx = 0
        else:
            self.ndet = 2 * nrow * ncol # A and B detectors.
            idx = 0

        beams=[]

        azs = np.linspace(-fov/2., fov/2., ncol)
        els = np.linspace(-fov/2., fov/2., nrow)

        for az_idx in range(azs.size):
            for el_idx in range(els.size):

                det_str = 'r{:03d}c{:03d}'.format(el_idx, az_idx)

                beam_a = Beam(az=azs[az_idx], el=els[el_idx],
                              name=det_str+'A', polang=polang,
                              dead=dead, pol='A', idx=idx,
                              **kwargs)

                beam_b = Beam(az=azs[az_idx], el=els[el_idx],
                              name=det_str+'B', polang=polang+ab_diff,
                              dead=dead or no_pairs, pol='B',
                              idx=idx+1, **kwargs)

                beams.append([beam_a, beam_b])
                idx += 2

        if scatter:
            # If MPI, distribute beams over ranks.
            # Distribute instead of scatter because all ranks
            # already have full list of beams.
            beams = self.distribute_array(beams)

        # Check for existing beams.
        if not combine:
            self.beams = beams
        else:
            self.beams += beams

    def input_focal_plane(self, azs, els, polangs, deads=None,
        combine=True, scatter=False, **kwargs):
        '''
        Create Beam objects for user-supplied pointing offsets and polangs

        Keyword arguments
        -----------------
        azs : array-like
            Detector az-offsets as an (npair x 2 array)
        els : array-like
            Detector el-offsets as an npair x 2 array
        polangs : array-like
            Detector polarization angle offsets
        deads : array-like (optional)
            Detector dead/alive values as an npair x 2 array
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

        if deads is None:
            deads = np.zeros_like(azs).astype(bool)

        if combine:
            self.ndet += 2 * len(azs)
            idxs = self.beams_idxs()
            try:
                idx= idxs.max() + 1
            except ValueError:
                # Empty list.
                idx = 0
        else:
            self.ndet = 2 * nrow * ncol # A and B detectors.
            idx = 0

        beams = []
        for i, (az, el, polang, dead) in enumerate(
                zip(azs, els, polangs, deads)):

            beam_a = Beam(az=az[0], el=el[0], name='det{}'.format(i),
                polang=polang[0], dead=dead[0], pol='A', idx=i, **kwargs)
            beam_b = Beam(az=az[1], el=el[1], name='det{}'.format(i),
                polang=polang[1], dead=dead[1], pol='B', idx=i, **kwargs)

            beams.append([beam_a, beam_b])
            idx += 2

        if scatter:
            # If MPI, distribute beams over ranks.
            # Distribute instead of scatter because all ranks
            # already have full list of beams.
            beams = self.distribute_array(beams)

        # Check for existing beams.
        if not combine:
            self.beams = beams
        else:
            self.beams += beams

    def load_focal_plane(self, bdir, tag=None, no_pairs=False,
            combine=True, scatter=False, polang_A=0., polang_B=0.,
            print_list=False, file_names=None, **kwargs):

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
            Polarization angle of B detector in pair [deg].
            Added to existing or provided polang. (default: 0.)
        file_names : list, None
            List of file names in directory that are loaded.
            No .pkl extension needed. Ignored if None. (default: None)
        print_list : bool
            Print list of beam files loaded up (for debugging).
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

        This method will overwrite the idx parameter of the
        beams.
        '''

        # Do all I/O on root.
        if self.mpi_rank == 0:

            if combine:
                idxs = self.beams_idxs()
                try:
                    idx= idxs.max() + 1
                except ValueError:
                    # Empty list.
                    idx = 0
            else:
                idx = 0

            beams = []

            # Ignore these kwargs and warn user.
            for key in ['pol', 'ghost', 'idx']:
                arg = kwargs.pop(key, None)
                if arg:
                    warn('{}={} option to `Beam.__init__()` is ignored'
                         .format(key, arg))

            opj = os.path.join
            tag = '' if tag is None else tag

            if file_names:
                file_list = []
                for fname in file_names:
                    if not tag in fname:
                        continue
                    ffile = opj(bdir, fname + '.pkl')
                    if not os.path.exists(ffile):
                        raise IOError("No such file: {}".format(ffile))
                    file_list.append(ffile)

            else:
                file_list = glob.glob(opj(bdir, '*' + tag + '*.pkl'))
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

                # Overrule options with given kwargs.
                beam_opts_a.update(kwargs)
                beam_opts_b.update(kwargs)

                beam_opts_a.update(dict(idx=idx))
                beam_opts_b.update(dict(idx=idx+1))

                idx += 2

                # Add polang A and B (perhaps on top of provided or existing
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
            # If MPI scatter to ranks.
            beams = self.scatter_list(beams, root=0)
        else:
            # Broadcast otherwise because root did all I/O.
            beams = self.broadcast(beams)

        # All ranks need to know total number of detectors.
        ndet = self.broadcast(ndet)

        # Check for existing beams.
        if not combine:
            self.beams = beams
        else:
            self.beams += beams

        if not hasattr(self, 'ndet') or not combine:
            self.ndet = ndet
        else:
            self.ndet += ndet

    def create_crosstalk_ghosts(self, azs, els,
            beams=None, ghost_tag='crosstalk_ghost', rand_stdev=0., **kwargs):
        '''
        Create crosstalk ghosts based on supplied detector
        offsets (azimuth and elevation).
        Polarization angles are rotated 90 deg by deafult.
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

        # Tag overrules ghost_tag.
        kwargs.setdefault('tag', ghost_tag)
        beams = np.atleast_2d(beams) #2D: we have pairs
        for pair, az_pair, el_pair in zip(beams, azs, els):
            for beam, az, el in zip(pair, az_pair, el_pair):
                dead_ghost = True if not beam or not az else False

                # Note, in python integers are immutable
                # so ghost offset is not updated when
                # beam offset is updated.
                crosstalk_ghost_opts = dict(az=az, el=el,
                    polang=beam.polang+90, dead=dead_ghost)

                kwargs.update(crosstalk_ghost_opts)

                if rand_stdev:
                    # Add perturbation to ghost amplitude.
                    amplitude = kwargs.get('amplitude', 1.)
                    amplitude += np.random.normal(scale=rand_stdev)
                    kwargs.update(dict(amplitude=amplitude))

                beam.create_ghost(**kwargs)

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

        # Tag overrules ghost_tag.
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
                    # Add perturbation to ghost amplitude.
                    amplitude = kwargs.get('amplitude', 1.)
                    amplitude += np.random.normal(scale=rand_stdev)
                    kwargs.update(dict(amplitude=amplitude))

                beam.create_ghost(**kwargs)

    def kill_channels(self, killfrac=0.2, pairs=False, rnd_state=None):
        '''
        Randomly identifies detectors in the beams list
        and sets their 'dead' attribute to True.

        Keyword arguments
        -----------------
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
            ndet = self.ndet // 2
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

    def set_global_prop(self, prop, incl_ghosts=True, no_B=False,
                        no_A=False):
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
        no_B : bool
            Do not set value for B beams. (default : False)
        no_A
            Do not set value for B beams. (default : False)

        Examples
        --------
        >>> S = Instrument()
        >>> S.create_focal_plane(nrow=10, ncol=10)
        >>> set_global_prop(dict(btype='PO'))

        Notes
        -----
        Ghosts share property with main beam.
        '''

        beams = np.atleast_2d(self.beams) #2D: we have pairs

        for pair in beams:
            for bidx, beam in enumerate(pair):

                # This assumes pairs are always [A, B].
                if no_B and bidx == 1:
                    continue
                if no_A and bidx == 0:
                    continue

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

    def set_global_prop_random(self, prop, incl_ghosts=True):
        '''
        Adds a random component to all beams on the focal plane.

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
        >>> set_global_prop_random(dict(polang_err=0.01))

        Notes
        -----
        Ghosts get different random deviations compared to main beam.

        '''

        beams = np.atleast_2d(self.beams) #2D: we have pairs

        for pair in beams:
            for beam in pair:

                if not beam:
                    continue

                for key in prop:
                    val = getattr(beam, key) + np.random.normal(1)*prop[key]
                    setattr(beam, key, val)

                if incl_ghosts:
                    for ghost in beam.ghosts:
                        for key in prop:
                            val = getattr(
                                ghost, key) + np.random.normal() * prop[key]
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
            Do not add value to B beams. (default : False)
        no_A
            Do not add value to A beams. (default : False)
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
        The attributes of ghosts and main beam are affected identically.
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

                # This assumes pairs are always [A, B].
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


    def __init__(self, duration=None, sample_rate=None, num_samples=None,
                 external_pointing=False, ctime0=None, **kwargs):
        '''
        Initialize scan parameters.

        Keyword arguments
        -----------------
        duration : float, None
            Mission duration in seconds (default : 0)
        sample_rate : float, None
            Sample rate in Hz. If `external_pointing` is set,
            make sure this matches the sample rate of the
            external pointing. (default : None)
        num_samples : int, None
            Number of samples in mission (default : None)
        external_pointing : bool
            If set, `implement_scan` loads up boresight pointing
            and time timestreams instead of calculating them.
            (default : False)
        ctime0 : float
            Start time in unix time. If None, use
            current time. Ignored if `external_pointing` is set
            (default : None)
        kwargs : {mpi_opts, instr_opts, qmap_opts}

        Notes
        -----
        Specify at least:

            duration, sample_rate
        or
            duration, nsamp
        or
            sample_rate, num_samples

        If duration, sample_rate and nsamp are given, they have to
        conform to num_samples = int(duration * sample_rate).
        '''

        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("Sample rate is not positive.")
        if duration == 0:
            # To avoid infinite samples.
            sample_rate = 0

        err_msg = ("Specify at least: duration and sample_rate or "
        "duration and num_samples or sample_rate and num_samples.")

        if duration is None:
            if num_samples is None:
                raise ValueError(err_msg)
            if sample_rate is None:
                raise ValueError(err_msg)
            duration = num_samples / float(sample_rate)

        elif sample_rate is None:
            if num_samples is None:
                raise ValueError(err_msg)
            sample_rate = num_samples / float(duration)

        else:
            # Sample_rate and duration are not None.
            nsamp = int(duration * sample_rate)
            if num_samples is not None:
                if nsamp != num_samples:
                    raise ValueError(
                        "num_samples != int(duration * sample_rate)")
            num_samples = nsamp

        self.__fsamp = float(sample_rate)
        self.__mlen = duration
        self.__nsamp = int(num_samples)

        self.ctime0 = ctime0

        self.ext_point = external_pointing

        self.rot_dict = {}
        self.hwp_dict = {}
        self.step_dict = {}
        self.filter_dict = {}
        self.set_instr_rot()
        self.set_hwp_mod()

        self._data = {}

        # Set some useful qpoint/qmap options as default.
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
        -----------------
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

        # Initialize rotation generators.
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
            self.rot_dict['angle'] = next(self.rot_angle_gen)

    def set_hwp_mod(self, mode=None,
                    freq=None, start_ang=0.,
                    angles=None, varphi=0.):
        '''
        Set options for modulating the polarized sky signal
        using a (stepped or continuously rotating) half-wave
        plate.

        Keyword arguments
        -----------------
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
        varphi : float, 0.
            If your HWP induces a phase shift of varphi between ingoing
            and outgoing polarization, adding its value to the dictionary
            ensures it will get substracted from hwp angles at bin_tod time
        '''

        self.hwp_dict['mode'] = mode
        self.hwp_dict['freq'] = freq
        if mode == 'stepped':
            self.hwp_dict['step_size'] = int(self.fsamp / float(freq))
        self.hwp_dict['angle'] = start_ang
        self.hwp_dict['start_ang'] = start_ang
        self.hwp_dict['remainder'] = 0
        self.hwp_dict['varphi'] = varphi

        if angles is None:
            angles = np.arange(start_ang, 360+start_ang, 22.5)
            np.mod(angles, 360, out=angles)

        self.hwp_dict['angles'] = angles

        # Initialize hwp ang generator.
        self.hwp_angle_gen = tools.angle_gen(angles)

    def reset_hwp_mod(self):
        '''
        "Reset" the hwp modulation generator.
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
        -----------------
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

        # Initialize el step generator.
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

    def set_filter_dict(self, w_c, m=1):
        '''
        Set options for the high-pass filtering of data chunks.
        
        Arguments
        ---------
        w_c : float
            Cutoff frequency in Hertz

        Keyword arguments
        -----------------
        m : int
            Order of the filter (default: 1)
        '''
        
        self.filter_dict["w_c"] = w_c
        self.filter_dict["m"] = m

    def partition_mission(self, chunksize=None):
        '''
        Divide up the mission in equal-sized chunks
        of nsample / chunksize (final chunk can be
        smaller).

        Keyword arguments
        -----------------
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

        for cidx, chunk in enumerate(range(nchunks)):
            end = start + chunksize
            end = nsamp if end >= (nsamp) else end
            chunks.append(dict(start=start, end=end, cidx=cidx))
            start += chunksize

            # Create slot for data, if needed later.
            self._data[str(cidx)] = {}

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
            Subchunks for each rotation of the boresight.
            
        Notes
        -----
        If input chunk had a `cidx` key, every subchunk inherits this value.
        '''

        period = self.rot_dict['period'] # Boresight rotation period.

        if not period:
        # No need for subchunks since the instrument is not rotating.
            return [chunk]
        # Samples in boresight rotation period.
        rot_chunk_size = self.rot_dict['rot_chunk_size'] 
        chunk_size = chunk['end'] - chunk['start'] # Samples in chunk.

        subchunks = []
        start = chunk['start']

        nchunks = (chunk_size - self.rot_dict['remainder'])
        nchunks /= float(rot_chunk_size)
        nchunks = int(np.ceil(nchunks))
        nchunks = 1 if nchunks < 1 else nchunks

        if nchunks == 1:
            # Rotation period larger or equal to chunksize.
            if self.rot_dict['remainder'] >= chunk_size:
                subchunks.append(dict(start=start, end=chunk['end']))
                self.rot_dict['remainder'] -= chunk_size

            else:
                # One subchunk that is just the remainder, if there is one.
                end = self.rot_dict['remainder'] + start

                if self.rot_dict['remainder']:
                    # In this rotation chunk, no rotation should be performed.
                    subchunks.append(dict(start=start, end=end, norot=True))

                # Another subchunk that is the rest of the chunk.
                subchunks.append(dict(start=end, end=chunk['end']))

                self.rot_dict['remainder'] = rot_chunk_size - (chunk['end'] - end)

        elif nchunks > 1:
            # You can fit at most nstep - 1 full steps in chunk,
            # remainder is at most stepsize.
            if self.rot_dict['remainder']:
                end = self.rot_dict['remainder'] + start

                # Again, no rotation should be performed.
                subchunks.append(dict(start=start, end=end, norot=True))

                start = end

            # Loop over full-sized rotation chunks.
            for step in range(nchunks-1):

                end = start + rot_chunk_size
                subchunks.append(dict(start=start, end=end))
                start = end

            # Fill last part and determine remainder.
            subchunks.append(dict(start=start, end=chunk['end']))
            self.rot_dict['remainder'] = rot_chunk_size - (chunk['end'] - start)

        # Subchunks get same cidx as parent chunk.
        if 'cidx' in chunk:
            cidx = chunk['cidx']
            for subchunk in subchunks:
                subchunk['cidx'] = cidx

        return subchunks

    def allocate_maps(self, nside=256):
        '''
        Allocate space in memory for binned output.

        Keyword arguments
        -----------------
        nside : int
            Nside of output (default : 256)
        '''

        self.vec = np.zeros((3, 12*nside**2), dtype=float)
        self.proj = np.zeros((6, 12*nside**2), dtype=float)
        self.nside_out = nside

    def init_detpair(self, alm, beam_a, beam_b=None,
                    beam_v=False, input_v=False, ground_alm=None,
                    **kwargs):
        '''
        Initialize the internal structure (the spinmaps)
        for a detector pair and all its ghosts.

        Arguments
        ---------
        alm : tuple
            Tuple containing (almI, almE, almB) as
            Healpix-formatted complex numpy arrays
            or (almI, almE, almB, almV) if the map
            contains V polarization
        beam_a : <detector.Beam> object
            The A detector.

        Keyword arguments
        -----------------
        beam_v : bool
                include the 4th blm component
        input_v : bool
                include the 4th alm component
        ground_alm : tuple
            Tuple containing (ground_almI, ground_almE, ground_almB) as
            Healpix-formatted complex numpy arrays
        beam_b : <detector.Beam> object
            The B detector. (default : None)
        kwargs : {spinmaps_opts}
            Extra kwargs are assumed input to `init_spinmaps()`

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

        # Ghosts with no blm attribute are given
        # identical Gaussian beams.
        if beam_a.ghosts:
            for gidx in range(beam_a.ghost_count):
                ghost_a = beam_a.ghosts[gidx]
                if gidx == 0:
                    if not hasattr(ghost_a, 'blm'):
                        ghost_a.gen_gaussian_blm()
                else:
                    if not hasattr(ghost_a, 'blm'):
                        ghost_a.reuse_blm(beam_a.ghosts[0])    
        if b_exists:
            if beam_b.ghosts:
                for gidx in range(beam_b.ghost_count):
                    ghost_b = beam_b.ghosts[gidx]
                    if gidx == 0:
                        if not hasattr(ghost_b, 'blm'):
                            ghost_b.reuse_blm(beam_a.ghosts[0])
                    else:
                        if not hasattr(ghost_b, 'blm'):
                            beam_b.ghosts[gidx].reuse_blm(
                                beam_a.ghosts[0])

        if ground_alm is not None:
            self.init_spinmaps(alm, beam_a, ground_alm = ground_alm, 
                            input_v=input_v, beam_v=beam_v, **kwargs)
        else:
            self.init_spinmaps(alm, beam_a, input_v=input_v,
                            beam_v=beam_v, **kwargs)

        # Free blm attributes.
        beam_a.delete_blm(del_ghosts_blm=True)
        if b_exists:
            beam_b.delete_blm(del_ghosts_blm=True)

    def scan_instrument_mpi(self, alm, ground_alm=None, verbose=1, binning=True,
            create_memmap=False, scatter=True, reuse_spinmaps=False,
            interp=False, save_tod=False, save_point=False, ctalk=0.,
            preview_pointing=False, filter_4fhwp=False, input_v=False,
            beam_v=False, filter_highpass=False, **kwargs):
        '''
        Loop over beam pairs, calculate boresight pointing
        in parallel, rotate or modulate instrument if
        needed, calculatesbeam-convolved tods, and,
        optionally, bin tods.

        Arguments
        ---------
        alm : tuple, None
            Tuple containing (almI, almE, almB) as Healpix-formatted
            complex numpy arrays. Can be None if preview_pointing is True.

        Keyword arguments
        -----------------
        ground_alm :  tuple, None
            Tuple containing (almI, almE, almB) as Healpix-formatted
            complex numpy arrays for a ground in  horizontal coordinates
            Used if one wants to add a ground template signal to the tod.
            Obviously made for ground-based scans only.
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
        save_tod : bool
            If set, save TOD of beams. (default : False)
        save_point : bool
            If set, save boresight quaternions, hwp_angle(s) [deg] and
            (per beam) pixel numbers and position angles [deg])
            (default : False)
        ctalk : float
            Fraction of cross-talk between detectors in pair.
            (default :  0)
        preview_pointing : bool
            If set, input alm is ignored, all SHT transforms are skipped,
            and no scanning is done (tod are zero). All pointing and binning
            steps are performed, so useful for quickly checking output hits-map
            and condition number (default : False)
        filter_4fhwp : bool
            Only use TOD modes modulated at 4 x the HWP frequency.
            Only allowed with spinning HWP. (default : False)
        input_v : bool
                include the 4th alm component if
                it exists
        beam_v : bool
                include the 4th blm component if
                it exists
        kwargs : {scan_opts, spinmaps_opts}
            Extra kwargs are assumed input to
            `implement_scan()` or `init_spinmaps()`.

        Notes
        -----
        save_tod and save_point options are meant for display and debugging
        purposes only.
        '''

        if alm is None and preview_pointing is False:
            raise TypeError(
                "input alm = None while `preview_pointing` != True")

        if preview_pointing and (save_tod is True or save_point is True):
            raise ValueError(
             "Cannot have `skip_scan` and `save_tod` or `save_point`")

        # Create a new spinmaps_opts dictionary and adds items max_spin and
        # nside_spin from init_spinmaps kwargs (if not None).
        max_spin = kwargs.pop('max_spin', None)
        nside_spin = kwargs.pop('nside_spin', None)
        spinmaps_opts = {}
        if max_spin:
            spinmaps_opts.update({'max_spin' : max_spin})
        if nside_spin:
            spinmaps_opts.update({'nside_spin' : nside_spin})

        if verbose and self.mpi_rank == 0:
            print('Scanning with {:d} detectors'.format(
                self.ndet))
            sys.stdout.flush()
        self.barrier() # Just to have summary print statement on top.

        # Init memmap on root.
        if create_memmap and not self.ext_point:
            if self.mpi_rank == 0:
                self.mmap = np.memmap('q_bore.npy', dtype=float,
                                      mode='w+', shape=(self.nsamp, 4),
                                      order='C')
            else:
                self.mmap = None

        # Scatter beams if needed, self.beams stays broadcasted.
        if scatter:
            beams = self.scatter_list(self.beams, root=0)
        else:
            beams = self.beams

        # Let every core loop over max number of beams per core
        # this makes sure that cores still participate in
        # calculating boresight quaternions.
        nmax = int(np.ceil(len(self.beams)/float(self.mpi_size)))

        for bidx in range(nmax): # Loop over beams.

            if bidx > 0:
                # reset instrument
                self.reset_instr_rot()
                self.reset_hwp_mod()
                self.reset_el_steps()

            do_ctalk = False # Starts False, might become True later.

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

            # Skip creating spinmaps when no beams or spinmaps
            # or already initialized with reuse_spinmaps.
            if not beam_a and not beam_b:
                pass
            elif preview_pointing is True:
                # Skip all SHT transforms.
                pass
            elif hasattr(self, 'spinmaps') and reuse_spinmaps:
                pass
            else:
                self.init_detpair(alm, beam_a, beam_b=beam_b,
                                input_v=input_v, beam_v=beam_v, 
                                ground_alm=ground_alm, **spinmaps_opts)

            if not hasattr(self, 'chunks'):
                # Assume no chunking is needed and use full mission length.
                self.partition_mission()

            # In case chunks have been added manually.
            self._init_data()

            for chunk in self.chunks:

                if verbose:
                    print(('[rank {:03d}]:\tWorking on chunk {:03}:'
                           ' samples {:d}-{:d}').format(self.mpi_rank,
                            chunk['cidx'], chunk['start'], chunk['end']))

                # Make the boresight move.
                scan_opts = kwargs.copy()
                scan_opts.update(chunk)
                if ground_alm is not None:
                    scan_opts.update({'ground':True})

                # Use precomputed pointing on subsequent passes.
                # Note, not used if memmap is not initialized.
                if bidx > 0:
                    scan_opts.update(dict(use_precomputed=True))

                self.implement_scan(**scan_opts)

                # If needed, allocate arrays for data.
                if bidx == 0:
                    self._allocate_hwp_data(save_point=save_point,
                                            **chunk)
                azel_point = True if (ground_alm is not None) else False
                if beam_a and not beam_a.dead:
                    self._allocate_detector_data(beam_a, save_tod=save_tod, 
                                                save_point=save_point, 
                                                azel_point=azel_point, **chunk)
                if beam_b and not beam_b.dead:
                    self._allocate_detector_data(beam_b, save_tod=save_tod,
                                                save_point=save_point, 
                                                azel_point=azel_point, **chunk)

                # If required, loop over boresight rotations.
                subchunks = self.subpart_chunk(chunk)
                # print(subchunks)
                for subchunk in subchunks:
                    if ground_alm is not None:
                        subchunk['ground'] = True
                    if verbose == 2:
                        print(('[rank {:03d}]:\t\t...'
                               ' samples {:d}-{:d}, norot={}').format(
                         self.mpi_rank, subchunk['start'], subchunk['end'],
                          subchunk.get('norot', False)))

                    # Rotate instrument and hwp if needed.
                    if not subchunk.get('norot', False):
                        self.rotate_instr()

                    self.rotate_hwp(**subchunk)

                    # Scan and bin.
                    if beam_a and not beam_a.dead:
                        self._scan_detector(beam_a, interp=interp,
                                            save_tod=save_tod,
                                            save_point=save_point,
                                            skip_scan=preview_pointing,
                                            **subchunk)

                        # Save memory by not copying if no pair.
                        if beam_b is None:
                            do_ctalk = False
                        else:
                            do_ctalk = ctalk * bool(beam_b) * (not beam_b.dead)
                            do_ctalk = bool(do_ctalk)

                        if do_ctalk:
                            tod_a = self.tod.copy()
                        elif binning:
                            self.bin_tod(beam_a, add_to_global=True, 
                                         filter_4fhwp=filter_4fhwp, 
                                         filter_highpass=filter_highpass,
                                         **subchunk)

                    if beam_b and not beam_b.dead:
                        self._scan_detector(beam_b, interp=interp,
                                            save_tod=save_tod,
                                            save_point=save_point,
                                            skip_scan=preview_pointing,
                                            **subchunk)

                        if do_ctalk:
                            tod_b = self.tod
                        elif binning:
                            self.bin_tod(beam_b, add_to_global=True, 
                                         filter_4fhwp=filter_4fhwp, 
                                         filter_highpass=filter_highpass,
                                         **subchunk)

                    if do_ctalk:
                        tools.cross_talk(tod_a, tod_b, ctalk=ctalk)

                        if save_tod:
                            # Update TOD with cross-talk leakage.
                            self._update_tod(beam_a, tod_a, **subchunk)
                            self._update_tod(beam_b, tod_b, **subchunk)

                        if binning:
                            self.bin_tod(beam_a, tod=tod_a, add_to_global=True,
                            filter_4fhwp=filter_4fhwp, 
                            filter_highpass=filter_highpass, **subchunk)
                            self.bin_tod(beam_b, tod=tod_b, add_to_global=True,
                            filter_highpass=filter_highpass,
                            filter_4fhwp=filter_4fhwp, **subchunk)

    def _chunk2idx(self, **kwargs):
        '''
        Return slice indices to chunk-sized array.

        Keyword arguments
        -----------------
        start : int
            Starting index.
        end : int
            Stopping index.
        cidx : int
            Chunk index.

        Returns
        -------
        qidx_start : int
            Starting index.
        qidx_end : int
            Stopping index.
        '''

        start = kwargs.get('start', False)
        end = kwargs.get('end', False)

        if start is False:
            raise ValueError('_chunk2idx called without start kwarg.')
        if end is False:
            raise ValueError('_chunk2idx called without end kwarg.')

        # Find the indices to the pointing and ctime arrays.
        if 'cidx' in kwargs:
            cidx = kwargs['cidx']
            qidx_start = start - self.chunks[cidx]['start']
            qidx_end = end - self.chunks[cidx]['start']
        else:
            qidx_start = 0
            qidx_end = end - start

        return qidx_start, qidx_end

    def _update_tod(self, beam, tod, **kwargs):
        '''
        Update the time-ordered data stored for given beam.
        Returns error when no data is present.

        Arguments
        ---------
        beam : <detector.Beam> object
            Main beam.
        tod : array-like
            Time-ordered data with size

        Keyword arguments
        -----------------
        kwargs : {chunk_opts}
        '''

        start, end = self._chunk2idx(**kwargs)
        cidx = kwargs.get('cidx') # Chunk index.

        self._data[str(cidx)][str(beam.idx)]['tod'][start:end] = tod

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
            # Step period larger or equal to chunksize.
            if step_dict['remainder'] >= chunk_size:

                arr[:] += step_dict['angle']
                step_dict['remainder'] -= chunk_size
            else:
                arr[:step_dict['remainder']] += step_dict['angle']

                step_dict['angle'] = next(step_gen)
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
            for step in range(nsteps-1):
                endidx = startidx + step_size

                step_dict['angle'] = next(step_gen)
                arr[startidx:endidx] += step_dict['angle']

                startidx = endidx

            # Fill last part and determine remainder.
            step_dict['angle'] = next(step_gen)
            arr[endidx:] += step_dict['angle']
            step_dict['remainder'] = step_size - arr[endidx:].size

            return arr


    def implement_scan(self, ra0=-10, dec0=-57.5, az_throw=90, scan_speed=1,
        az_prf='triangle', check_interval=600, el_min=45, cut_el_min=False,
        use_precomputed=False, ground=False, q_bore_func=None,
        q_bore_kwargs=None, ctime_func=None, ctime_kwargs=None, 
        use_l2_scan=False, **kwargs):

        '''
        Populates scanning quaternions.
        If no other options are selected, makes boresight scan back and forth in
        azimuth, starting centered at ra0, dec0, while keeping elevation
        constant.

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
            Scan speed in degrees per second
        az_prf : str
            Azimuthal profile. Current options:
                triangle : (default) triangle wave with total
                           width=az_throw.
                sawtooth : sawtooth wave with period given by
                           az_throw.
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
        ground : bool
            If True, will compute q_boreground, the pointing quaternion in
            az, el coordinates
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
        use_l2_scan : bool
            If true, implements a l2-like scan strategy
        start : int
            Start index
        end : int
            End index

        Notes
        -----
        Creates the following class attributes:
        ctime : ndarray
            Unix time array. Size = (end - start)
        q_bore : ndarray
            Boresight quaternion array. Shape = (end - start, 4)

        When using `external_pointing` is set, this method just loads
        the external pointing using the provided functions and ignores
        every other kwarg except `start` and `end`.
        '''

        start = kwargs.pop('start')
        end = kwargs.pop('end')

        # Complain when non-chunk kwargs are given.
        cidx = kwargs.pop('cidx', None)
        hwpang = kwargs.pop('hwpang', None)

        if kwargs:
            raise TypeError("implement_scan() got unexpected "
                "arguments '{}'".format(list(kwargs)))

        if self.ext_point and not use_l2_scan:
            # Use external pointing, so skip rest of function.
            self.ctime = ctime_func(start=start, end=end, cidx=cidx, 
                                                            **ctime_kwargs)
            if ground:
                self.q_bore, self.q_boreground = q_bore_func(start=start, 
                                                    end=end, cidx=cidx, 
                                                    ground=True, **q_bore_kwargs)
            else:
                self.q_bore = q_bore_func(start=start, end=end, cidx=cidx, 
                                                                **q_bore_kwargs)
            return

        elif use_l2_scan:

            print('Implementing L2 scan') # Lagrange point 2
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
        check_len = int(check_interval * self.fsamp) # min_el checks.

        nchecks = int(np.ceil(chunk_size / float(check_len)))
        p_len = check_len * nchecks # Longer than chunk for nicer slicing.

        ra0 = np.atleast_1d(ra0)
        dec0 = np.atleast_1d(dec0)
        npatches = dec0.shape[0]

        az0, el0, _ = self.radec2azel(ra0[0], dec0[0], 0,
            self.lon, self.lat, ctime[::check_len])

        flag0 = np.zeros(el0.size, dtype=bool)

        # Check and fix cases where boresight el < el_min.
        n = 1
        while np.any(el0 < el_min):

            if n < npatches:
                # Run check again with other ra0, dec0 options.
                azn, eln, _ = self.radec2azel(ra0[n], dec0[n], 0,
                                      self.lon, self.lat, ctime[::check_len])

                elidx = (el0<el_min)
                el0[elidx] = eln[elidx]
                az0[elidx] = azn[elidx]

            else:
                elidx = (el0<el_min)
                el0[elidx] = el_min

                if cut_el_min:
                    # Not scanning this elevation check chunk.
                    flag0[elidx] = True
                    warn('Cutting el min', RuntimeWarning)

                else:
                    # Scanning at fixed elevation.
                    warn('Keeping el0 at {:.1f} for part of scan'.format(el_min),
                        RuntimeWarning)
            n += 1

        # Scan boresight, note that it will slowly drift away from az0, el0.
        if scan_speed == 0:
            # Replace with small number to simulate staring.
            scan_speed = 1e-12

        if az_throw == 0:
            az = np.zeros(chunk_size)

        # NOTE, put these in seperate functions
        elif az_prf == 'triangle':

            scan_period = 2 * az_throw / float(scan_speed)

            az = np.arange(p_len, dtype=float)
            az *= (2 * np.pi / scan_period / float(self.fsamp))
            np.sin(az, out=az)
            np.arcsin(az, out=az)
            az *= (az_throw / np.pi)

        elif az_prf == 'sawtooth':

            # deg / s / (samp / s)
            deg_per_samp = scan_speed / float(self.fsamp)
            az = tools.sawtooth_wave(p_len, deg_per_samp, az_throw)
            az -= az_throw / 2.

        # Slightly complicated way to add az to az0
        # while avoiding expanding az0 to p_len.
        az = az.reshape(nchecks, check_len)
        az += az0[:, np.newaxis]
        az = az.ravel()
        az = az[:chunk_size] # Discard extra entries.

        el = np.zeros((nchecks, check_len), dtype=float)
        el += el0[:, np.newaxis]
        el = el.ravel()
        el = el[:chunk_size]

        # Do elevation stepping, if necessary.
        if self.step_dict.get('period', None):
            el = self.step_array(el, self.step_dict, self.el_step_gen)

        # Transform from horizontal frame to celestial, i.e. az, el -> ra, dec.
        if self.mpi:
            # Calculate boresight quaternion in parallel.
            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = divmod(chunk_size,
                                        self.mpi_size)
            sub_size += quot

            if remainder:
                # Give first ranks one extra quaternion.
                sub_size[:int(remainder)] += 1

            sub_start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            sub_end = sub_start + sub_size[self.mpi_rank]

            q_bore = np.empty(chunk_size * 4, dtype=float)
            
            if ground:
                q_boreground = np.empty(chunk_size * 4, dtype=float)
                caz = np.cos(np.radians(az[sub_start:sub_end])/2.)
                saz = np.sin(np.radians(az[sub_start:sub_end])/2.)
                cel = np.cos(np.pi/4.-np.radians(el[sub_start:sub_end])/2.)
                sel = np.sin(np.pi/4.-np.radians(el[sub_start:sub_end])/2.)
                q_boresubground = np.array([-saz*cel, caz*sel, 
                                             saz*sel, caz*cel]).swapaxes(0,1)
                q_boresubground = q_boresubground.ravel()
                ground_offsets = np.zeros(self.mpi_size)
                ground_offsets[1:] = np.cumsum(4*sub_size)[:-1] # start * 4
                self._comm.Allgatherv(q_boresubground,
                                     [q_boreground, 
                                      4*sub_size, 
                                      ground_offsets, 
                                      self._mpi_double])

                self.q_boreground = q_boreground.reshape(chunk_size, 4)

            # Calculate section of q_bore.
            q_boresub = self.azel2bore(az[sub_start:sub_end],
                                    el[sub_start:sub_end],
                                    None, None, self.lon, self.lat,
                                    ctime[sub_start:sub_end])
            q_boresub = q_boresub.ravel()

            sub_size *= 4 # For the flattened quat array.

            offsets = np.zeros(self.mpi_size)
            offsets[1:] = np.cumsum(sub_size)[:-1] # start * 4

            # Combine all sections on all ranks.
            self._comm.Allgatherv(q_boresub,
                            [q_bore, sub_size, offsets, self._mpi_double])
            self.q_bore = q_bore.reshape(chunk_size, 4)

        else:
            if ground:
                caz = np.cos(np.radians(az)/2.)
                saz = np.sin(np.radians(az)/2.)
                cel = np.cos(np.pi/4.-np.radians(el)/2.)
                sel = np.sin(np.pi/4.-np.radians(el)/2.)
                self.q_boreground = np.array([-saz*cel, caz*sel, 
                                               saz*sel, caz*cel]).swapaxes(0,1)

            self.q_bore = self.azel2bore(az, el, None, None, self.lon,
                                     self.lat, ctime)

        # Store boresight quat in memmap, if needed.
        if hasattr(self, 'mmap'):
            if self.mpi_rank == 0:
                self.mmap[start:end] = self.q_bore
            # wait for I/O
            if self.mpi:
                self._comm.barrier()

    def l2_ctime(self, **kwargs):
        '''
        A function to produce unix time (ctime) for a given chunk. Used when
        generating TODs for L2 (Lagrance point 2) scan strategies.

        Keyword arguments
        -----------------
        kwargs : {chunk}

        Returns
        -------
        ctime : ndarray
            Unix time array. Size = (end - start)
        '''

        start = kwargs.pop('start')
        end = kwargs.pop('end')

        ctime = np.arange(start, end, dtype=float)
        ctime /= float(self.fsamp)
        ctime += self.ctime0

        return ctime

    def l2_scan(self, theta_antisun=45., theta_boresight = 50.,
        freq_antisun = 192.348, freq_boresight = 0.314, sample_rate = 19.1,
        jitter_amp=0.0, **kwargs):

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
        start : int
            Start index
        end : int
            End index
        cidx : int

        Returns
        -------
        (az, el, lon, lat,) q_bore : array-like
            Depending on return_all

        Notes
        -----
        See Wallis et al., 2017, MNRAS, 466, 425.
        '''

        siad = 86400.           # seconds in a day
        today_julian = 1        # Just set to 1 by hand
        sample_rate = 19.1      # Hz
        ydays = 20              # duration of the mission (in days)
        nsiade = 256
        runtime_i = time.time()

        dt = 1 / float(self.fsamp)
        nsamp = self.ctime.size # ctime determines chunk size

        if self.mpi:

            # Calculate boresight quaternion in parallel.
            chunk_size = nsamp
            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = divmod(chunk_size, self.mpi_size)
            sub_size += quot

            if remainder:
                # Give first ranks one extra quaternion.
                sub_size[:int(remainder)] += 1

            sub_start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            sub_end = sub_start + sub_size[self.mpi_rank]

            q_bore = np.empty(chunk_size * 4, dtype=float)
            q_boresub = scanning.ctime2bore(self.ctime[sub_start:sub_end])
            q_boresub = q_boresub.ravel()

            sub_size *= 4 # for the flattened quat array

            offsets = np.zeros(self.mpi_size)
            offsets[1:] = np.cumsum(sub_size)[:-1] # start * 4

            # Combine all sections on all ranks.
            self._comm.Allgatherv(q_boresub,
                            [q_bore, sub_size, offsets, self._mpi_double])
            q_bore = q_bore.reshape(chunk_size, 4)

        else:

            q_bore = scanning.ctime2bore(self.ctime)

        self.flag = np.zeros(len(q_bore), dtype=bool)

        return q_bore

    def satellite_ctime(self, **kwargs):
        '''
        A function to produce unix time (ctime) for a given chunk

        Keyword arguments
        -----------------
        kwargs : {chunk}

        Returns
        -------
        ctime : ndarray
            Unix time array. Size = (end - start)

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
        A function to simulate satellite scanning strategy using a relatively
        coarse trick in qpoint. For a more advanced implementation, see l2_scan.

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
        cidx : int

        Returns
        -------
        (az, el, lon, lat,) q_bore : array-like
            Depending on return_all

        Notes
        -----
        See Wallis et al., 2017, MNRAS, 466, 425.
        '''

        deg_per_day = 360.9863
        dt = 1 / float(self.fsamp)
        nsamp = self.ctime.size # ctime determines chunk size.
        ndays = float(nsamp) / self.fsamp / (24 * 3600.)

        az = np.mod(np.arange(nsamp)*dt*360/float(beta_period), 360)

        if jitter_amp != 0.:
            jitter = jitter_amp * np.random.randn(int(nsamp))
            el = beta * np.ones_like(az) + jitter
        else:
            el = beta * np.ones_like(az)

        # Continue from position left chunk finished.
        if self.lon:
            lon_0 = self.lon
        else:
            # No position yet, so start at zero.
            lon_0 = 0.

        if self.lat:
            lat_0 = self.lat
        else:
            lat_0 = 0.

        # Anti sun at all times.
        lon = np.mod(-np.linspace(lon_0, ndays * deg_per_day + lon_0, nsamp),
                     360.)
        # The starting argument of the sin. Zero if lat_0 = 0.
        t_start = np.arcsin(lat_0 / float(alpha))

        lat = np.sin(np.linspace(
            t_start, 2*np.pi*dt*nsamp/float(alpha_period) + t_start,
            num=nsamp, endpoint=False))
        lat *= alpha

        # Store last lon, lat coord for start next chunk.
        self.lon = lon[-1]
        self.lat = lat[-1]

        if self.mpi:
            # Calculate boresight quaternion in parallel.
            chunk_size = nsamp
            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = divmod(chunk_size,
                                        self.mpi_size)
            sub_size += quot

            if remainder:
                # Give first ranks one extra quaternion.
                sub_size[:int(remainder)] += 1

            sub_start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            sub_end = sub_start + sub_size[self.mpi_rank]

            q_bore = np.empty(chunk_size * 4, dtype=float)

            # Calculate section of q_bore.
            q_boresub = self.azel2bore(az[sub_start:sub_end],
                                    el[sub_start:sub_end],
                                    None, None,
                                    lon[sub_start:sub_end],
                                    lat[sub_start:sub_end],
                                    self.ctime[sub_start:sub_end])
            q_boresub = q_boresub.ravel()

            sub_size *= 4 # For the flattened quat array.

            offsets = np.zeros(self.mpi_size)
            offsets[1:] = np.cumsum(sub_size)[:-1]
            
            # Combine all sections on all ranks.
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

    def balloon_night_qbore(self, el0=35., az0=0., scan_speed=30., ground=False, 
                          **kwargs):
        '''
        Populates scanning quaternions for a scan strategy with no 
        elevation change. Let boresight sweep at a given speed
        in azimuth.

        Keyword Arguments
        -----------------
        el0 : float
            Boresight elevation in degrees
            (default : 35.)
        az0 : float
            Boresight start azimuth in degrees
            (default : 0.)
        scan_speed : float
            Scan speed in degrees per second
            (default : 30.)
        ground : bool
            Whether to compute azel quaternions for a ground pickup scan
            (default : False)
        '''

        start = kwargs.pop('start')
        end = kwargs.pop('end')
        chunk_size = end-start
        # Complain when non-chunk kwargs are given.
        cidx = kwargs.pop('cidx', None)
        hwpang = kwargs.pop('hwpang', None)

        if kwargs:
            raise TypeError("strictly_az() got unexpected "
                "arguments '{}'".format(list(kwargs)))

        # Scan boresight, note that it will slowly drift away from az0, el0.
        if scan_speed == 0:
            # Replace with small number to simulate staring.
            scan_speed = 1e-12
        #Find which night the chunk starts in, to select scan direction 
        night_starts = np.argwhere(self.ctime[1:]-self.ctime[:-1]>2*self.fsamp)
        night_starts = night_starts.flatten()+1
        chunk_start_night = np.sum(start>night_starts)
        rot_sign = int(2*(chunk_start_night%2)-1)
        #Focus on chunk and find the scan breaks within it. 
        ctime = self.ctime[start:end]
        lon = self.lon[start:end]
        lat = self.lat[start:end]
        scan_breaks = np.argwhere(ctime[1:]-ctime[:-1]>2*self.fsamp).flatten()+1
        #Create azimuth vector that reverses direction at every break
        if (scan_breaks.size==0):
            az = np.arange(start, start+(end-start)*rot_sign, rot_sign, 
                           dtype=float)
        else:
            az = np.arange(0, rot_sign*scan_breaks[0], rot_sign, dtype=float)
            for i in range(scan_breaks.size-1):
                rot_sign *= -1
                interval = scan_breaks[i+1]-scan_breaks[i]
                interval_az = np.arange(az[-1], az[-1]+interval*rot_sign, 
                    rot_sign, dtype=float)
                az = np.concatenate((az, interval_az))

            last_leg = ctime.size-scan_breaks[-1]
            az = np.concatenate((az, 
                np.arange(az[-1], az[-1]-rot_sign*last_leg, -rot_sign)))

        az *= scan_speed
        az /= self.fsamp
        az += az0

        el = np.zeros(chunk_size, dtype=float)
        el += el0
        
        # Transform from horizontal frame to celestial, i.e. az, el -> ra, dec.
        if self.mpi:
            # Calculate boresight quaternion in parallel.
            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = divmod(chunk_size,
                                        self.mpi_size)
            sub_size += quot
            if remainder:

                # Give first ranks one extra quaternion.
                sub_size[:int(remainder)] += 1

            sub_start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            sub_end = sub_start + sub_size[self.mpi_rank]
            q_bore = np.empty(chunk_size * 4, dtype=float)

            if ground:
                q_boreground = np.empty(chunk_size * 4, dtype=float)
                caz = np.cos(np.radians(az[sub_start:sub_end])/2.)
                saz = np.sin(np.radians(az[sub_start:sub_end])/2.)
                cel = np.cos(np.pi/4.-np.radians(el[sub_start:sub_end])/2.)
                sel = np.sin(np.pi/4.-np.radians(el[sub_start:sub_end])/2.)
                q_boresubground = np.array([-saz*cel, caz*sel, 
                                             saz*sel, caz*cel]).swapaxes(0,1)
                q_boresubground = q_boresubground.ravel()
                ground_offsets = np.zeros(self.mpi_size)
                ground_offsets[1:] = np.cumsum(4*sub_size)[:-1] # start * 4
                self._comm.Allgatherv(q_boresubground,
                                [q_boreground, 4*sub_size, ground_offsets, 
                                 self._mpi_double])
                q_boreground = q_boreground.reshape(chunk_size, 4)

            # Calculate section of q_bore.
            q_boresub = self.azel2bore(az[sub_start:sub_end],
                                       el[sub_start:sub_end],
                                       None, None, lon[sub_start:sub_end], 
                                       lat[sub_start:sub_end],
                                       ctime[sub_start:sub_end])
            q_boresub = q_boresub.ravel()

            sub_size *= 4 # For the flattened quat array.

            offsets = np.zeros(self.mpi_size)
            offsets[1:] = np.cumsum(sub_size)[:-1] # start * 4
            # Combine all sections on all ranks.
            self._comm.Allgatherv(q_boresub,
                            [q_bore, sub_size, offsets, self._mpi_double])
            q_bore = q_bore.reshape(chunk_size, 4)            

        else:
            if ground:
                caz = np.cos(np.radians(az)/2.)
                saz = np.sin(np.radians(az)/2.)
                cel = np.cos(np.pi/4.-np.radians(el)/2.)
                sel = np.sin(np.pi/4.-np.radians(el)/2.)
                q_boreground = np.array([-saz*cel, caz*sel, 
                                          saz*sel, caz*cel]).swapaxes(0,1)
            q_bore = self.azel2bore(az, el, None, None, lon, lat, ctime)
        self.flag = np.zeros_like(az, dtype=bool)
        if ground:
            return q_bore, q_boreground
        else:
            return q_bore


    def parse_schedule_file(self, schedule_file=None):
        '''
        Read a text file with standard input and generate arrays that can be
        circulated between

        Keyword arguments
        -----------------
        schedule_file : file path (string)

        Returns
        -------
        nces : ndarray (int)
            A monotonically increasing array of integers corresponding
            to scan number
        az0s : ndarray (float)
            Lower limit on azimuthal range
        az1s : ndarray (float)
            Upper limit on azimuthal range
        els : ndarray (float)
            Boresight elevation value for the scan
        t0s : ndarray (float)
            Start times for each scan (unix time)
        t1s : ndarray (float)
            End times for each scan (unix time)
        '''

        def mjd2ctime(mjd):

            return (mjd - 40587.) * 86400.0

        data = np.loadtxt(schedule_file)
        t0s  = mjd2ctime(data[:, 0])
        t1s  = mjd2ctime(data[:, 1])
        az0s = data[:, 2]
        az1s = data[:, 3]
        els  = data[:, 4]

        N = len(az0s)

        return np.arange(N), az0s, az1s, els, t0s, t1s


    def partition_schedule_file(self, filename='', chunksize=None):
        '''
        Divide up the mission in equal-sized chunks
        of nsample / chunksize (final chunk can be
        smaller).

        Keyword arguments
        ---------
        chunksize : interp
            Chunk size in samples. If left None, use
            full mission length (default : None)

        Returns
        -------
        chunks : list of dicts
            Dictionary with start, end indices and index of each
            chunk.
        '''

        chunks = []
        ctime_starts = []
        chunknum = 0
        samplenum = 0
        done_chunking = False
        nsamp = self.nsamp

        nces, az0s, az1s, els, t0s, t1s = self.parse_schedule_file(filename)
        az0i, az1i, eli, t0i, t1i = [], [], [], [], []

        for i, (t0, t1) in enumerate(zip(t0s, t1s)):
            print('CES {}/{}'.format(i, len(t0s)))
            print('Chunksize = {}'.format(chunksize))

            if done_chunking:
                break

            nsamp_full_ces = (t1 - t0) * self.fsamp
            if not chunksize or chunksize >= nsamp_full_ces:
                print('Chunk larger than nsamp_full_ces')
                chunksize2use = int(nsamp_full_ces)
                nchunks = 1
            else:
                print('nsamp_full_ces | chunksize')
                print('{} | {}'.format(nsamp_full_ces, chunksize))
                chunksize2use = int(chunksize)
                nchunks = int(np.ceil(nsamp_full_ces / float(chunksize2use)))

            start = samplenum
            tstart = t0
            for cidx in range(nchunks):
                end = start + chunksize2use
                if cidx == nchunks-1:
                    end = samplenum+nsamp_full_ces
                if end >= nsamp:
                    end = nsamp
                if end!=start:
                    chunks.append(dict(start=int(start), end=int(end),
                        cidx=int(chunknum + cidx)))

                    az0i.append(az0s[i])
                    az1i.append(az1s[i])
                    eli.append(els[i])
                    t0i.append(t0s[i])
                    t1i.append(t1s[i])

                    ctime_starts.append(tstart)

                    start += chunksize2use
                    tstart += float(chunksize2use) / self.fsamp

                if start >= nsamp:
                    done_chunking=True
                    break

            if done_chunking:
                # chunknum += nchunks
                chunknum += cidx+1
            else:
                # chunknum += cidx+1
                chunknum += nchunks

            samplenum = end

        self.chunks = chunks
        self.ctime_starts = ctime_starts

        # Assigning attributes that will be used by schedule_ctime and
        # schedule_scan.
        self.az0s = az0i
        self.az1s = az1i
        self.els = eli
        self.t0s = t0i
        self.t1s = t1i

        return chunks

    def schedule_ctime(self, **kwargs):
        '''
        A function to produce unix time (ctime) for a given chunk.
        Used as part of a simulation pipeline that reads from a schedule file.

        Keyword arguments
        -----------------
        kwargs : {chunk}

        Returns
        -------
        ctime : ndarray
            Unix time array. Size = (end - start)

        '''

        if self.ctime_starts is None:
            raise ValueError('Have to partition schedule file first')

        start = kwargs.pop('start')
        end = kwargs.pop('end')
        cidx = kwargs.pop('cidx')

        ctime = self.ctime_starts[cidx] + \
            np.arange(end-start, dtype=float) / float(self.fsamp)

        return ctime

    def schedule_scan(self, scan_speed=2.5,
        return_all=False, ground=False, az_prf='triangle', **kwargs):
        '''

        Reads in a schedule file following a certain format and procuces
        boresight quaternions.

        Keyword arguments
        -----------------
        scan_speed : float [deg/s]
            Scan speed in deg/s
        return_all : bool
            If set, return az, el, lat, lon as well as q_bore
        ground : bool
            If True, return q_boreground
        az_prf : str
            Type of scan profile, currently only admits 'triangle'

        Returns
        -------
        (az, el, lon, lat,) q_bore : array-like
            Depending on return_all

        '''

        nsamp = self.ctime.size # ctime determines chunk size.

        # The idx to the last CES t0 before ctime[0].
        idx_ces = kwargs.pop('cidx')

        t0_ces = self.t0s[idx_ces]
        el0 = self.els[idx_ces]
        dt = self.ctime[0] - t0_ces
        az0 = self.az0s[idx_ces]
        az1 = self.az1s[idx_ces]
        flag0 = np.zeros(el0.size, dtype=bool)

        if az0 > az1:
            az_throw = (360 - az0) + az1
        else:
            az_throw = az1 - az0

        assert az_throw > 0., 'az_throw should be positive'

        # Scan boresight, note that it will slowly drift away from az0, el0.
        if az_throw == 0:
            az = np.zeros(chunk_size)

        # NOTE, put these in seperate functions.
        elif az_prf == 'triangle':

            scan_half_period = az_throw / float(scan_speed)
            nsamp_per_scan = int(scan_half_period * self.fsamp)

            assert nsamp_per_scan > 0, 'number of samples per scan should be positive'

            nsamp_per_period = 2*nsamp_per_scan-1

            phase = np.remainder(dt / float(self.fsamp), float(nsamp_per_period))
            phase_idx = np.ceil(phase * nsamp_per_period)

            # Number of triangle scane in this chunk.
            nmult = np.ceil(float(nsamp) / nsamp_per_period)

            # One full triangle scan.
            az_single = np.hstack((np.linspace(az0, az1, nsamp_per_scan - 1, endpoint=False),
                np.linspace(az1, az0, nsamp_per_scan, dtype=float)))

            az_full = np.roll(np.tile(az_single, int(nmult)), int(phase_idx))
            az = az_full[:nsamp]

        else:
            raise ValueError('Other scan options currently not implemented')

        el = el0 * np.ones_like(az)

        if self.mpi:
            # Calculate boresight quaternion in parallel.
            chunk_size = nsamp
            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = divmod(chunk_size,
                                        self.mpi_size)
            sub_size += quot

            if remainder:
                # give first ranks one extra quaternion
                sub_size[:int(remainder)] += 1

            sub_start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            sub_end = sub_start + sub_size[self.mpi_rank]

            q_bore = np.empty(chunk_size * 4, dtype=float)

            if ground:
                q_boreground = np.empty(chunk_size * 4, dtype=float)
                caz = np.cos(np.radians(az[sub_start:sub_end])/2.)
                saz = np.sin(np.radians(az[sub_start:sub_end])/2.)
                cel = np.cos(np.pi/4.-np.radians(el[sub_start:sub_end])/2.)
                sel = np.sin(np.pi/4.-np.radians(el[sub_start:sub_end])/2.)
                q_boresubground = np.array([-saz*cel, caz*sel, saz*sel, caz*cel]).swapaxes(0,1)
                q_boresubground = q_boresubground.ravel()
                ground_offsets = np.zeros(self.mpi_size)
                ground_offsets[1:] = np.cumsum(4*sub_size)[:-1]
                self._comm.Allgatherv(q_boresubground,
                                [q_boreground, 4*sub_size, ground_offsets, self._mpi_double])
                q_boreground = q_boreground.reshape(chunk_size, 4)


            # Calculate section of q_bore.
            q_boresub = self.azel2bore(az[sub_start:sub_end],
                                    el[sub_start:sub_end],
                                    None, None,
                                    self.lon,
                                    self.lat,
                                    self.ctime[sub_start:sub_end])
            q_boresub = q_boresub.ravel()

            sub_size *= 4 # for the flattened quat array

            offsets = np.zeros(self.mpi_size)
            offsets[1:] = np.cumsum(sub_size)[:-1] # start * 4

            # Combine all sections on all ranks.
            self._comm.Allgatherv(q_boresub,
                            [q_bore, sub_size, offsets, self._mpi_double])
            q_bore = q_bore.reshape(chunk_size, 4)

        else:
            if ground:
                caz = np.cos(np.radians(az)/2.)
                saz = np.sin(np.radians(az)/2.)
                cel = np.cos(np.pi/4.-np.radians(el)/2.)
                sel = np.sin(np.pi/4.-np.radians(el)/2.)
                q_boreground = np.array([-saz*cel, caz*sel, 
                                          saz*sel, caz*cel]).swapaxes(0,1)
            q_bore = self.azel2bore(az, el, None, None, self.lon,
                                         self.lat, self.ctime)

        self.flag = np.zeros_like(az, dtype=bool)

        if return_all and ground:
            return az, el, self.lon, self.lat, q_bore, q_boreground
        elif return_all and not ground:
            return az, el, self.lon, self.lat, q_bore
        else:
            if ground:
                return q_bore, q_boreground
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
        -----------------
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

        # If HWP does not move, just return current angle.
        if not self.hwp_dict['freq'] or not self.hwp_dict['mode']:

            if kwargs.get('hwpang'):
                self.hwp_ang = np.radians(kwargs.get('hwpang'))
            else:
                self.hwp_ang = np.radians(self.hwp_dict['angle'])

            return

        # If needed, compute hwp angle array.
        freq = self.hwp_dict['freq'] # cycles per sec for cont. rot hwp
        start_ang = np.radians(self.hwp_dict['angle'])

        if self.hwp_dict['mode'] == 'continuous':

            self.hwp_ang = np.linspace(start_ang,
                   start_ang + 2 * np.pi * chunk_size / (self.fsamp / float(freq)),
                   num=chunk_size, endpoint=False, dtype=float) # In radians (w = 2 pi freq)

            # Update mod 2pi start angle for next chunk.
            self.hwp_dict['angle'] = np.degrees(np.mod(self.hwp_ang[-1], 2*np.pi))
            return

        if self.hwp_dict['mode'] == 'stepped':

            hwp_ang = np.zeros(chunk_size, dtype=float)
            hwp_ang = self.step_array(hwp_ang, self.hwp_dict, self.hwp_angle_gen)
            np.radians(hwp_ang, out=hwp_ang)

            # Update mod 2pi start angle for next chunk.
            self.hwp_dict['angle'] = np.degrees(np.mod(hwp_ang[-1], 2*np.pi))
            self.hwp_ang = hwp_ang

    def _init_data(self):
        '''
        Make sure data has a key for each chunk.
        '''

        for chunk in self.chunks:

            cidx = chunk['cidx']
            if str(cidx) not in self._data:
                self._data[str(cidx)] = {}

    def _allocate_hwp_data(self, save_point=False, **chunk):
        '''
        Allocate inernal arrays for saving HWP angles for given
        chunk.

        Keyword arguments
        -----------------
        save_point : bool
            If set, allocate hwp_angles [deg], pixel
            numbers and position angles [deg]) (default : False)
        start : int
            Start index.
        end : int
            End index.
        cidx : int
            Chunk index.
        '''

        if save_point:

            start = chunk.get('start')
            end = chunk.get('end')
            cidx = chunk.get('cidx')

            tod_size = end - start  # Size in samples.

            hwp_ang = np.zeros(tod_size, dtype=float)

            self._data[str(cidx)]['hwp_ang'] = hwp_ang

    def _allocate_detector_data(self, beam, save_tod=False, save_point=False, 
                                    azel_point=False, **chunk):
        '''
        Allocate internal arrays for saving data for given chunk
        and beam.

        Arguments
        ---------
        beam : <detector.Beam> object
            The main beam of the detector.

        Keyword arguments
        -----------------
        save_tod : bool
            If set, allocate TOD. (default : False)
        save_point : bool
            If set, allocate hwp_angles [deg], pixel
            numbers and position angles [deg] (default : False)
        azel_point : bool
            If set, and save_point is True, allocate pixel numbers and 
            position angles [deg] in the horizontal frame (default : False)
        start : int
            Start index.
        end : int
            End index.
        cidx : int
            Chunk index.
        '''

        start = chunk.get('start')
        end = chunk.get('end')
        cidx = chunk.get('cidx')

        tod_size = end - start  # Size in samples.

        self._data[str(cidx)][str(beam.idx)] = {}

        if save_tod and not save_point:
            tod = np.zeros(tod_size, dtype=float)

            self._data[str(cidx)][str(beam.idx)]['tod'] = tod

        elif not save_tod and save_point:
            pix = np.zeros(tod_size, dtype=int)
            pa = np.zeros(tod_size, dtype=float)

            self._data[str(cidx)][str(beam.idx)]['pix'] = pix
            self._data[str(cidx)][str(beam.idx)]['pa'] = pa

        elif save_tod and save_point:
            tod = np.zeros(tod_size, dtype=float)
            pix = np.zeros(tod_size, dtype=int)
            pa = np.zeros(tod_size, dtype=float)

            self._data[str(cidx)][str(beam.idx)]['tod'] = tod
            self._data[str(cidx)][str(beam.idx)]['pix'] = pix
            self._data[str(cidx)][str(beam.idx)]['pa'] = pa

        if save_point and azel_point:
            azel_pix = np.zeros(tod_size, dtype=int)
            azel_pa = np.zeros(tod_size, dtype=float)
            self._data[str(cidx)][str(beam.idx)]['azel_pix'] = azel_pix
            self._data[str(cidx)][str(beam.idx)]['azel_pa'] = azel_pa



    def _scan_detector(self, beam, interp=False, save_tod=False,
        save_point=False, skip_scan=False, **kwargs):
        '''
        Convenience function that adds ghost(s) TOD to main beam TOD
        and saves TOD and pointing if needed.

        Arguments
        ---------
        beam : <detector.Beam> object
            The main beam of the detector.

        Keyword arguments
        -----------------
        interp : bool
            If set, use bi-linear interpolation to obtain TOD.
            (default : False)
        save_tod : bool
            If set, save TOD of beams. (default : False)
        save_point : bool
            If set, save boresight quaternions, hwp_angle(s) [deg] and
            (per beam) pixel numbers and position angles [deg])
            (default : False)
        skip_scan : bool
            If set, tod attribute is not populated by scanning
            over spinmaps but filled with zeros. Effectively
            skip whole method (default : False).
        kwargs : {chunk}

        Notes
        -----
        Returns ValueError when `beam` is a ghost.
        '''
        if beam.ghost:
            raise ValueError('_scan_detector() called with ghost.')

        if skip_scan and (save_tod is True or save_point is True):
            raise ValueError(
             "Cannot have `skip_scan` and `save_tod` or `save_point`")

        n = 0 # Count alive ghosts.
        if any(beam.ghosts):
            for gidx, ghost in enumerate(beam.ghosts):

                if ghost.dead:
                    continue

                # First alive ghost must start TOD.
                add_to_tod = False if n == 0 else True
                n += 1

                self.scan(ghost,
                          add_to_tod=add_to_tod,
                          interp=interp,
                          skip_scan=skip_scan,
                          **kwargs)

        tod_exists = True if n > 0 else False
        # Find indices to slice of chunk.
        start = kwargs.get('start')
        end = kwargs.get('end')
        cidx = kwargs.get('cidx')

        q_start, q_end = self._chunk2idx(start=start, end=end, cidx=cidx)

        #Add ground tod.
        if 'ground' in kwargs:
            azel_ret = self.scan(beam, interp=interp, add_to_tod=tod_exists,
                return_point=save_point, **kwargs)
            if save_point: 
                azel_pix, azel_nside, azel_pa, azel_hwp = azel_ret
                self._data[str(cidx)][str(beam.idx)]['azel_pix'][q_start:q_end] = azel_pix
                self._data[str(cidx)][str(beam.idx)]['azel_pa'][q_start:q_end] = azel_pa

            kwargs.pop('ground')
            tod_exists = True

        # Scan with main beam. Add to ghost TOD if present.
        ret = self.scan(beam, interp=interp, return_tod=save_tod,
                        add_to_tod=tod_exists,
                        return_point=save_point,
                        skip_scan=skip_scan,
                        **kwargs)


        if save_tod and not save_point:
            # Only TOD is returned.
            tod = ret
            self._data[str(cidx)][str(beam.idx)]['tod'][q_start:q_end] = tod

        elif not save_tod and save_point:
            ret_pix, ret_nside, ret_pa, ret_hwp = ret
            self._data[str(cidx)][str(beam.idx)]['pix'][q_start:q_end] = ret_pix
            self._data[str(cidx)][str(beam.idx)]['pa'][q_start:q_end] = ret_pa
            self._data[str(cidx)]['hwp_ang'][q_start:q_end] = ret_hwp

        elif save_tod and save_point:
            tod, ret_pix, ret_nside, ret_pa, ret_hwp = ret
            self._data[str(cidx)][str(beam.idx)]['tod'][q_start:q_end] = tod
            self._data[str(cidx)][str(beam.idx)]['pix'][q_start:q_end] = ret_pix
            self._data[str(cidx)][str(beam.idx)]['pa'][q_start:q_end] = ret_pa
            self._data[str(cidx)]['hwp_ang'][q_start:q_end] = ret_hwp

    def data(self, chunk, beam=None, data_type='tod'):
        '''
        Return calculated data.

        Arguments
        ---------
        chunk : dict
            Chunk (see `partition mission`) for which to
            provide data.

        Keyword arguments
        -----------------
        beam : <detector.Beam> object, None
            Beam for which to provide data. Can be None for
            HWP angle data. (default : None)
        data_type : str
            Type of data to be returned. Choices are "tod",
            "pix", "pa", "hwp_ang". (default : tod)

        Returns
        -------
        data : array-like
            Time-ordered data of specified type.
        '''

        if data_type == 'hwp_ang':
            return self._data[str(chunk['cidx'])]['hwp_ang']
        else:
            return self._data[str(chunk['cidx'])][str(beam.idx)][data_type]

    def scan(self, beam, add_to_tod=False, interp=False,
             return_tod=False, return_point=False, skip_scan=False,
             **kwargs):
        '''
        Update boresight pointing with detector offset, and
        use it to bin spinmaps into a tod.

        Arguments
        ---------
        beam : <detector.Beam> object

        Kewword arguments
        ---------
        add_to_tod : bool
            Add resulting TOD to existing tod attribute and do not
            internally store the detector offset pointing.
            (default: False)
        interp : bool
            If set, use bi-linear interpolation to obtain TOD.
            (default : False)
        return_tod : bool
            If set, return TOD. (default : False)
        return_point : bool
            If set, return pix, nside, pa, hwp_ang (HEALPix
            pixel numbers, nside, position angle [deg] and HWP
            angle(s) [deg]) (default : False)
        start : int
            Start index
        end : int
            End index

        Returns
        -------
        tod : array-like
            If return_tod=True: TOD of length: end - start.
        pix : array-like
            If return_point=True: healpy (ring) pixel numbers
            of length: end - start.
        nside : int
            Nside parameter corresponding to pixel numbers.
        pa : array-like
            If return_point=True: position angle [deg] of
            length: end - start.
        hwp_ang : array-like, float
            If return_point=True: HWP angle(s) [deg] of
            length: end - start for continuously spinning HWP
            otherwise single angle.
        skip_scan : bool
            If set, tod attribute is not populated by scanning
            over spinmaps but filled with zeros. Effectively
            skip whole method (default : False).

        Notes
        -----
        Creates following class attributes:

        tod : ndarray
           Array of time-ordered data. Size = (end - start)
        pix : ndarray
            Array of HEALPix pixel numbers. Size = (end - start).
            Only if interp = False and skip_scan = False.

        Modifies the following attributes of the beam:

        q_off : ndarray
            Unit quaternion with detector offset (az, el)
            and, possibly, instrument rotation.
        '''

        if beam.dead:
            raise ValueError('scan() called with dead beam: {}'.format(
                beam.name))

        if skip_scan and (return_tod is True or return_point is True):
            raise ValueError(
             "Cannot have `skip_scan` and `return_tod` or `return_point`")

        start = kwargs.get('start')
        end = kwargs.get('end')
        if start is None or end is None:
            raise ValueError("Scan() called without start and end.")
        tod_size = end - start  # size in samples

        # Offset quaternion defined without polang.
        # polang is applied at the beam level later.
        az_off = beam.az_truth #True az for scanning
        el_off = beam.el_truth #True el for scanning
        polang = beam.polang_truth # True polang for scanning.
        q_off_true = self.det_offset(az_off, el_off, 0)
        q_off = self.det_offset(beam.az, beam.el, 0)

        # Rotate offset given rot_dict. Rotate the centroid
        # around the boresight. q_bore * q_rot * q_off.
        ang = np.radians(self.rot_dict['angle'])
        q_rot = np.asarray([np.cos(ang/2.), 0., 0., np.sin(ang/2.)])
        q_off_true = tools.quat_left_mult(q_rot, q_off_true)
        q_off = tools.quat_left_mult(q_rot, q_off)

        # Expose estimated pointing offset for mapmaking. Not for ghosts.
        if not beam.ghost:
            beam.q_off = q_off

        if skip_scan:
            # Allocate a fake tod.
            tod = np.zeros(tod_size, dtype=float)
            if add_to_tod and hasattr(self, 'tod'):
                self.tod += tod
            else:
                self.tod = tod

            # Skip all other scanning.
            return

        beam_type = 'main_beam' if not beam.ghost else 'ghosts'
        if 'ground' in kwargs:
            beam_type = 'ground'
        if beam_type == 'ghosts':
            spinmaps = self.spinmaps[beam_type][beam.ghost_idx]
        else:
            spinmaps = self.spinmaps[beam_type]
        #This way, you can get the "main_beam" or the "ground" spin map

        # Do some sanity checks on spinmaps.
        for conv_type in spinmaps:
            maps = spinmaps[conv_type]['maps']
            s_vals = spinmaps[conv_type]['s_vals']
            if maps.shape[0] != s_vals.size:
                raise ValueError(
                    'Shape maps {} and s_vals {} do not match for {} {}.'.
                    format(maps.shape[0], s_vals.size, beam_type, conv_type))
            nside_spin = hp.npix2nside(maps.shape[1])
            try:
                if nside_spin != nside_spin_previous:
                    raise ValueError('spinmaps have different nside')
            except NameError:
                pass
            nside_spin_previous = nside_spin

        # Find the indices to the pointing and ctime arrays.
        qidx_start, qidx_end = self._chunk2idx(**kwargs)

        if interp:
            ra = np.empty(tod_size, dtype=np.float64)
            dec = np.empty(tod_size, dtype=np.float64)
            pa = np.empty(tod_size, dtype=np.float64)
            #If ground, point the detector in horizontal coordinates.
            if 'ground' in kwargs:
                self.bore2radec(q_off_true,
                            self.ctime[qidx_start:qidx_end],
                            self.q_boreground[qidx_start:qidx_end],
                            q_hwp=None, sindec=False, return_pa=True,
                            ra=ra, dec=dec, pa=pa)

            else:
                self.bore2radec(q_off_true,
                            self.ctime[qidx_start:qidx_end],
                            self.q_bore[qidx_start:qidx_end],
                            q_hwp=None, sindec=False, return_pa=True,
                            ra=ra, dec=dec, pa=pa)

            # Convert qpoint output to input healpy (in-place),
            # needed for interpolation later.
            tools.radec2colatlong(ra, dec)

        else:
            # In no interpolation is required, we can go straight
            # from quaternion to pix and pa.
            if 'ground' in kwargs:
                pix, pa = self.bore2pix(q_off_true,
                            self.ctime[qidx_start:qidx_end],
                            self.q_boreground[qidx_start:qidx_end],
                            q_hwp=None, nside=nside_spin, return_pa=True)
            else:
                pix, pa = self.bore2pix(q_off_true,
                            self.ctime[qidx_start:qidx_end],
                            self.q_bore[qidx_start:qidx_end],
                            q_hwp=None, nside=nside_spin, return_pa=True)

            # Expose pixel indices for test centroid.
            self.pix = pix

        np.radians(pa, out=pa)
        # Conversion healpix -> IAU convention.
        pa *= -1
        pa += np.pi

        # NOTE for now we should be in this block even if we are using
        # a hwp_mueller matrix.
        tod_c = np.zeros(tod_size, dtype=np.complex128)

        if interp:
            pix = (ra, dec)

        # Check for HWP angle array.
        if not hasattr(self, 'hwp_ang'):
            hwp_ang = 0
        else:
            hwp_ang = self.hwp_ang

        # Find out if old or new HWP behaviour is desired.
        if set(spinmaps.keys()) == set(['s0a0', 's2a4']):
            # Old behaviour.
            self._scan_modulate_pa(tod_c, pix, pa,
                                   spinmaps['s2a4']['maps'],
                                   spinmaps['s2a4']['s_vals'],
                                   reality=False, interp=interp)


            expm2 = np.exp(1j * (4 * hwp_ang + 2 * np.radians(polang)))

            tod_c *= expm2
            tod = np.real(tod_c) # Note, shares memory with tod_c.

            # Add unpolarized tod.
            self._scan_modulate_pa(tod, pix, pa,
                                   spinmaps['s0a0']['maps'],
                                   spinmaps['s0a0']['s_vals'],
                                   reality=True, interp=interp)

        else:
            # New behaviour.
            self._scan_modulate_pa(tod_c, pix, pa,
                                   spinmaps['s2a4']['maps'],
                                   spinmaps['s2a4']['s_vals'],
                                   reality=False, interp=interp)

            # Modulate by HWP angle and polarization angle.
            expm2 = np.exp(1j * (4 * hwp_ang + 2 * np.radians(polang)))
            tod_c *= expm2

            tod = np.real(tod_c).copy()

            tod_c *= 0.
            self._scan_modulate_pa(tod_c, pix, pa,
                                   spinmaps['s2a2']['maps'],
                                   spinmaps['s2a2']['s_vals'],
                                   reality=False, interp=interp)

            expm2 = np.exp(1j * (2 * hwp_ang))
            tod_c *= expm2
            tod += np.real(tod_c)

            tod_c *= 0
            self._scan_modulate_pa(tod_c, pix, pa,
                                   spinmaps['s2a0']['maps'],
                                   spinmaps['s2a0']['s_vals'],
                                   reality=False, interp=interp)

            tod += np.real(tod_c)
            del tod_c

            # Add unpolarized tod.
            tod_tmp = np.zeros_like(tod)

            # The s0a2 case is a bit complicated. The spinmaps in this case 
            # correspond to sqrt2 sum_lm -2bls+2 Cpi alm sYlm for positive
            # and negative s. However, we need sum_lm bls^Re alm sYlm cos(2 alpha) and 
            # sum_lm bls^Im alm sYlm sin(2 alpha) for positive s. Here
            # bls^Re = (-2bls+2 Cpi + 2bls-2 Cp*i)/sqrt2 and bls^Im = 
            # i (-2bls+2 Cpi - 2bls-2 Cp*i)/sqrt2. 
            # For a given |s| these are given by 0.5 (map_|s| + (map_-|s|)^*) cos(2 alpha)
            # and 0.5 i (map_|s| - (map_-|s|)^*) sin(2 alpha). 
            # To construct these we use that (map_|s|)^* = map_-|s|.

            s_vals = spinmaps['s0a2']['s_vals']
            idx_nonneg = np.where(s_vals >= 0)[0]
            npix = spinmaps['s0a2']['maps'][0].size
            maps_re = np.zeros((idx_nonneg.size, npix), dtype=np.complex128)
            maps_im = maps_re.copy()

            for oidx, idx in enumerate(idx_nonneg):

                spin = s_vals[idx]
                if spin != 0:
                    idx_neg = np.where(s_vals == -spin)[0][0]
                else:
                    idx_neg = idx
                map_pos = spinmaps['s0a2']['maps'][idx].copy()
                map_neg = spinmaps['s0a2']['maps'][idx_neg].copy()

                map_pos *= np.exp(1j * 2 * np.radians(polang))
                map_neg *= np.exp(-1j * 2 * np.radians(polang))
                
                map_re = 0.5 * (map_pos + np.conj(map_neg))
                map_im = 0.5j * (map_pos - np.conj(map_neg))
                
                maps_re[oidx] = map_re
                maps_im[oidx] = map_im

            self._scan_modulate_pa(tod_tmp, pix, pa, maps_re, s_vals[idx_nonneg],
                                   reality=True, interp=interp)
            tod_tmp *= np.cos(2 * hwp_ang)
            tod += tod_tmp

            tod_tmp *= 0
            self._scan_modulate_pa(tod_tmp, pix, pa, maps_im, s_vals[idx_nonneg],
                                   reality=True, interp=interp)
            tod_tmp *= np.sin(2 * hwp_ang)
            tod += tod_tmp
            del tod_tmp
                                    
            # Normal unpolarized part.
            self._scan_modulate_pa(tod, pix, pa,
                                   spinmaps['s0a0']['maps'],
                                   spinmaps['s0a0']['s_vals'],
                                   reality=True, interp=interp)

        if add_to_tod and hasattr(self, 'tod'):
            self.tod += tod
        else:
            self.tod = tod

        # Handle returned values.
        if return_tod:
            ret_tod = self.tod.copy()
        if return_point:
            if interp:
                # Note, dec and ra are already theta, phi.
                ret_pix = hp.ang2pix(nside_spin, dec, ra, nest=False)
            else:
                ret_pix = pix.copy()

            ret_nside = nside_spin
            ret_pa = np.degrees(pa)
            ret_hwp = np.degrees(hwp_ang) # Note, 0 if not set.

        if return_tod and not return_point:
            return ret_tod

        elif not return_tod and return_point:
            return ret_pix, ret_nside, ret_pa, ret_hwp

        elif return_tod and return_point:
            return ret_tod, ret_pix, ret_nside, ret_pa, ret_hwp

    @staticmethod
    def _scan_modulate_pa(tod, pix, pa, maps, s_vals,
                          reality=False, interp=False):
        '''
        Populate TOD with maps[pix] * exp( -i s psi).

        Arguments
        ---------
        tod : (nsamp) array
            TOD to be added to.
        pix : (nsamp) int array or sequence of arrays.
            HEALPix pixel indices or (ra, dec) if interpolation
            is used.
        pa : (nsamp) array
            Psi angle (position angle).
        maps : (nspin, npix) array
            Maps to be scanned.
        s_vals : (nspin) array
            Spin values.

        Keyword Arguments
        -----------------
        reality : bool
            Whether or not the maps obey the reality condition:
            sf = -sf^*. If set, use the s <-> -s symmetry.
        interp : bool
            If interpolation should be used while scanning.

        Raises
        ------
        ValueError
            If negative spin values are given while reality symmetry
            is used.
            If pix argument is not [ra, dec] if interpolation is used.
        '''

        if reality and s_vals.min() < 0:
            raise ValueError('Negative spin not allowed when reality '
                             'symmetry is used')

        # Use recursion of exponent if spin values are spaced by 1.
        if np.array_equal(s_vals, np.arange(s_vals[0], s_vals[-1] + 1)):
            recursion = True
            expmipa = np.exp(-1j * pa) # Used for recursion
            expmipas = np.exp(-1j * pa * s_vals[0])
        else:
            recursion = False

        for sidx, spin in enumerate(s_vals):

            if recursion and sidx != 0:
                # Already initialized for first spin.
                expmipas *= expmipa
            else:
                expmipas = np.exp(-1j * pa * spin)

            if interp:
                ra, dec = pix
                scan = hp.get_interp_val(maps[sidx], dec, ra)
            else:
                scan = maps[sidx, pix] # Is already a copy.

            if spin != 0:
                scan *= expmipas

            if reality:
                scan = np.real(scan)
                if spin != 0:
                    # For -s and +s.
                    scan *= 2

            tod += scan


    def init_spinmaps(self, alm, beam, ground_alm=None, max_spin=5, nside_spin=256,
                      symmetric=False, input_v=False, beam_v=False):
        '''
        Compute appropriate spinmaps for beam and
        all its ghosts.

        Arguments
        ---------
        alm : tuple of array-like
            Sky alms (alm, almE, almB) or (alm, almE, almB, almV)
        beam : <detector.Beam> object
            If provided, create spinmaps for main beam and
            all ghosts (if present).

        Keyword arguments
        -----------------
        ground_alm : tuple of array-like, optional
            Ground alms (ground_almI, ground_almE, ground_almB) 
        max_spin : int, optional
            Maximum azimuthal mode describing the beam (default : 5)
        nside_spin : int
            Nside of spin maps (default : 256)
        symmetric : bool
            If set, only use s=0 (intensity) and s=2 (lin. pol).
            `max_spin` kwarg is ignored (default : False)
        input_v : bool
            include the 4th alm component
        beam_v : bool
            include the 4th blm component


        Notes
        -----
        Populates following class attributes:

        spinmaps : dict
            Dictionary constaining all spinmaps for main beam and
            possible ghosts.

        The `ghost_idx` attribute decides whether ghosts have
        distinct beams. See `Beam.__init__()`.
        '''

        self.spinmaps = {'main_beam' : {},
                         'ground': {},
                         'ghosts': []}

        max_s = min(beam.mmax, max_spin)
        blm = beam.blm # main beam
        # calculate spinmaps for main beam

        spinmap_dict = self._init_spinmaps(alm,
                    blm, max_s, nside_spin, symmetric=beam.symmetric,
                    hwp_mueller=beam.hwp_mueller, input_v=input_v,
                    beam_v=beam_v)

        # Names: s0a0, s0a2, s2a0, s2a2, s2a4.
        # s refers to spin value under psi, a to spin value under HWP rot.
        self.spinmaps['main_beam'] = spinmap_dict

        if beam.ghosts:

            # Find unique ghost beams.
            assert len(beam.ghosts) == beam.ghost_count
            # Ghost_indices.
            g_indices = np.empty(beam.ghost_count, dtype=int)

            for gidx, ghost in enumerate(beam.ghosts):
                g_indices[gidx] = ghost.ghost_idx

            unique, u_indices = np.unique(g_indices, return_index=True)

            # Calculate spinmaps for unique ghost beams.
            for uidx, u in enumerate(unique):

                self.spinmaps['ghosts'].append([])
                self.spinmaps['ghosts'][u] = {}

                # Use the blms from the first occurrence of unique
                # ghost_idx.
                ghost = beam.ghosts[u_indices[uidx]]

                blm = ghost.blm
                max_s = min(ghost.mmax, max_spin)

                spinmap_dict = self._init_spinmaps(alm,
                            blm, max_s, nside_spin, symmetric=ghost.symmetric,
                            hwp_mueller=ghost.hwp_mueller,
                            input_v=input_v, beam_v=beam_v)
                self.spinmaps['ghosts'][u] = spinmap_dict

        if ground_alm is not None:
            self.spinmaps['ground'] = self._init_spinmaps(ground_alm,
                                    blm, max_s, nside_spin, symmetric=beam.symmetric,
                                    hwp_mueller=beam.hwp_mueller, input_v=input_v,
                                    beam_v=beam_v)


    @staticmethod
    def _init_spinmaps(alm, blm, max_spin, nside,
                       symmetric=False, hwp_mueller=None,
                       input_v=False, beam_v=False):
        '''
        Compute convolution of map with different spin modes
        of the beam.

        Arguments
        ---------
        alm : tuple of array-like
            Sky alms (alm, almE, almB) or (alm, almE, almB, almV)
        blm : tuple of array-like
            Beam alms (blmI, blmm2, blmp2) or (blmI, blmm2, blmp2, blmV)
        max_spin : int
            Maximum azimuthal mode describing the beam.
        nside : int
            Nside of output maps.

        Keyword arguments
        -----------------
        symmetric : bool
            If set, only use s=0 (I, V) and s=2 (Q, U).
            `max_spin` kwarg is ignored (default : False).
        hwp_mueller : (4, 4) array, None
            Unrotated Mueller matrix of half-wave plate.
        input_v : bool
            include the 4th alm component if
            it exists
        beam_v : bool
            include the 4th blm component if
            it exists

        returns
        -------
        spinmap_dict : dict of dicts
            Container for spinmaps and spin values for each type, see notes.

        Notes
        -----
        Keys of output dictionary refer to the behaviour under sky rotation (s)
        and HWP rotation (a) of the maps in case of azimuthally symmetric beams.
        So s0a0 is completely scalar, s2a4 is the standard linear polarization.

        Uses minimum lmax value of beam and sky SWSH coefficients.
        Uses min value of max_spin and the mmax of blm as azimuthal
        band-limit.
        '''
        # Output.
        spinmap_dict = {}

        # Check for nans in alms. E.g from a nan pixel in original maps.
        crash_a = False
        crash_b = False
        i = 0
        if np.shape(alm)[0] != 4 and input_v:
            raise ValueError("There is no V sky component")

        if np.shape(blm)[0] != 4 and beam_v:
            raise ValueError("There is no V beam component")

        # break the condition
        while i < np.shape(alm)[0] and not crash_a:
            crash_a = ~np.isfinite(np.sum(alm[i][:]))
            i += 1
        while i < np.shape(blm)[0] and not crash_b:
            crash_b = ~np.isfinite(np.sum(blm[i][:]))
            i +=1
        if crash_a or crash_b:
            name = 'alm' if crash_a else 'blm'
            raise ValueError('{}[{}] contains nan/inf.'.format(name, i-1))

        # Match up bandlimits beam and sky.
        lmax_sky = hp.Alm.getlmax(alm[0].size)
        lmax_beam = hp.Alm.getlmax(blm[0].size)
        if lmax_sky > lmax_beam:
            alms = tools.trunc_alm([alm[0],alm[1],alm[2]], lmax_beam)
            if input_v:
                almV = tools.trunc_alm(alm[3], lmax_beam)
                alm = [alms[0],alms[1],alms[2],almV]
            else:
                alm = [alms[0],alms[1],alms[2]]
            lmax = lmax_beam
        elif lmax_beam > lmax_sky:
            blms = tools.trunc_alm([blm[0],blm[1],blm[2]], lmax_sky)
            if beam_v:
                blmV = tools.trunc_alm(blm[3], lmax_sky)
                blm = [blms[0],blms[1],blms[2],blmV]
            else:
                blm = [blms[0],blms[1],blms[2]]
            lmax = lmax_sky
        else:
            lmax = lmax_sky
        if symmetric:
            spin_values_unpol = np.array([0], dtype=int)
            spin_values_pol = np.array([2], dtype=int)
            # We need separate spin values for the s0a2 non-ideal HWP terms.
            spin_values_s0a2 = spin_values_unpol.copy()
        else:
            # Intensity only needs s >= 0 maps.
            spin_values_unpol = np.arange(max_spin + 1)
            # Linear polarization needs -s,...,+s maps.
            spin_values_pol = np.arange(-max_spin, max_spin + 1)
            spin_values_s0a2 = spin_values_pol.copy()

        almE = alm[1]
        almB = alm[2]


        if hwp_mueller is not None:

            hwp_spin = tools.mueller2spin(hwp_mueller)

            # s0a0.
            spinmap_dict['s0a0'] = {}
            blm_s0a0 = ScanStrategy.blmxhwp(blm, hwp_spin, 's0a0',
                                    beam_v=beam_v)
            spinmap_dict['s0a0']['maps'] = ScanStrategy._spinmaps_real(
                alm[0], blm_s0a0, spin_values_unpol, nside)

            if input_v:
                blm_s0a0_v = ScanStrategy.blmxhwp(blm, hwp_spin, 's0a0_v',
                                                    beam_v=beam_v)
                spinmap_dict['s0a0']['maps'] += ScanStrategy._spinmaps_real(
                    alm[3], blm_s0a0_v, spin_values_unpol, nside)
                del blm_s0a0_v

            spinmap_dict['s0a0']['s_vals'] = spin_values_unpol
            del blm_s0a0

            # s2a4.
            spinmap_dict['s2a4'] = {}
            blmm2, blmp2 = ScanStrategy.blmxhwp(blm, hwp_spin, 's2a4')
            blmE, blmB = tools.spin2eb(blmm2, blmp2)
            spinmap_dict['s2a4']['maps'] = ScanStrategy._spinmaps_complex(
                almE, almB, blmE, blmB, spin_values_pol, nside)
            spinmap_dict['s2a4']['s_vals'] = spin_values_pol

            # s0a2.
            spinmap_dict['s0a2'] = {}
            blmm2, blmp2 = ScanStrategy.blmxhwp(blm, hwp_spin, 's0a2')
            blmE, blmB = tools.spin2eb(blmp2, blmm2)
            # Minus sign is because eb2spin applied to alm_I results in (-1) alm_I.
            spinmap_dict['s0a2']['maps'] = ScanStrategy._spinmaps_complex(
                -alm[0], alm[0] * 0, blmE, blmB, spin_values_s0a2, nside)

            if input_v:
                blmm2, blmp2 = ScanStrategy.blmxhwp(blm, hwp_spin, 's0a2_v')
                blmE, blmB = tools.spin2eb(blmp2, blmm2)
                spinmap_dict['s0a2']['maps'] += ScanStrategy._spinmaps_complex(
                    -alm[3], alm[3] * 0, blmE, blmB, spin_values_s0a2, nside)
            spinmap_dict['s0a2']['s_vals'] = spin_values_s0a2

            # s2a2.
            spinmap_dict['s2a2'] = {}
            blmm2, blmp2 = ScanStrategy.blmxhwp(blm, hwp_spin, 's2a2',
                                beam_v=beam_v)
            blmE, blmB = tools.spin2eb(blmm2, blmp2)
            spinmap_dict['s2a2']['maps'] = ScanStrategy._spinmaps_complex(
                almE, almB, blmE, blmB, spin_values_pol, nside)
            spinmap_dict['s2a2']['s_vals'] = spin_values_pol

            # s2a0.
            spinmap_dict['s2a0'] = {}
            blmm2, blmp2 = ScanStrategy.blmxhwp(blm, hwp_spin, 's2a0')
            blmE, blmB = tools.spin2eb(blmm2, blmp2)
            spinmap_dict['s2a0']['maps'] = ScanStrategy._spinmaps_complex(
                almE, almB, blmE, blmB, spin_values_pol, nside)
            spinmap_dict['s2a0']['s_vals'] = spin_values_pol

        else:
            # Old default behaviour.
            blmE, blmB = tools.spin2eb(blm[1], blm[2])

            # Unpolarized sky and beam.
            spinmap_dict['s0a0'] = {}
            spinmap_dict['s0a0']['maps'] = ScanStrategy._spinmaps_real(
                alm[0], blm[0], spin_values_unpol, nside)

            if beam_v and input_v:
                spinmap_dict['s0a0']['maps'] += ScanStrategy._spinmaps_real(
                    alm[3], blm[3], spin_values_unpol, nside)

            spinmap_dict['s0a0']['s_vals'] = spin_values_unpol

            # Linearly polarized sky and beam.
            spinmap_dict['s2a4'] = {}
            spinmap_dict['s2a4']['maps'] = ScanStrategy._spinmaps_complex(
                almE, almB, blmE, blmB, spin_values_pol, nside)
            spinmap_dict['s2a4']['s_vals'] = spin_values_pol

        return spinmap_dict

    @staticmethod
    def blmxhwp(blm, hwp_spin, mode, beam_v=False):
        '''
        Arguments
        ---------
        blm : sequence of arrays
            Beam alms (blmI, blmm2, blmp2) or (blmI, blmm2, blmp2, blmV)
        hwp_spin : (4, 4) array, None
            Unrotated Mueller matrix of half-wave plate in complex basis.
            See `tools.mueller2spin`.
        mode : str
            Pick between 's0a0', 's0a0_v', s2a4', 's0a2', 's0a2_v', 's2a2' or 's2a0'.
        beam_v : bool
            include the 4th blm component when constructing the spinmaps

        Returns
        -------
        blmxhwp : array or tuple of arrays

        Raises
        ------
        ValueError
            If mode is not recognized.

        Notes
        -----
        We have the SWSH modes of the beam (the blms).
        However, we want to use the SHWSH modes of the
        beam x HWP Mueller matrix elements. Instead of
        Multiplying the two fields in real space, we
        effecitively convolve the two in harmonic space,
        this way we can use the blms and avoid going back
        to real space.
        '''
        if mode == 's0a0':
            blm_s0a0 = blm[0] * hwp_spin[0,0]
            if beam_v:
                blm_s0a0 += blm[3] * hwp_spin[3,0]
            return blm_s0a0

        elif mode == 's0a0_v':
            blm_s0a0_v = blm[0] * hwp_spin[0,3]
            if beam_v:
                blm_s0a0_v += blm[3] * hwp_spin[3,3]

            return blm_s0a0_v

        elif mode == 's2a4':
            # Note the switch, blmm2 becomes blmp2 and vice versa.
            blmp2, blmm2 = tools.shift_blm(blm[1], blm[2], 4, eb=False)
            #blmm2 *= hwp_spin[1,2]
            #blmp2 *= hwp_spin[2,1]
            blmm2 *= hwp_spin[2,1]
            blmp2 *= hwp_spin[1,2]

            return blmm2, blmp2

        elif mode == 's0a2':
            blmm2, blmp2 = tools.shift_blm(blm[1], blm[2], 2, eb=False)            
            blmm2 *= hwp_spin[0,2] * np.sqrt(2)
            #blmm2 *= hwp_spin[1,0] * np.sqrt(2)
            blmp2 *= hwp_spin[2,0] * np.sqrt(2)
            return blmm2, blmp2

        elif mode == 's0a2_v':
            blmm2_v, blmp2_v = tools.shift_blm(blm[1], blm[2], 2, eb=False)
            blmm2_v *= hwp_spin[3,2] * np.sqrt(2)
            #blmm2_v *= hwp_spin[1,3] * np.sqrt(2)
            blmp2_v *= hwp_spin[2,3] * np.sqrt(2)
            return blmm2_v, blmp2_v

        elif mode == 's2a2':
            blmm2 = blm[0]
            blmp2 = blmm2
            # Note the switch, blmm2 becomes blmp2 and vice versa.
            blmp2, blmm2 = tools.shift_blm(blmm2, blmp2, 2, eb=False)
            blmm2 *= hwp_spin[0,2] * np.sqrt(2)
            blmp2 *= hwp_spin[0,1] * np.sqrt(2)
            #blmm2 *= hwp_spin[0,1] * np.sqrt(2)
            #blmp2 *= hwp_spin[0,2] * np.sqrt(2)
            
            if beam_v:
                blmm2_v = blm[3]
                blmp2_v = blmm2_v
                blmp2_v, blmm2_v = tools.shift_blm(blmm2_v, blmp2_v, 2, eb=False)
                blmm2_v *= hwp_spin[3,2] * np.sqrt(2)
                blmp2_v *= hwp_spin[3,1] * np.sqrt(2)
                
                blmm2 += blmm2_v
                blmp2 += blmp2_v

            return blmm2, blmp2

        elif mode == 's2a0':
            blmm2 = blm[1] * hwp_spin[2,2]
            blmp2 = blm[2] * hwp_spin[1,1]
            return blmm2, blmp2

        else:
            raise ValueError('{} is unrecognized mode'.format(mode))

    @staticmethod
    def _spinmaps_real(alm, blm, spin_values, nside):
        '''
        Return spinmaps for a "real-valued" spin field with
        spin-weighted spherical harmonic coefficients given
        by sflm = alm * bls.

        Arguments
        ---------
        alm : complex array
            Sky alms.
        blm : complex array
            Beam alms.
        spin_values : (nspin) array
            Nonnegative spin values to be considered.
        nside : int
            Nside of output.

        Returns
        -------
        spinmaps : (nspin, 12*nside**2) array
            Real if single spin value is 0, otherwise complex.

        Raises
        ------
        ValueError
            If spin values contain negative values.

        Notes
        -----
        We only consider non-negative spin values here because for
        a real spin field sf we have sf^* = -sf, or, in terms of the
        harmonic coefficients: sflm^* = -sfl-m (-1)^(s+m). We can
        therefore reconstruct the fields for negative s if needed.
        '''

        if spin_values.min() < 0:
            raise ValueError('Negatve spin provided for real beams and sky')

        lmax = hp.Alm.getlmax(alm.size)

        if len(spin_values) == 1 and spin_values[0] == 0:
            # Symmetric case
            func = np.zeros((1, 12*nside**2), dtype=float)
        else:
            func = np.zeros((len(spin_values), 12*nside**2),
                            dtype=np.complex128)

        for sidx, s in enumerate(spin_values): # s are azimuthal modes in bls.
            bell = tools.blm2bl(blm, m=s, full=True)

            if s == 0: # Scalar transform.

                flms = hp.almxfl(alm, bell, inplace=False)
                func[sidx,:] = hp.alm2map(flms, nside, verbose=False)

            else: # Spin transforms.

                # The positive s values.
                flms = hp.almxfl(alm, bell, inplace=False)

                # The negative s values: alm bl-s = alm bls^* (-1)^s.
                flmms = hp.almxfl(alm, (-1) ** s * np.conj(bell), inplace=False)

                # Turn into plus and minus (think E and B) modes for healpy's
                # alm2map_spin.
                spinmaps = hp.alm2map_spin(tools.spin2eb(flmms, flms, spin=s),
                                           nside, s, lmax, lmax)

                func[sidx,:] = spinmaps[0] + 1j * spinmaps[1]

        return func

    @staticmethod
    def _spinmaps_complex(almE, almB, blmE, blmB, spin_values, nside):
        '''
        Return spinmaps for a complex spin field with harmonic
        coefficients given by sflm = 2alm -2bls. Where 2alm and
        -2bls are the harmonic coefficients of two (real) spin
        fields.

        Arguments
        ---------
        almE : complex array
            Sky E-mode alms.
        almB : complex array
            Sky B-mode alms
        blmE : complex array
            Beam E-mode alms.
        blmB : complex array
            Beam B-mode alms.
        spin_values : (nspin) array
            Spin values to be considered.
        nside : int
            Nside of output.

        Returns
        -------
        spinmaps : (nspin, 12*nside**2) complex array

        Notes
        -----
        A complex spin field is one that does not obey the reality
        condition for spin fields: sf^* != -sf, or, equivalently,
        sflm^* != -sfl-m (-1)^(s+m). This means that the spin 0 output
        map is complex and that the s < 0 output maps are not generally
        given by the complex conjugate of the s > 0 maps.
        '''

        almm2, almp2 = tools.eb2spin(almE, almB)
        blmm2, blmp2 = tools.eb2spin(blmE, blmB)

        lmax = hp.Alm.getlmax(almm2.size)

        func_c = np.zeros((len(spin_values), 12*nside**2),
                              dtype=np.complex128)

        for sidx, s in enumerate(spin_values):
            bellp2 = tools.blm2bl(blmp2, m=abs(s), full=True)
            bellm2 = tools.blm2bl(blmm2, m=abs(s), full=True)

            if s >= 0:

                ps_flm_p, ps_flm_m = tools.spin2eb(
                    hp.almxfl(almm2, np.conj(bellm2) * (-1) ** abs(s)),
                    hp.almxfl(almp2, bellm2),
                    spin = abs(s))

            if s <= 0:

                ms_flm_p, ms_flm_m = tools.spin2eb(
                    hp.almxfl(almp2, np.conj(bellp2) * (-1) ** abs(s)),
                    hp.almxfl(almm2, bellp2),
                    spin = abs(s))

            if s == 0:
                # The (-1) factor for spin 0 is explained in HEALPix doc.
                spinmaps = [hp.alm2map(-ps_flm_p, nside, verbose=False),
                            hp.alm2map(ms_flm_m, nside, verbose=False)]
                func_c[sidx,:] = spinmaps[0] + 1j * spinmaps[1]

            if s > 0:
                spinmaps = hp.alm2map_spin([ps_flm_p, ps_flm_m],
                                           nside, s, lmax, lmax)
                func_c[sidx,:] = spinmaps[0] + 1j * spinmaps[1]

            elif s < 0:
                spinmaps = hp.alm2map_spin([ms_flm_p, ms_flm_m],
                                           nside, abs(s), lmax, lmax)
                func_c[sidx,:] = spinmaps[0] - 1j * spinmaps[1]

        return func_c



    def bin_tod(self, beam, tod=None, flag=None, init=True, add_to_global=True,
                filter_4fhwp=False, filter_highpass=False, **kwargs):
        '''
        Take internally stored tod and boresight
        pointing, combine with detector offset,
        and bin into map and projection matrices.

        Arguments
        ---------
        beam : <detector.Beam> object
            The main beam of the detector.

        Keyword arguments
        -----------------
        tod : array-like, None
            Time-ordered data corresponding to beam, if None
            tries to use `tod` attribute. (default : None)
        flag : array-like, None, bool
            Time-ordered flagging corresponding to beam, if None
            tries to use `flag` attribute. Same size as `tod`
            If False, do not use flagging. (default : None)
        init : bool
            Call `init_dest()` before binning. (default : True)
        add_to_global : bool
            Add local maps to maps allocated by `allocate_maps`.
            (default : True)
        filter_4fhwp : bool
            Only use TOD modes modulated at 4 x the HWP frequency.
            Only allowed with spinning HWP. (default : False)
        kwargs : {chunk_opts}
        '''

        # HWP does not depend on particular detector.
        q_hwp = self.hwp_quat(np.degrees(self.hwp_ang)-self.hwp_dict['varphi'])

        qidx_start, qidx_end = self._chunk2idx(**kwargs)

        self.init_point(q_bore=self.q_bore[qidx_start:qidx_end],
                        ctime=self.ctime[qidx_start:qidx_end],
                        q_hwp=q_hwp)


        # Use q_off quat with polang (and instr. ang) included.
        polang = beam.polang # Possibly offset polang for binning.

        # Get detector offset quaternion that includes boresight rotation.
        q_off = beam.q_off
        # Add polang to q_off as first rotation. Note minus sign.
        polang = -np.radians(polang)
        q_polang = np.asarray([np.cos(polang/2.), 0., 0., np.sin(polang/2.)])
        q_off = tools.quat_left_mult(q_off, q_polang)

        if init:
            self.init_dest(nside=self.nside_out, pol=True, vpol=False, reset=True)

        q_off = q_off[np.newaxis]

        if tod is None:
            tod = self.tod

        if filter_4fhwp:
            if self.hwp_dict['mode'] == 'continuous':
                hwp_freq = self.hwp_dict['freq']
                tools.filter_tod_hwp(tod, self.fsamp, hwp_freq)
            else:
                raise ValueError(
                    'filter_4fhwp not valid with hwp mode : {}'.format(
                    self.hwp_dict['mode']))
        if filter_highpass:
            if bool(self.filter_dict):
                w_c =  self.filter_dict['w_c']
                forder = self.filter_dict['m']
                #Butterworth filter
                tools.filter_tod_highpass(tod, self.fsamp, w_c, forder)
            else:
                tod  -= np.average(tod) #Very rough filtering method

        # Note that qpoint wants (,nsamples) shaped tod array.
        tod = tod[np.newaxis]

        if flag is False:
            flag  = None
        elif flag is None and hasattr(self, 'flag'):
            flag = self.flag[qidx_start:qidx_end]
            flag = flag[np.newaxis]
        elif flag:
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
            Fill value for unobserved pixels (default : hp.UNSEEN)
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


        if self.mpi_rank == 0:
            # Suppress warnings from numpy linalg.
            with catch_warnings(record=True) as w:
                filterwarnings('ignore', category=RuntimeWarning)

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
