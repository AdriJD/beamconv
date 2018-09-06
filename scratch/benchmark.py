import time
import os
import copy

def t_lmax():
    '''Time init_detpair as function of lmax, mmax.'''

    os.environ["OMP_NUM_THREADS"] = "1" 
    import numpy as np
    import healpy as hp
    from beamconv import ScanStrategy
    from beamconv import Beam
    
    scan_opts = dict(duration=3600,
                     sample_rate=100)

    lmax_range = np.logspace(np.log10(500), np.log10(3000), 8, dtype=int)
    nsides = np.ones_like(lmax_range) * np.nan
    nside_range = 2 ** np.arange(15)

    #mmax_range = np.array([2])
    timings = np.ones((mmax_range.size, lmax_range.size)) * np.nan
    timings_cpu = np.ones((mmax_range.size, lmax_range.size)) * np.nan

    alm = np.zeros((3, hp.Alm.getsize(lmax=4000)), dtype=np.complex128)

    S = ScanStrategy(**scan_opts)


    for lidx, lmax in enumerate(lmax_range):

        nside = nside_range[np.digitize(0.5 * lmax, nside_range)]
        nsides[lidx] = nside

        beam_opts = dict(az=0, el=0, polang=0,
                             fwhm=40, btype='Gaussian', 
                             lmax=lmax)

        beam = Beam(**beam_opts)
        beam.blm

        for midx, mmax in enumerate(mmax_range):

            t0 = time.time()
            t0c = time.clock()
            S.init_detpair(alm, beam, beam_b=None, nside_spin=nside, max_spin=mmax,
                           verbose=False)
            t1 = time.time()
            t1c = time.clock()

            print('{}, {}, {}: {}'.format(lmax, mmax, nside, t1-t0))
            print('{}, {}, {}: {}'.format(lmax, mmax, nside, t1c-t0c))

            timings[midx,lidx] = t1 - t0
            timings_cpu[midx,lidx] = t1c - t0c

    np.save('./timings.npy', timings)
    np.save('./timings_cpu.npy', timings_cpu)
    np.save('./lmax_range.npy', lmax_range)
    np.save('./mmax_range.npy', mmax_range)
    np.save('./nsides.npy', nsides)

def scan(lmax=500, nside=512, mmax=2):
    '''Time scanning single detector.'''

    os.environ["OMP_NUM_THREADS"] = "1" 
    import numpy as np
    import healpy as hp
    from beamconv import ScanStrategy
    from beamconv import Beam

    ndays_range = np.logspace(np.log10(0.001), np.log10(50), 15)
    lmax = lmax
    nside = nside
    freq = 100.

    timings = np.ones((ndays_range.size), dtype=float)
    timings_cpu = np.ones((ndays_range.size), dtype=float)

    S = ScanStrategy(duration=24*3600, sample_rate=freq)
    beam_opts = dict(az=1, el=1, polang=10.,
                     fwhm=40, btype='Gaussian', 
                     lmax=lmax)

    beam = Beam(**beam_opts)
    S.add_to_focal_plane(beam)
    alm = np.zeros((3, hp.Alm.getsize(lmax=lmax)), dtype=np.complex128)
    print('init detpair...')
    S.init_detpair(alm, beam, beam_b=None, nside_spin=nside, max_spin=mmax,
                   verbose=True)
    print('...done')

    spinmaps = copy.deepcopy(S.spinmaps)
    
    # Calculate q_bore for 0.1 day of scanning and reuse this.
    const_el_opts = dict(az_throw=50.,
                         scan_speed=10.,
                         dec0=-70.)
    S.partition_mission()
    print('cons_el_scan...')
    const_el_opts.update(dict(start=0, end=int(24*3600*100*0.1)))
    S.constant_el_scan(**const_el_opts)
    print('...done')        

    q_bore_day = S.q_bore
    ctime_day = S.ctime
    
    for nidx, ndays in enumerate(ndays_range):

        duration = ndays * 24 * 3600
        nsamp = duration * freq

        S = ScanStrategy(duration=duration, sample_rate=freq, external_pointing=True)
        S.add_to_focal_plane(beam)

        S.partition_mission()

        q_bore = np.repeat(q_bore_day, np.ceil(10*ndays), axis=0)
        ctime = np.repeat(q_bore_day, np.ceil(10*ndays))
        
        def q_bore_func(start=None, end=None, q_bore=None):
            return q_bore[int(start):int(end),:]

        def ctime_func(start=None, end=None, ctime=None):
            return ctime[int(start):int(end)]
        
        S.spinmaps = spinmaps
        t0 = time.time()
        t0c = time.clock()
        S.scan_instrument_mpi(alm, binning=False, reuse_spinmaps=True,
                              interp=False, q_bore_func=q_bore_func,
                              ctime_func=ctime_func, q_bore_kwargs={'q_bore':q_bore},
                              ctime_kwargs={'ctime':ctime},
                              start=0, end=int(nsamp), cidx=0)
        t1 = time.time()
        t1c = time.clock()
        print t1 - t0
        print t1c - t0c

        timings[nidx] = t1 - t0
        timings_cpu[nidx] = t1c - t0c

    np.save('./scan_timing_lmax{}_nside{}_mmax{}.npy'.format(lmax, nside, mmax),
            timings)
    np.save('./scan_timing_cpu_lmax{}_nside{}_mmax{}.npy'.format(lmax, nside, mmax),
            timings_cpu)
    np.save('./scan_ndays_lmax{}_nside{}_mmax{}.npy'.format(lmax, nside, mmax),
            ndays_range)

if __name__ == '__main__':
    
    
    #t_lmax()
    #scan(lmax=3000, nside=2048, mmax=8)
    #scan(lmax=3000, nside=2048, mmax=2)
    scan(lmax=501, nside=256, mmax=2)
