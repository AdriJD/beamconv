# beamconv

**Authors**: Adri J. Duivenvoorden and Jon E. Gudmundsson

**contact**: adri.j.duivenvoorden@gmail.com

Simulate the scanning of the CMB sky while incorporating realistic beams and
scan strategies.

This code uses (spin-)spherical harmonic representations of the (polarized) beam response
and sky to generate simulated CMB detector signal timelines. Beams can be arbitrarily shaped.
Pointing timelines can be read in or calculated on the fly. Optionally, the results can be
binned on the sphere.

### Usage

See example scripts [`beamconv/test.py`](../../tree/master/beamconv/test.py) and Jupyter notebooks in [`notebooks`](../../tree/master/notebooks).

### Dependencies
Apart from the standard libraries, [NumPy](https://github.com/numpy/numpy), [Healpy](https://github.com/healpy/healpy), and [Matplotlib](https://github.com/matplotlib/matplotlib), this code makes use of [qpoint](https://github.com/arahlin/qpoint), a lightweight quaternion-based library for telescope pointing, written by Sasha Rahlin.

### Installation

```
python setup.py install --user
```
run unittests:

```
python setup.py test
```






