# beamconv

**Authors**: Adri J. Duivenvoorden and Jon E. Gudmundsson

**contact**: adriaand@princeton.edu

Simulate the scanning of the CMB sky while incorporating realistic beams and
scan strategies.

This code uses (spin-)spherical harmonic representations of the (polarized) beam response
and sky to generate simulated CMB detector signal timelines. Beams can be arbitrarily shaped.
Pointing timelines can be read in or calculated on the fly. Optionally, the results can be
binned on the sphere.

### Usage

See example scripts [`beamconv/test.py`](../../tree/master/beamconv/test.py) and Jupyter notebooks in [`notebooks`](../../tree/master/notebooks). In particular, we suggest that the user try running [`notebooks/introduction.ipynb`](../../tree/master/notebooks/introduction.ipynb) followed by [`notebooks/simple_scan.ipynb`](../../tree/master/notebooks/simple_scan.ipynb).

### Dependencies
Apart from the standard libraries, [NumPy](https://github.com/numpy/numpy), [Healpy](https://github.com/healpy/healpy), and [Matplotlib](https://github.com/matplotlib/matplotlib), this code makes use of [qpoint](https://github.com/arahlin/qpoint), a lightweight quaternion-based library for telescope pointing, written by Sasha Rahlin. The code has been tested with Python 2.7 and 3.6.

### Installation

```
python setup.py install --user
```

or, when using pip and virtuelenv:

```
pip install .
```

Run tests:

```
python -m pytest tests
```

Testing requires the `pytest` package, this can be automatically obtained during installation by running:

```
pip install .[test]
```

Consider adding the `-e` flag to the `pip install` command to enable automatic updating of code changes when developing.






