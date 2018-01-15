# cmb_beams

**Authors**: Adri J. Duivenvoorden and Jon E. Gudmundsson

**contact**: adri.j.duivenvoorden@gmail.com

Code to simulate the scanning of the CMB sky while incorporating realistic beams and
scan strategies. This code uses a spherical harmonic representation of the beam response
in I, Q, and U to generate a signal timeline that can then be binned into a map.


### Dependencies

 * [qpoint](https://github.com/arahlin/qpoint)
   * A lightweight quaternion-based library for telescope pointing, written by Sasha Rahlin.
 * [NumPy](https://github.com/numpy/numpy)
 * [Healpy](https://github.com/healpy/healpy)
 * [Matplotlib](https://github.com/matplotlib/matplotlib)

### Installation

```
python setup.py install --user
```
run unittests:

```
python setup.py test
```

### Usage

See example scripts in `cmb_beams/test.py` and Jupyter notebooks in `notebooks`
