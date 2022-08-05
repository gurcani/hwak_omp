# hwak_omp

This is a pseudo-spectral 2D Hasegawa-Wakatani solver, written in python, using [numba](https://github.com/numba/numba) for number crunching. It uses [pyfftw](https://github.com/pyFFTW/pyFFTW) for computing the fast fourier transform, and [h5py](https://github.com/h5py/h5py) for file I/O. It also features a wrapper for the [sundials](https://github.com/LLNL/sundials) cvode suite of solvers, for the adaptive time stepping. 
## Usage:
The minimal usage with the default parameters is something like:
```python
from hwak_omp import hasegawa_wakatani
hw=hasegawa_wakatani()
hw.run()
```
