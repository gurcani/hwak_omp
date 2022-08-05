# hwak_omp

This is a pseudo-spectral 2D Hasegawa-Wakatani solver, written in python, using [numba](https://github.com/numba/numba) for number crunching. It uses [pyfftw](https://github.com/pyFFTW/pyFFTW) for computing the fast fourier transform, and [h5py](https://github.com/h5py/h5py) for file I/O. It also features a wrapper for the [sundials](https://github.com/LLNL/sundials) cvode suite of solvers, for the adaptive time stepping. 
## Usage:
The minimal usage with the default parameters is something like:
```python
from hwak_omp import hasegawa_wakatani
hw=hasegawa_wakatani()
hw.run()
```
The solver will recognize a number of variables such as:
```pyhton
hw=hasegawa_wakatani(modified=True,C=10.0,kap=1.0, Npx=1024, Npy=1024)
 ```
the full list can be found in [hwak_omp.py](https://github.com/gurcani/hwak_omp/blob/main/hwak_omp.py) as *default_parameters*, *default_solver_parameters*, *default_controls*.
