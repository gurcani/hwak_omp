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
hw=hasegawa_wakatani(modified=True, # do we solve the modified hasegawa wakatani?
                     wecontinue=False, # do we continue an existing run
                     flname="out.h5", # filename to write
                     C=10.0, # the C parameter of the HW
                     kap=1.0, # the kappa parameter of the HW
                     Npx=1024, # padded resolution
                     Npy=1024, # padded resolution
                     nu=1e-4, # viscosity
                     D=1e-4, # density dissipation
                     nuZF=1e-4, # ZF friction
                     DZF=0.0, # particle loss
                     t1=1000, # tfinal 
                     dtstep=1.0, # time step for the loop
                     dtout=1.0, # output time step
                     Amp0=1.0e3, # initial amplitude
                     nthreads=8) # the number of threads to use (openmp).
 ```
