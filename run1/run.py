#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:10:50 2021

@author: ogurcan
"""
import sys
import os
import numpy as np
sys.path.insert(1, os.path.realpath('../'))
from hwak_omp import hasegawa_wakatani

hw=hasegawa_wakatani(modified=True,
                     wecontinue=False,
                     flname="out.h5",
                     C=10.0,
                     kap=1.0,
                     Npx=1024,
                     Npy=1024,
                     nu=1e-4,
                     D=1e-4,
                     nuZF=1e-4,
                     DZF=0.0,
                     t1=1000,
                     dtstep=1.0,
                     dtout=1.0,
                     Amp0=1.0e3,
                     nthreads=8)
hw.run()
