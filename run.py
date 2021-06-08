#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:10:50 2021

@author: ogurcan
"""
from hwak_omp import hasegawa_wakatani

hw=hasegawa_wakatani(nthreads=4)
hw.run()
