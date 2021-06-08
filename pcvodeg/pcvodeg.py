#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:25:28 2020

@author: ogurcan
"""
import os
import numpy as np
from ctypes import cdll,CFUNCTYPE,POINTER,c_double,c_int,byref

class pcvodeg:
    def __init__(self, fn, y, t0, t1, nthreads, **kwargs):
        self.libpcvod = cdll.LoadLibrary(os.path.dirname(__file__)+'/libpcvodeg.so')
        self.fnpytype=CFUNCTYPE(None, c_double, POINTER(c_double), POINTER(c_double))
        self.shape=y.shape
        self.size=int(y.size*y.dtype.itemsize/np.dtype(float).itemsize)
        self.nthreads=nthreads
        self.fn=fn
        self.kwargs=kwargs
        self.t0=t0
        self.t1=t1
        self.y=y
        self.t=t0
        self.state=0
        self.atol = kwargs.get('atol',1e-8)
        self.rtol = kwargs.get('rtol',1e-6)
        self.mxsteps = int(kwargs.get('mxsteps',10000))
        self.fnpcvod=self.fnpytype(lambda x,y,z : self.fnforw(x,y,z))
        self.libpcvod.init_solver(self.size,self.nthreads,self.y.ctypes.data_as(POINTER(c_double))
                             ,c_double(self.t0),self.fnpcvod
                             ,c_double(self.atol),c_double(self.rtol),c_int(self.mxsteps));

    def fnforw(self,t,y,dydt):
        y_ar=np.ctypeslib.as_array(y,(self.size,)).view(dtype=complex).reshape(self.shape)
        dydt_ar=np.ctypeslib.as_array(dydt,(self.size,)).view(dtype=complex).reshape(self.shape)
        u=np.ndarray(self.shape,dtype=complex,buffer=y_ar)
        dudt=np.ndarray(self.shape,dtype=complex,buffer=dydt_ar)
        self.fn(t,u,dudt)

    def integrate_to(self,tnext):
        t=c_double()
        state=c_int()
        self.libpcvod.integrate_to(c_double(tnext),byref(t),byref(state))
        self.t=t.value
        self.state=state.value

    def successful(self):
        return self.state==0
#        self.libpcvod.integrate_to(c_double(tnext),POINTER(c_double)(tp),POINTER(c_double)(statusp))
#        self.t=tp.contents
#        self.status=statusp.contents

# if __name__ == "__main__":    
#     phi=DistArray((10,10),subcomm=(1,0),dtype=complex)
#     dphidt=DistArray((10,10),subcomm=(1,0),dtype=complex)
#     gam=0.1
#     phi[:,:]=1.0
#     def fntest(t,y,dydt):
#         print("t=",t)
#         dydt[:,:] = gam*y[:,:]
#     pcv=pcvodeg(fntest,phi,dphidt,0.0,100.0,atol=1e-12,rtol=1e-8)
#     pcv.integrate_to(10.0)
