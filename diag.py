#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:28:30 2021

@author: ogurcan
"""
import numpy as np
from hwak_omp import hasegawa_wakatani
hw=hasegawa_wakatani(flname='out2.h5',onlydiag=True)

t=hw.fl['fields/t'][()]
kx=hw.fl['data/kx'][()]
ky=hw.fl['data/ky'][()]
ksqr=kx**2+ky**2
k=np.sqrt(ksqr)
ukres=hw.fl['fields/uk']

Nt=t.shape[0]
g=2.0
k0=np.min(k[k>0])
k1=np.max(k)
Nsh=int(np.ceil(np.log(k1/k0)/np.log(g)))
kn=k0*g**(np.arange(Nsh))
Nx=kx.shape[0]
Ny=ky.shape[0]*2-2
En=np.zeros((Nt,Nsh))
dEndt=np.zeros((Nt,Nsh))

uk=hw.uk
dukdt=hw.dukdt
s=np.sqrt(g)

for i in range(Nt):
    print(i)
    uk[:]=ukres[i,]
    phik,nk=uk
    hw.rhs(t,uk,dukdt)
    dphikdt,dnkdt=dukdt
    Ek=np.abs(phik)**2*ksqr/2/Nx**2/Ny**2
    dEkdt=np.real(dphikdt*phik.conj()*ksqr)/Nx**2/Ny**2
    for l in range(Nsh):
        krng=((k<=kn[l]*s) | (l==Nsh-1)) & ((k>kn[l]/s) | (l==0)) & ((ky>0) | (kx>=0))
#        krng_zonal=((k<=kn[l]*s) | (l==Nsh-1)) & ((k>kn[l]/s) | (l==0)) & ((ky==0) & (kx>=0))
#        krng_nonzonal=((k<=kn[l]*s) | (l==Nsh-1)) & ((k>kn[l]/s) | (l==0)) & ((ky>0))
        En[i,l]=np.sum(Ek[krng])
        dEndt[i,l]=np.sum(dEkdt[krng])
hw.fl.close()
