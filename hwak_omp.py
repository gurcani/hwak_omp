#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:16:26 2021

@author: ogurcan
"""

import numpy as np
from pyfftw import FFTW,empty_aligned
from time import time
from numba import njit, prange, set_num_threads
import h5py as h5
import scipy.integrate as spi
import sys,os
sys.path.insert(1, os.path.realpath(os.path.dirname(__file__)+'/pcvodeg'))
from pcvodeg import pcvodeg

default_parameters={
    'C':0.1,
    'kap':0.2,
    'nu':2e-4,
    'D':2e-4,
    'Npx':1024,
    'Npy':1024,
    'padx':3/2,
    'pady':3/2,
    'Lx':16*np.pi,
    'Ly':16*np.pi,
    'modified':False,
    'DZF':0.0,
    'nuZF':0.0,
    'nthreads': 4,
    'nuksqrpow' : 1,
}

default_solver_parameters={
    'solver':'pcvodeg',
    't0':0.0,
    't1':1000.0,
    'dtstep':1.0,
    'dtout':1.0,
    'atol' : 1e-16,
    'rtol' : 1e-8,
    'mxsteps' : 10000,
}

default_controls={
    'wecontinue':False,
    'onlydiag':False,
    'saveresult':True,
    'flname':"out.h5",
    'nthreads':1,
}

def oneover(x):
    res=np.zeros_like(x)
    inds=np.nonzero(x)
    res[inds]=1/x[inds]
    return res

def hermsymznyq(u):
    Nx=u.shape[-2]
    Ny=u.shape[-1]
    ll=tuple([slice(0,l,None) for l in u.shape[:-2]])+(slice(1,int(Nx/2),None),slice(0,1,None))
    l0=ll[:-2]+(slice(0,1,None),slice(0,1,None))
    u[l0]=np.real(u[l0])
    llp=ll[:-2]+(slice(Nx,int(Nx/2),-1),slice(0,1,None))
    u[llp]=u[ll].conj()
    u[ll[:-2]+(slice(int(Nx/2),int(Nx/2)+1,None),slice(0,Ny,None))]=0.0
    u[ll[:-2]+(slice(0,Nx,None),slice(Ny-1,Ny,None))]=0.0

@njit(fastmath=True,parallel=True)
def multin(u,F,kx,ky,ksqr):
    Nx=u.shape[1]
    Ny=u.shape[2]
    Npx=F.shape[1]
    Npy=F.shape[2]
    for i in prange(Nx):
        ix=i+int(i/(Nx/2))*(Npx-Nx)
        for j in range(Ny):
                F[0,ix,j]=1j*kx[i,j]*u[0,i,j]/Nx/(Ny*2-2)
                F[1,ix,j]=1j*ky[i,j]*u[0,i,j]/Nx/(Ny*2-2)
                F[2,ix,j]=-1j*ksqr[i,j]*kx[i,j]*u[0,i,j]/Npx/(Npy*2-2)
                F[3,ix,j]=-1j*ksqr[i,j]*ky[i,j]*u[0,i,j]/Npx/(Npy*2-2)
                F[4,ix,j]=1j*kx[i,j]*u[1,i,j]/Npx/(Npy*2-2)
                F[5,ix,j]=1j*ky[i,j]*u[1,i,j]/Npx/(Npy*2-2)

@njit(fastmath=True,parallel=True)
def multvec(dydt,y,a,x,b,f):
    Nx=y.shape[1]
    Ny=y.shape[2]
    Npx=x.shape[1]
    for i in prange(Nx):
        ix=i+int(i/(Nx/2))*(Npx-Nx)
        for j in range(Ny):
            dydt[0,i,j]=a[0,0,i,j]*y[0,i,j]+a[0,1,i,j]*y[1,i,j]+b[0,i,j]*x[0,ix,j]+f[0,i,j]
            dydt[1,i,j]=a[1,0,i,j]*y[0,i,j]+a[1,1,i,j]*y[1,i,j]+b[1,i,j]*x[1,ix,j]+f[1,i,j]

@njit(fastmath=True,parallel=True)
def multout62(F):
    for i in prange(F.shape[1]):
        for j in range(F.shape[2]-2):
            dxphi=F[0,i,j];
            dyphi=F[1,i,j];
            dxom=F[2,i,j];
            dyom=F[3,i,j];
            dxn=F[4,i,j];
            dyn=F[5,i,j];
            F[0,i,j]=(dxphi*dyom-dyphi*dxom)
            F[1,i,j]=(dxphi*dyn-dyphi*dxn)

def init_kspace_grid(Nx,Ny,Lx,Ly):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    kxl=np.r_[0:int(Nx/2),-int(Nx/2):0]*dkx
    kyl=np.r_[0:int(Ny/2+1)]*dky
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    return kx,ky

def init_ffts(Npx,Npy,nthreads):
    datk=empty_aligned((6,Npx,int(Npy/2+1)),dtype=complex);
    dat=datk.view(dtype=float)
#    dat=empty_aligned((2,Npx,Npy+1),dtype=float);
    pf6 = FFTW(datk[:6,:,:], datk[:6,:,:].view(dtype=float)[:,:,:-2], axes=(-2, -1),direction='FFTW_BACKWARD',threads=nthreads,normalise_idft=False)
    pf2 = FFTW(datk[:2,:,:].view(dtype=float)[:,:,:-2],datk[:2,:,:], axes=(-2, -1),direction='FFTW_FORWARD',threads=nthreads,normalise_idft=False)
    return datk,dat,pf6,pf2

def init_linmats(pars,kx,ky,ksqr):
    #Initializing the linear matrices
    C,kap,nu,D,nuZF,DZF=[pars[l] for l in ['C','kap','nu','D','nuZF','DZF']]
    nuksqrpow=pars['nuksqrpow']
    lm=np.zeros((2,2)+kx.shape,dtype=complex)
    nlm=np.zeros((2,)+kx.shape,dtype=float)
    forcelm=np.zeros((2,)+kx.shape,dtype=complex)
    forcelm[:]=0
    lm[0,0,:,:]=-C*oneover(ksqr)-nu*ksqr**nuksqrpow
    lm[0,1,:,:]=C*oneover(ksqr)
    lm[1,0,:,:]=-1j*kap*ky+C
    lm[1,1,:,:]=-C-D*ksqr**nuksqrpow
    lm[:,:,0,0]=0.0
    if(pars['modified']):
        lm[0,0,:,0]=-nuZF
        lm[0,1,:,0]=0.0
        lm[1,0,:,0]=0.0
        lm[1,1,:,0]=-DZF
    nlm[0,:,:]=1*oneover(ksqr)
    nlm[1,:,:]=-1.0
    nlm[:,0,0]=0.0
    hermsymznyq(lm)
    nlm[:,:,-1]=0.0
    nlm[:,int(nlm.shape[1]/2),:]=0.0
    return lm,nlm,forcelm

def init_fields(uk,kx,ky):
    kx0,ky0=0,0
    sigkx,sigky=0.5,0.5
    A=1e-4
    th=np.zeros(kx.shape)
    th[:,:]=np.random.rand(kx.shape[0],kx.shape[1])*2*np.pi;
    phik0=A*np.exp(-(kx-kx0)**2/2/sigkx**2-(ky-ky0)**2/2/sigky**2)*np.exp(1j*th);
    nk0=A*np.exp(-(kx-kx0)**2/2/sigkx**2-(ky-ky0)**2/2/sigky**2)*np.exp(1j*th);
    uk[:,:,:]=phik0,nk0
    uk[:,0,0]=0.0
    hermsymznyq(uk)

def load_pars(fl):
    pars={}
    for l,m in fl['params'].items():
        pars[l]=m[()]
    return pars

def save_pars(fl,pars):
    if not ('params' in fl):
        fl.create_group('params')
    for l,m in pars.items():
        if l not in fl['params'].keys():
            fl['params'][l]=m

def save_fields(fl,**kwargs):
    if not ('fields' in fl):
        grp=fl.create_group('fields')
    else:
        grp=fl['fields']
    for l,m in kwargs.items():
        if not l in grp:
            if(np.isscalar(m)):
                grp.create_dataset(l,(1,),maxshape=(None,),dtype=type(m))
            else:                   
                grp.create_dataset(l,(1,)+m.shape,maxshape=(None,)+m.shape,dtype=m.dtype)
        lptr=grp[l]
        lptr.resize((lptr.shape[0]+1,)+lptr.shape[1:])
        lptr[-1,]=m

def save_data(fl,**kwargs):
    if not ('data' in fl):
        grp=fl.create_group('data')
    else:
        grp=fl['data']
    for l,m in kwargs.items():
        if(l not in grp.keys()):
            grp[l]=m
            

class hasegawa_wakatani:
    def __init__(self,**kwargs):
        controls=default_controls.copy()
        params=default_parameters.copy()
        svpars=default_solver_parameters.copy()
        for l,m in kwargs.items():
            if(l in default_controls.keys()):
                controls[l]=m
            elif(l in default_parameters.keys()):
                params[l]=m
            elif(l in default_solver_parameters.keys()):
                svpars[l]=m
            else:
                print(l,'is neither a parameter nor a control flag')
        if('onlydiag' in kwargs.keys() and 'saveresult' not in kwargs.keys() and controls['onlydiag']):
            controls['saveresult']==False
        if(controls['onlydiag'] or controls['wecontinue']):
            fl=h5.File(controls['flname'], 'r+')
            params=load_pars(fl)
        else:
            if(controls['flname']):
                fl=h5.File(controls['flname'],'w')
        Npx,Npy=params['Npx'],params['Npy']
        padx,pady=params['padx'],params['pady']
        Lx,Ly=params['Lx'],params['Ly']
        Nx,Ny=int(Npx/padx/2)*2,int(Npy/pady/2)*2
        if(controls['onlydiag'] or controls['wecontinue']):
            kx=fl['data/kx'][()]
            ky=fl['data/ky'][()]
        else:
            kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
        ksqr=kx**2+ky**2
        lm,nlm,forcelm=init_linmats(params,kx,ky,ksqr)
        uk=np.zeros((2,)+kx.shape,dtype=complex)
        if(controls['onlydiag'] or controls['wecontinue']):
            uk[:]=fl['fields/uk'][-1,]
            t0=fl['fields/t'][-1]
        else:
            init_fields(uk,kx,ky)
            t0=svpars['t0']
        if(controls['saveresult']):
            save_pars(fl,params)
            save_data(fl,kx=kx,ky=ky)
        nthreads=controls['nthreads']
        datk,dat,pf6,pf2=init_ffts(Npx,Npy,nthreads)
        set_num_threads(nthreads)
#        init_solver(params)
#        self.controls=controls
#        self.params=params
        self.kx=kx
        self.ky=ky
        self.ksqr=ksqr
        self.datk=datk
        self.dat=dat
        self.pf6=pf6
        self.pf2=pf2
        self.lm=lm
        self.nlm=nlm
        self.forcelm=forcelm
        self.uk=uk
        self.dukdt=np.zeros_like(uk)
        self.svpars=svpars
        self.controls=controls
        self.fl=fl
        self.t0=t0
#        self.run=self.run_pcvodeg()

    def rhs(self,t,y,dukdt):
        uk=y.view(dtype=complex).reshape(dukdt.shape)
        self.datk.fill(0)
        hermsymznyq(uk)
        multin(uk,self.datk,self.kx,self.ky,self.ksqr)
        self.pf6()
        multout62(self.datk.view(dtype=float))
        self.pf2()
        multvec(dukdt,uk,self.lm,self.datk[:2,],self.nlm,self.forcelm)
        hermsymznyq(dukdt)
        return dukdt.ravel().view(dtype=float)

    def run(self):
        t1,dtstep,dtout,atol,rtol,mxsteps=[self.svpars[l] for l in ['t1','dtstep','dtout','atol','rtol','mxsteps']]
        t0=self.t0
        uk=self.uk
        ct=time()
        uk0=uk.copy()
        if(self.svpars['solver']=='pcvodeg'):
            f = lambda t,y,dydt : self.rhs (t, y, dydt)
            r=pcvodeg(f,self.uk,t0,t1,self.controls['nthreads'],atol=atol,rtol=rtol,mxsteps=mxsteps)
            r.integrate=r.integrate_to
        elif(self.svpars['solver']=='vode'):
            f = lambda t,y : self.rhs (t, y, self.dukdt)
            r=spi.ode(f).set_initial_value(uk0.ravel().view(dtype=float),t0)
            r.set_integrator('vode',atol=atol,rtol=rtol,max_step=dtstep,nsteps=mxsteps)
        elif(self.svpars['solver']=='lsoda'):
            f = lambda t,y : self.rhs (t, y, self.dukdt)
            r=spi.ode(f).set_initial_value(uk0.ravel().view(dtype=float),t0)
            r.set_integrator('lsoda',atol=atol,rtol=rtol,max_step=dtstep,nsteps=mxsteps)
        else:
            print("solver:",self.svcontrol['solver'],'not implemented, using vode instead')
            f = lambda t,y : self.rhs (t, y, self.dukdt)
            r=spi.ode(f).set_initial_value(uk0.ravel().view(dtype=float),t0)
            r.set_integrator('vode',atol=atol,rtol=rtol,max_step=dtstep,nsteps=mxsteps)
        j=0
        toutnext=t0+(j+1)*dtout
        while(r.t<t1):
            r.integrate(r.t+dtstep)
            if(r.t>=toutnext):
                t=toutnext
                print('t='+str(r.t)+', '+str(time()-ct)+" secs elapsed")
                uk[:]=r.y.view(dtype=complex).reshape(uk.shape)
                save_fields(self.fl,uk=uk,t=t)
                j+=1
                toutnext=t0+(j+1)*dtout
        self.fl.close()
