#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:33:00 2018

@author: ogurcan
"""
import sys
import os
import shutil
import numpy as np
import h5py as h5
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pylab as plt
from mpi4py import MPI
comm=MPI.COMM_WORLD

def get_tc(fl,fres):
    tc=None
    for j in fl['fields'].keys():
        if(tc is None):
            if(j[0]=='t'):
                if(len(j[0])==1 or np.char.isdigit(j[1])):
                    if(fres.shape[0]==fl['fields/'+j].shape[0]):
                        tc=fl['fields/'+j][()]
    assert(tc is not None)
    return tc

def animate(infl,outfl):
    fl=h5.File(infl,"r")
    uk=fl['fields/uk']
    Lx=fl['params/Lx'][()]
    Ly=fl['params/Ly'][()]
    Nx=uk.shape[1]
    Ny=(uk.shape[2]*2-2)
    dx=Lx/Nx;
    dy=Ly/Ny;
    x=np.arange(0,Nx)*dx
    y=np.arange(0,Ny)*dy
    w, h = plt.figaspect(0.5)
    fig,ax=plt.subplots(1,2,sharey=True,figsize=(w,h))
    qd=[]
    u0=np.fft.irfft2(uk[1,0,:,:]).T
    u1=np.fft.irfft2(uk[1,1,:,:]).T
    qd.append(ax[0].imshow(np.fft.irfft2(uk[1,0,:,:]).T,cmap='seismic',rasterized=True,vmin=u0.min(),vmax=u0.max()))
    qd.append(ax[1].imshow(np.fft.irfft2(uk[1,1,:,:]).T,cmap='seismic',rasterized=True,vmin=u1.min(),vmax=u1.max()))
    ax[0].set_title('$\Phi$')
    ax[1].set_title('$n$')
    t=fl['fields/t'][()]
    Nt=t.shape[0]

    for l in range(len(qd)):
        fig.colorbar(qd[l],ax=ax[l],format="%.2g", aspect=40,shrink=0.8,pad=0.05)
        ax[l].axis('square')
        fig.tight_layout()
#        qd[l].set_clim(vmin=u0.min(),vmax=u0.max())

    if (comm.rank==0):
        lt=np.arange(Nt)
        lt_loc=np.array_split(lt,comm.size)
        if not os.path.exists('_tmpimg_folder'):
            os.makedirs('_tmpimg_folder')
    else:
        lt_loc=None
    lt_loc=comm.scatter(lt_loc,root=0)
    
    for j in lt_loc:
        print(j)
        u0=np.fft.irfft2(uk[j,0,:,:]).T
        u1=np.fft.irfft2(uk[j,1,:,:]).T
        qd[0].set_data(u0)
        qd[1].set_data(u1)
        vmin,vmax=qd[0].get_clim()
        qd[0].set_clim(vmin=min(vmin,u0.min()),vmax=max(vmax,u0.max()))
        vmin,vmax=qd[1].get_clim()
        qd[1].set_clim(vmin=min(vmin,u1.min()),vmax=max(vmax,u1.max()))
        fig.savefig("_tmpimg_folder/tmpout%04i"%j+".png",dpi=200)#,bbox_inches='tight')
    comm.Barrier()
    if comm.rank==0:
        os.system("ffmpeg -i _tmpimg_folder/tmpout%04d.png -c:v libx264 -vf fps=25 "+outfl)
        shutil.rmtree("_tmpimg_folder")
    
def usage_and_exit():
    if(comm.rank==0):
        print("usage: python diag.py infile.h5 outfile.mp4");
    sys.exit()

if(len(sys.argv)!=3):
    usage_and_exit()
else:
    infl=str(sys.argv[1])
    outfl=str(sys.argv[2])
    if(os.path.isfile(infl)):
        animate(infl,outfl)
    else:
        usage_and_exit()
