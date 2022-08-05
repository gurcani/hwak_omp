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
from matplotlib.transforms import Bbox
from mpi4py import MPI
comm=MPI.COMM_WORLD

def animate(infl,outfl):
    fl=h5.File(infl,'r',libver='latest',swmr=True)
    uk=fl['fields/uk']
    Lx=fl['params/Lx'][()]
    Ly=fl['params/Ly'][()]
    kx=fl['data/kx'][()]
    ky=fl['data/ky'][()]
    C=fl['params/C'][()]
    kap=fl['params/kap'][()]
    ksqr=kx**2+ky**2
    Nx=uk.shape[1]
    Ny=(uk.shape[2]*2-2)
    dx=Lx/Nx;
    dy=Ly/Ny;
    x=np.arange(0,Nx)*dx
    y=np.arange(0,Ny)*dy
    w, h = plt.figaspect(0.5)
    #ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    fig,ax=plt.subplots(1,2,sharey=True,figsize=(w,h))
    qd=[]
    u0=np.fft.irfft2(-uk[1,0,:,:]*ksqr).T
    u1=np.fft.irfft2(uk[1,1,:,:]).T
    qd.append(ax[0].imshow(u0,cmap='seismic',rasterized=True,vmin=-10,vmax=10))
    qd.append(ax[1].imshow(u1,cmap='seismic',rasterized=True,vmin=-10,vmax=10))
    ax[0].set_title('$\Omega$')
    ax[1].set_title('$n$')
    fig.text(0.45, 0.95,'$C='+str(C)+'$, $\kappa='+str(kap)+'$', fontsize=14)
    t=fl['fields/t'][()]
    Nt=t.shape[0]

    for l in range(len(qd)):
#        fig.colorbar(qd[l],ax=ax[l],format="%.2g", aspect=40,shrink=0.8,pad=0.05)
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
        u0=np.fft.irfft2(-uk[j,0,:,:]*ksqr).T
        u1=np.fft.irfft2(uk[j,1,:,:]).T
        qd[0].set_data(u0)
        qd[1].set_data(u1)
#        vmin,vmax=qd[0].get_clim()
#        qd[0].set_clim(vmin=min(vmin,u0.min()),vmax=max(vmax,u0.max()))
#        vmin,vmax=qd[1].get_clim()
#        qd[1].set_clim(vmin=min(vmin,u1.min()),vmax=max(vmax,u1.max()))
#        fig.savefig("_tmpimg_folder/tmpout%04i"%j+".png",dpi=256,bbox_inches=0.0)
        fig.savefig("_tmpimg_folder/tmpout%04i"%j+".png",dpi=200,bbox_inches=Bbox([[0, 0], [9.6, 4.8]]))
    comm.Barrier()
    if comm.rank==0:
        os.system("ffmpeg -y -i _tmpimg_folder/tmpout%04d.png -c:v libx264 -pix_fmt yuv420p -vf fps=25 "+outfl)
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
