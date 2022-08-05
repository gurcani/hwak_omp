import h5py as h5
import numpy as np
#import matplotlib.pylab as plt
import sys,os
sys.path.insert(1, os.path.realpath('../'))
from hwak_omp import hasegawa_wakatani
hw=hasegawa_wakatani(onlydiag=True,flname='out.h5')
fl=hw.fl
uk=fl['fields/uk']
t=fl['fields/t'][()]
kx=fl['data/kx'][()]
ky=fl['data/ky'][()]
ksqr=kx**2+ky**2
kk=np.sqrt(ksqr)
Nx=kx.shape[0]
Ny=ky.shape[0]*2-2
k0=np.min(kk[kk>0])
k1=np.max(kk)
dk=k0
kn=np.arange(k0,k1,dk)
Nk=kn.shape[0]
om=hw.linfreq()
lt=np.arange(-101,0,10)
#lt=-1
gam=om[0].imag[0,:]
inds=[(kk<=kn[l]+dk/2)&(kk>kn[l]-dk/2)&((kx>=0)|(ky>0)) for l in range(Nk)]
indsZ=inds&(ky==0)
indsNZ=inds&(ky>0)
inds=[indsZ,indsNZ]

shp=(np.size(lt),2,Nk)
print(shp)
En=np.zeros(shp)
Fn=np.zeros_like(En)
Norm=Nx**2*Ny**2

#Xk=uk[lt,0,]*uk[lt,1,].conj()

for j in range(np.size(lt)):
    Ik=np.abs(uk[lt[j],0,])**2
    Fk=np.abs(uk[lt[j],1,])**2
    print(j)
    for l in range(Nk):
        for i in range(2): # i.e. zonal and nonzonal components.
            En[j,i,l]=np.sum((Ik*ksqr)[inds[i][l]])/Norm
            Fn[j,i,l]=np.sum(Fk[inds[i][l]])/Norm

fl.close()
fl=h5.File('specdat.h5','w')
fl['En']=En
fl['Fn']=Fn
fl['kn']=kn
fl['gam']=gam
fl['ky']=ky[0,:]
fl.close()
#plt.loglog(kn,En.T)
